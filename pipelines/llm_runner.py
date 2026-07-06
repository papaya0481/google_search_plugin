"""LLM 调用包装。

设计要点:
- 必须显式传 ``model=`` 参数,否则 host 端 ``resolve_task_name("")``
  会按字母序回退到 ``embedding`` task,导致 chat completion 失败。
- 区分"调用失败"(异常 / success=False / 超时)与"模型返空响应":
  前者抛 :class:`LLMCallError`,调用方据此给出"服务暂不可用"文案;
  后者返回空字符串,调用方给出"无法确定"文案。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List
import asyncio
import logging

from ._envelope import peel_envelope

if TYPE_CHECKING:
    from maibot_sdk import PluginContext

    from ..config import ModelsSection

logger = logging.getLogger(__name__)


@dataclass
class LLMToolResponse:
    """一次带私有工具的 LLM 响应。"""

    response: str = ""
    reasoning: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class LLMCallError(RuntimeError):
    """LLM 调用层失败(异常/success=False/超时)。

    与"模型自然返回空字符串"区分开,让上层 pipeline 能给出更准确的用户文案。
    """


class LLMRunner:
    """简单的 LLM 调用包装器。

    持有 ``ctx`` + ``ModelsSection`` 配置,封装"传 prompt → 拿 string"的流程。
    """

    def __init__(self, ctx: "PluginContext", model_config: "ModelsSection") -> None:
        self._ctx = ctx
        self._config = model_config

    async def generate(self, prompt: str) -> str:
        """生成文本。

        Args:
            prompt: 完整 prompt 字符串

        Returns:
            str: LLM 响应文本(成功时);模型返空时返回 ""

        Raises:
            LLMCallError: LLM 调用层失败(异常 / success=False / 超时)
        """
        if not prompt or not prompt.strip():
            logger.warning("prompt 为空,跳过 LLM 调用")
            return ""

        target_model = str(self._config.model_name or "replyer")
        temperature = self._config.temperature
        timeout = max(int(self._config.llm_timeout_seconds or 60), 1)
        logger.info(
            "调用 ctx.llm.generate, model=%s temperature=%s prompt_len=%d timeout=%ds",
            target_model,
            temperature,
            len(prompt),
            timeout,
        )

        try:
            result = await asyncio.wait_for(
                self._ctx.llm.generate(
                    prompt=prompt,
                    model=target_model,            # 必须显式传,空字符串会被 host 回退到 embedding
                    temperature=temperature,
                    # 必须显式传给 RPC 层:不传时 Runner 默认 30s 超时,
                    # 外层 wait_for 的配置超时根本轮不到生效(曾致 summarize 30s 必炸)
                    timeout_ms=timeout * 1000,
                ),
                timeout=timeout + 5,  # 外层只做兜底,略宽于 RPC 超时避免抢跑
            )
        except asyncio.TimeoutError as exc:
            logger.error("ctx.llm.generate 超时(%ds, model=%s)", timeout, target_model)
            raise LLMCallError(f"LLM 调用超时 ({timeout}s)") from exc
        except Exception as exc:
            logger.error("ctx.llm.generate 抛异常: %s", exc, exc_info=True)
            raise LLMCallError(f"LLM 调用异常: {exc}") from exc

        # SDK 2.4 / 新版 Runner 会多包一层 {"success": True, "result": {...}}
        # 信封,SDK 的 _normalize_capability_result 没剥干净,这里手动剥。
        result = peel_envelope(result)

        if not isinstance(result, dict):
            logger.warning("ctx.llm.generate 返回非 dict: type=%s value=%r", type(result).__name__, result)
            raise LLMCallError(f"LLM 返回非 dict: {type(result).__name__}")

        success = bool(result.get("success", False))
        response_text = str(result.get("response") or "")
        if not success:
            err = result.get("error") or "<no error key>"
            logger.error(
                "LLM 调用失败 (model=%s): error=%s | full_result_keys=%s",
                target_model,
                err,
                sorted(result.keys()),
            )
            raise LLMCallError(f"LLM 调用失败 (model={target_model}): {err}")

        if not response_text:
            # 模型自然返回空 —— 不抛异常,让上层判断这是"无内容"还是"无法判断"
            logger.warning(
                "LLM 调用 success=True 但 response 为空 (model=%s) full_result_keys=%s",
                target_model,
                sorted(result.keys()),
            )
            return ""

        preview = response_text[:200].replace("\n", "\\n")
        logger.info(
            "LLM 响应成功 (model=%s) response_len=%d preview=%r",
            target_model,
            len(response_text),
            preview,
        )
        return response_text.strip()

    async def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> LLMToolResponse:
        """执行一次带私有工具定义的 LLM 调用。

        重试由调用方管理，以便将实际请求次数与 subagent 决策轮次分离。
        """
        if not messages:
            raise LLMCallError("带工具 LLM 调用缺少消息")

        target_model = self._config.model_name
        temperature = self._config.temperature
        timeout = max(int(self._config.llm_timeout_seconds), 1)
        logger.info(
            "调用 ctx.llm.generate_with_tools, model=%s messages=%d tools=%d timeout=%ds",
            target_model,
            len(messages),
            len(tools),
            timeout,
        )
        try:
            result = await asyncio.wait_for(
                self._ctx.llm.generate_with_tools(
                    prompt=messages,
                    tools=tools,
                    model=target_model,
                    temperature=temperature,
                    timeout_ms=timeout * 1000,
                ),
                timeout=timeout + 5,
            )
        except asyncio.TimeoutError as exc:
            raise LLMCallError(f"带工具 LLM 调用超时 ({timeout}s)") from exc
        except Exception as exc:
            raise LLMCallError(f"带工具 LLM 调用异常: {exc}") from exc

        result = peel_envelope(result)
        if not isinstance(result, dict):
            raise LLMCallError(f"带工具 LLM 返回非 dict: {type(result).__name__}")
        if not bool(result.get("success", False)):
            raise LLMCallError(f"带工具 LLM 调用失败: {result.get('error') or '<no error key>'}")

        response_text = str(result.get("response") or "").strip()
        reasoning = str(result.get("reasoning") or "").strip()
        raw_tool_calls = result.get("tool_calls")
        if raw_tool_calls is not None and not isinstance(raw_tool_calls, list):
            raise LLMCallError("带工具 LLM 的 tool_calls 不是数组")
        if isinstance(raw_tool_calls, list) and any(not isinstance(item, dict) for item in raw_tool_calls):
            raise LLMCallError("带工具 LLM 的 tool_calls 包含非对象项")
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) else []
        if not response_text and not tool_calls:
            raise LLMCallError("带工具 LLM 返回空响应且没有工具调用")
        return LLMToolResponse(response=response_text, reasoning=reasoning, tool_calls=tool_calls)
