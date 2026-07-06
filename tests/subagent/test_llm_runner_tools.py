from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
TEST_PACKAGE = "google_search_llm_runner_under_test"


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    spec = spec_from_file_location(module_name, PLUGIN_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载测试模块: {relative_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_llm_runner_module() -> ModuleType:
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    sys.modules[TEST_PACKAGE] = package
    pipelines_name = f"{TEST_PACKAGE}.pipelines"
    pipelines_package = ModuleType(pipelines_name)
    pipelines_package.__path__ = [str(PLUGIN_ROOT / "pipelines")]  # type: ignore[attr-defined]
    sys.modules[pipelines_name] = pipelines_package
    _load_module(f"{pipelines_name}._envelope", "pipelines/_envelope.py")
    return _load_module(f"{pipelines_name}.llm_runner", "pipelines/llm_runner.py")


llm_runner_module = _load_llm_runner_module()
LLMCallError = llm_runner_module.LLMCallError
LLMRunner = llm_runner_module.LLMRunner


class FakeLLMCapability:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


def build_runner(result: Any) -> tuple[Any, Any]:
    capability = FakeLLMCapability(result)
    context = SimpleNamespace(llm=capability)
    config = SimpleNamespace(model_name="replyer", temperature=0.2, llm_timeout_seconds=5)
    return LLMRunner(context, config), capability


@pytest.mark.asyncio
async def test_generate_with_tools_preserves_valid_tool_calls() -> None:
    runner, capability = build_runner(
        {
            "success": True,
            "response": "",
            "reasoning": "需要结束",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "finish", "arguments": {"mode": "results", "result_ids": ["r1"]}},
                }
            ],
        }
    )

    response = await runner.generate_with_tools(
        [{"role": "user", "content": "问题"}],
        [{"name": "finish", "parameters_schema": {"type": "object"}}],
    )

    assert response.tool_calls[0]["function"]["name"] == "finish"
    assert capability.calls[0]["model"] == "replyer"
    assert capability.calls[0]["timeout_ms"] == 5000


@pytest.mark.asyncio
async def test_generate_with_tools_rejects_malformed_tool_calls() -> None:
    runner, _ = build_runner(
        {
            "success": True,
            "response": "看起来像正常文本",
            "tool_calls": "not-a-list",
        }
    )

    with pytest.raises(LLMCallError, match="tool_calls 不是数组"):
        await runner.generate_with_tools(
            [{"role": "user", "content": "问题"}],
            [{"name": "finish", "parameters_schema": {"type": "object"}}],
        )
