from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
TEST_PACKAGE = "google_search_subagent_under_test"


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    spec = spec_from_file_location(module_name, PLUGIN_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载测试模块: {relative_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_subagent_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    sys.modules[TEST_PACKAGE] = package

    pipelines_name = f"{TEST_PACKAGE}.pipelines"
    pipelines_package = ModuleType(pipelines_name)
    pipelines_package.__path__ = [str(PLUGIN_ROOT / "pipelines")]  # type: ignore[attr-defined]
    sys.modules[pipelines_name] = pipelines_package
    _load_module(f"{pipelines_name}._envelope", "pipelines/_envelope.py")
    llm_module = _load_module(f"{pipelines_name}.llm_runner", "pipelines/llm_runner.py")
    subagent_module = _load_module(f"{pipelines_name}.search_subagent", "pipelines/search_subagent.py")

    if "bs4" not in sys.modules:
        bs4_module = ModuleType("bs4")
        bs4_module.BeautifulSoup = object
        bs4_module.MarkupResemblesLocatorWarning = Warning
        sys.modules["bs4"] = bs4_module
    search_engines_name = f"{TEST_PACKAGE}.search_engines"
    search_package = ModuleType(search_engines_name)
    search_package.__path__ = [str(PLUGIN_ROOT / "search_engines")]  # type: ignore[attr-defined]
    sys.modules[search_engines_name] = search_package
    base_module = _load_module(f"{search_engines_name}.base", "search_engines/base.py")
    return subagent_module, llm_module, base_module


subagent_module, llm_module, base_module = _load_subagent_modules()
TavilySearchSubagent = subagent_module.TavilySearchSubagent
LLMCallError = llm_module.LLMCallError
LLMToolResponse = llm_module.LLMToolResponse
SearchResult = base_module.SearchResult


def tool_response(name: str, arguments: dict[str, Any], *, call_id: str = "call-1") -> Any:
    return LLMToolResponse(
        tool_calls=[
            {
                "id": call_id,
                "function": {"name": name, "arguments": arguments},
            }
        ]
    )


class FakeLLMRunner:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        self.calls.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeTavilyEngine:
    def __init__(self, outcome: Any = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outcome = outcome

    async def extract(self, urls: list[str], **kwargs: Any) -> Any:
        self.calls.append({"urls": urls, **kwargs})
        if self.outcome is not None:
            return self.outcome
        return SimpleNamespace(
            contents={urls[0]: "抽取到的正文"},
            failed_urls={},
            error="",
        )


def build_config(**overrides: Any) -> Any:
    values = {
        "max_rounds": 4,
        "llm_max_retries": 2,
        "max_extract_calls": 2,
        "extract_max_retries": 2,
        "extract_max_urls": 3,
        "extract_depth": "basic",
        "extract_chunks_per_source": 3,
        "extract_timeout_seconds": 30,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def build_results() -> list[Any]:
    return [
        SearchResult(
            title="Example",
            url="https://example.com/a",
            snippet="初始摘要",
            abstract="初始摘要",
            score=0.9,
        )
    ]


@pytest.mark.asyncio
async def test_extract_then_finish_summary_with_canonical_source() -> None:
    llm = FakeLLMRunner(
        [
            tool_response("extract", {"result_ids": ["r1"]}),
            tool_response("finish", {"mode": "summary", "answer": "最终答案", "result_ids": ["r1"]}),
        ]
    )
    tavily = FakeTavilyEngine()
    agent = TavilySearchSubagent(
        config=build_config(),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=tavily,
        llm_runner=llm,
    )

    result = await agent.run("问题", build_results(), bot_name="麦麦")

    assert result == "最终答案\n\n来源：\n- [Example](https://example.com/a)"
    assert tavily.calls[0]["urls"] == ["https://example.com/a"]
    assert "抽取到的正文" in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_llm_retries_do_not_consume_decision_round(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLMRunner(
        [
            LLMCallError("timeout-1"),
            LLMCallError("timeout-2"),
            tool_response("finish", {"mode": "summary", "answer": "重试成功", "result_ids": ["r1"]}),
        ]
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(subagent_module.asyncio, "sleep", no_sleep)
    agent = TavilySearchSubagent(
        config=build_config(max_rounds=1),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=FakeTavilyEngine(),
        llm_runner=llm,
    )

    result = await agent.run("问题", build_results(), bot_name="麦麦")

    assert result.startswith("重试成功")
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_round_exhaustion_forces_summary_without_extra_round() -> None:
    llm = FakeLLMRunner(
        [
            LLMToolResponse(response="我暂时不调用动作"),
            tool_response("finish", {"mode": "summary", "answer": "无有关信息", "result_ids": ["r1"]}),
        ]
    )
    agent = TavilySearchSubagent(
        config=build_config(max_rounds=1),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=FakeTavilyEngine(),
        llm_runner=llm,
    )

    result = await agent.run("问题", build_results(), bot_name="麦麦")

    assert result.startswith("无有关信息")
    assert len(llm.calls) == 2
    assert [tool["name"] for tool in llm.calls[1]["tools"]] == ["finish"]


@pytest.mark.asyncio
async def test_forced_summary_retry_exhaustion_returns_initial_results(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = FakeLLMRunner(
        [
            LLMToolResponse(response="没有动作"),
            LLMCallError("forced-1"),
            LLMCallError("forced-2"),
            LLMCallError("forced-3"),
        ]
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(subagent_module.asyncio, "sleep", no_sleep)
    agent = TavilySearchSubagent(
        config=build_config(max_rounds=1),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=FakeTavilyEngine(),
        llm_runner=llm,
    )

    result = await agent.run("问题", build_results(), bot_name="麦麦")

    assert result.startswith("强制总结失败，返回初始 Tavily 搜索资料：")
    assert "https://example.com/a" in result
    assert len(llm.calls) == 4


@pytest.mark.asyncio
async def test_finish_results_returns_canonical_search_data() -> None:
    llm = FakeLLMRunner([tool_response("finish", {"mode": "results", "result_ids": ["r1"]})])
    agent = TavilySearchSubagent(
        config=build_config(),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=FakeTavilyEngine(),
        llm_runner=llm,
    )

    result = await agent.run("问题", build_results(), bot_name="麦麦")

    assert result == "搜索资料：\n- [Example](https://example.com/a)，相关度 0.9000\n  初始摘要"


@pytest.mark.asyncio
async def test_invalid_result_id_consumes_round_then_forces_summary() -> None:
    llm = FakeLLMRunner(
        [
            tool_response("finish", {"mode": "summary", "answer": "伪造来源", "result_ids": ["r99"]}),
            tool_response("finish", {"mode": "summary", "answer": "无有关信息", "result_ids": ["r1"]}),
        ]
    )
    agent = TavilySearchSubagent(
        config=build_config(max_rounds=1),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=FakeTavilyEngine(),
        llm_runner=llm,
    )

    result = await agent.run("问题", build_results(), bot_name="麦麦")

    assert result.startswith("无有关信息")
    assert len(llm.calls) == 2
    assert [tool["name"] for tool in llm.calls[1]["tools"]] == ["finish"]


@pytest.mark.asyncio
async def test_extract_failure_is_formatted_without_python_dict_repr() -> None:
    llm = FakeLLMRunner(
        [
            tool_response("extract", {"result_ids": ["r1"]}),
            tool_response("finish", {"mode": "summary", "answer": "无有关信息", "result_ids": ["r1"]}),
        ]
    )
    tavily = FakeTavilyEngine(
        SimpleNamespace(
            contents={},
            failed_urls={"https://example.com/a": "blocked"},
            error="",
        )
    )
    agent = TavilySearchSubagent(
        config=build_config(),
        backend_config=SimpleNamespace(max_content_length=3000),
        tavily_engine=tavily,
        llm_runner=llm,
    )

    await agent.run("问题", build_results(), bot_name="麦麦")

    tool_result = llm.calls[1]["messages"][-1]["content"]
    assert tool_result == "Extract 失败：\n- [r1] Example：blocked\n请基于现有摘要调用 finish。"
    assert "{'https://example.com/a': 'blocked'}" not in tool_result
