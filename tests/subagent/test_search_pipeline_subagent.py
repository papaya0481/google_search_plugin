from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
import sys

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
TEST_PACKAGE = "google_search_pipeline_under_test"


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    spec = spec_from_file_location(module_name, PLUGIN_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载测试模块: {relative_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_search_pipeline_module() -> ModuleType:
    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    sys.modules[TEST_PACKAGE] = package
    pipelines_name = f"{TEST_PACKAGE}.pipelines"
    pipelines_package = ModuleType(pipelines_name)
    pipelines_package.__path__ = [str(PLUGIN_ROOT / "pipelines")]  # type: ignore[attr-defined]
    sys.modules[pipelines_name] = pipelines_package
    _load_module(f"{pipelines_name}._envelope", "pipelines/_envelope.py")
    _load_module(f"{pipelines_name}.llm_runner", "pipelines/llm_runner.py")
    _load_module(f"{pipelines_name}.prompts", "pipelines/prompts.py")
    return _load_module(f"{pipelines_name}.search_pipeline", "pipelines/search_pipeline.py")


search_pipeline_module = _load_search_pipeline_module()
SearchPipeline = search_pipeline_module.SearchPipeline


class FakeEngineChain:
    def __init__(self, last_success_engine: str) -> None:
        self.last_success_engine = last_success_engine
        self.last_tavily_answer = "legacy Tavily answer"
        self.calls: list[dict[str, Any]] = []

    async def search_with_fallback(self, question: str, max_results: int, **kwargs: Any) -> list[Any]:
        self.calls.append({"question": question, "max_results": max_results, **kwargs})
        return [
            SimpleNamespace(
                title="Example",
                url="https://example.com/a",
                abstract="摘要",
                snippet="摘要",
            )
        ]


class FakeFetcher:
    def __init__(self) -> None:
        self.integrated = False
        self.fetched = False

    def integrate_inline_content(self, results: list[Any], answer: str) -> None:
        del results, answer
        self.integrated = True

    async def fetch_batch(self, results: list[Any], *, last_success_engine: str) -> list[Any]:
        del last_success_engine
        self.fetched = True
        return results


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        assert prompt
        self.calls += 1
        return "旧流程总结"


class FakeSubagent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, question: str, results: list[Any], *, bot_name: str) -> str:
        self.calls.append({"question": question, "results": results, "bot_name": bot_name})
        return "subagent 结果"


def build_pipeline(*, enabled: bool, engine_name: str) -> tuple[Any, Any, Any, Any, Any]:
    engines = FakeEngineChain(engine_name)
    fetcher = FakeFetcher()
    llm = FakeLLM()
    subagent = FakeSubagent()
    pipeline = SearchPipeline(
        backend_cfg=SimpleNamespace(max_results=5, fetch_content=True),
        engine_chain=engines,
        content_fetcher=fetcher,
        llm_runner=llm,
        tavily_subagent_config=SimpleNamespace(enabled=enabled),
        tavily_subagent=subagent,
    )
    return pipeline, engines, fetcher, llm, subagent


@pytest.mark.asyncio
async def test_enabled_tavily_path_uses_lightweight_search_and_subagent() -> None:
    pipeline, engines, fetcher, llm, subagent = build_pipeline(enabled=True, engine_name="tavily")

    result = await pipeline.run("问题", bot_name="麦麦")

    assert result == "subagent 结果"
    assert engines.calls[0]["tavily_force_lightweight"] is True
    assert len(subagent.calls) == 1
    assert fetcher.integrated is False
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_disabled_tavily_path_preserves_legacy_summary() -> None:
    pipeline, engines, fetcher, llm, subagent = build_pipeline(enabled=False, engine_name="tavily")

    result = await pipeline.run("问题", bot_name="麦麦")

    assert result == "旧流程总结"
    assert engines.calls[0]["tavily_force_lightweight"] is False
    assert fetcher.integrated is True
    assert llm.calls == 1
    assert not subagent.calls


@pytest.mark.asyncio
async def test_tavily_fallback_to_bing_preserves_non_tavily_path() -> None:
    pipeline, engines, fetcher, llm, subagent = build_pipeline(enabled=True, engine_name="bing")

    result = await pipeline.run("问题", bot_name="麦麦")

    assert result == "旧流程总结"
    assert engines.calls[0]["tavily_force_lightweight"] is True
    assert fetcher.fetched is True
    assert llm.calls == 1
    assert not subagent.calls
