from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any
import sys

import pytest


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE = "google_search_plugin_under_test"


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    """在不依赖插件部署目录名称的前提下加载相对导入模块。"""
    spec = spec_from_file_location(module_name, PLUGIN_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载测试模块: {relative_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_tavily_module() -> ModuleType:
    # base.py 的 HTML 解析依赖不会参与 Tavily 客户端测试，提供最小模块接口即可。
    if "bs4" not in sys.modules:
        bs4_module = ModuleType("bs4")
        bs4_module.BeautifulSoup = object
        bs4_module.MarkupResemblesLocatorWarning = Warning
        sys.modules["bs4"] = bs4_module

    package = ModuleType(TEST_PACKAGE)
    package.__path__ = [str(PLUGIN_ROOT)]  # type: ignore[attr-defined]
    sys.modules[TEST_PACKAGE] = package
    search_package_name = f"{TEST_PACKAGE}.search_engines"
    search_package = ModuleType(search_package_name)
    search_package.__path__ = [str(PLUGIN_ROOT / "search_engines")]  # type: ignore[attr-defined]
    sys.modules[search_package_name] = search_package

    _load_module(f"{search_package_name}.base", "search_engines/base.py")
    return _load_module(f"{search_package_name}.tavily", "search_engines/tavily.py")


tavily_module = _load_tavily_module()
TavilyEngine = tavily_module.TavilyEngine


class FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback

    async def text(self) -> str:
        return self._body


class FakeSession:
    def __init__(self, responses: list[FakeResponse], requests: list[dict[str, Any]], **kwargs: Any) -> None:
        del kwargs
        self._responses = responses
        self._requests = requests

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self._requests.append({"url": url, **kwargs})
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_search_preserves_tavily_score(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        FakeResponse(
            200,
            '{"results":[{"title":"Example","url":"https://example.com/a",'
            '"content":"query snippet","score":0.875}]}',
        )
    ]
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tavily_module.aiohttp,
        "ClientSession",
        lambda **kwargs: FakeSession(responses, requests, **kwargs),
    )
    engine = TavilyEngine({"api_key": "tvly-test", "max_results": 5})
    monkeypatch.setattr(engine, "_iter_api_keys", lambda: ["tvly-test"])

    results = await engine.search("example query", 5)

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.875)
    assert results[0].content == "query snippet"
    assert requests[0]["json"]["include_answer"] is True
    assert requests[0]["json"]["include_raw_content"] is True


@pytest.mark.asyncio
async def test_lightweight_search_ignores_legacy_content_options(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [FakeResponse(200, '{"results":[]}')]
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tavily_module.aiohttp,
        "ClientSession",
        lambda **kwargs: FakeSession(responses, requests, **kwargs),
    )
    engine = TavilyEngine(
        {
            "api_key": "tvly-test",
            "include_answer": True,
            "include_raw_content": True,
        }
    )
    monkeypatch.setattr(engine, "_iter_api_keys", lambda: ["tvly-test"])

    await engine.search("example query", 5, force_lightweight=True)

    assert requests[0]["json"]["include_answer"] is False
    assert requests[0]["json"]["include_raw_content"] is False


@pytest.mark.asyncio
async def test_extract_retries_transient_failure_and_limits_content(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        FakeResponse(500, "temporary"),
        FakeResponse(
            200,
            '{"results":[{"url":"https://example.com/a","raw_content":"abcdefgh"}],'
            '"failed_results":[]}',
        ),
    ]
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        tavily_module.aiohttp,
        "ClientSession",
        lambda **kwargs: FakeSession(responses, requests, **kwargs),
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(tavily_module.asyncio, "sleep", no_sleep)
    engine = TavilyEngine({"api_key": "tvly-test"})
    monkeypatch.setattr(engine, "_iter_api_keys", lambda: ["tvly-test"])

    outcome = await engine.extract(
        ["https://example.com/a"],
        query="example query",
        max_retries=2,
        max_content_length=5,
    )

    assert outcome.contents == {"https://example.com/a": "abcde"}
    assert outcome.error == ""
    assert len(requests) == 2
    assert requests[0]["json"]["query"] == "example query"
    assert requests[0]["json"]["extract_depth"] == "basic"
    assert requests[0]["headers"]["Authorization"] == "Bearer tvly-test"


def test_parse_extract_response_keeps_per_url_failures() -> None:
    outcome = TavilyEngine._parse_extract_response(
        {
            "results": [{"url": "https://example.com/a", "raw_content": "content"}],
            "failed_results": [{"url": "https://example.com/b", "error": "blocked"}],
        },
        max_content_length=100,
    )

    assert outcome.contents == {"https://example.com/a": "content"}
    assert outcome.failed_urls == {"https://example.com/b": "blocked"}
