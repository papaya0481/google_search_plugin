from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import asyncio
import json
import logging

import aiohttp

from .base import ApiKeyMixin, BaseSearchEngine, SearchResult, mask_api_key

logger = logging.getLogger(__name__)


@dataclass
class TavilyExtractOutcome:
    """Tavily Extract 的规范化结果。"""

    contents: Dict[str, str] = field(default_factory=dict)
    failed_urls: Dict[str, str] = field(default_factory=dict)
    error: str = ""


class TavilyEngine(BaseSearchEngine, ApiKeyMixin):
    """Implementation of the Tavily search engine client."""

    BASE_URL = "https://api.tavily.com"
    SEARCH_ENDPOINT = "/search"
    EXTRACT_ENDPOINT = "/extract"

    search_depth: str
    include_raw_content: bool
    include_answer: bool
    topic: Optional[str]
    turbo: bool

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._init_api_keys(self.config, "TAVILY_API_KEY")
        self.search_depth = self.config.get("search_depth", "basic")
        self.include_raw_content = self.config.get("include_raw_content", True)
        self.include_answer = self.config.get("include_answer", True)

        topic_cfg = self.config.get("topic")
        if isinstance(topic_cfg, str):
            topic_cfg = topic_cfg.strip()
        self.topic = topic_cfg or None
        self.turbo = self.config.get("turbo", False)
        self.last_answer: Optional[str] = None

    async def search(
        self,
        query: str,
        num_results: int,
        *,
        topic: Optional[str] = None,
        force_lightweight: bool = False,
    ) -> List[SearchResult]:
        """Execute a search request via the Tavily API."""
        api_keys = self._iter_api_keys()
        if not api_keys:
            logger.warning("Tavily API key is not configured; skip Tavily search.")
            return []

        self.last_answer = None

        topic_value = self.topic
        if topic is not None:
            topic_value = topic.strip() if isinstance(topic, str) else None
            topic_value = topic_value or None

        request_max_results = min(num_results if num_results > 0 else self.max_results, self.max_results)

        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": self.search_depth,
            "max_results": request_max_results,
            "include_answer": False if force_lightweight else self.include_answer,
            "include_raw_content": False if force_lightweight else self.include_raw_content,
            "topic": topic_value,
            "turbo": self.turbo,
        }

        def _include_value(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
            return True

        payload = {key: value for key, value in payload.items() if _include_value(value)}

        timeout = aiohttp.ClientTimeout(total=self.TIMEOUT)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for api_key in api_keys:
                payload_with_key = dict(payload)
                payload_with_key["api_key"] = api_key

                try:
                    async with session.post(
                        f"{self.BASE_URL}{self.SEARCH_ENDPOINT}",
                        json=payload_with_key,
                        headers=headers,
                        proxy=self.proxy,
                    ) as response:
                        response_text = await response.text()
                        if response.status >= 400:
                            logger.error(
                                "Tavily search request failed with status %s for key %s; response body: %s",
                                response.status,
                                mask_api_key(api_key),
                                response_text,
                            )
                            continue

                        if not response_text:
                            logger.error("Tavily returned an empty response for key %s.", mask_api_key(api_key))
                            continue

                        try:
                            data = json.loads(response_text)
                        except json.JSONDecodeError:
                            logger.error(
                                "Failed to parse Tavily response as JSON for key %s: %s",
                                mask_api_key(api_key),
                                response_text,
                            )
                            continue

                except Exception as exc:
                    logger.error(
                        "Tavily search raised an exception for key %s: %s",
                        mask_api_key(api_key),
                        exc,
                        exc_info=True,
                    )
                    continue

                if isinstance(data, dict):
                    answer = data.get("answer")
                    self.last_answer = answer.strip() if isinstance(answer, str) else None
                else:
                    self.last_answer = None

                results_data = data.get("results", []) if isinstance(data, dict) else []
                results: List[SearchResult] = []

                for index, item in enumerate(results_data):
                    if not isinstance(item, dict):
                        continue

                    title = self.tidy_text(item.get("title", ""))
                    url = item.get("url", "")
                    if not title or not self._is_valid_url(url):
                        continue

                    snippet_source = item.get("content") or item.get("snippet") or item.get("raw_content") or ""
                    snippet = self.tidy_text(snippet_source)
                    content = item.get("raw_content") or snippet
                    score_value = item.get("score")
                    score = float(score_value) if isinstance(score_value, (int, float)) else None

                    results.append(
                        SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            abstract=snippet,
                            rank=index,
                            content=content,
                            score=score,
                        )
                    )

                return results[: min(len(results), num_results)]

        return []

    async def extract(
        self,
        urls: List[str],
        *,
        query: str,
        extract_depth: str = "basic",
        chunks_per_source: int = 3,
        timeout: int = 30,
        max_retries: int = 2,
        max_content_length: int = 3000,
    ) -> TavilyExtractOutcome:
        """抽取指定 URL 与问题相关的正文片段。

        一次重试轮会依次尝试全部 API key。只有超时、限流、服务端错误、
        网络异常或无效响应会进入下一轮；确定性的请求参数错误会立即返回。
        """
        normalized_urls = list(dict.fromkeys(url.strip() for url in urls if url.strip()))
        if not normalized_urls:
            return TavilyExtractOutcome(error="没有可抽取的 URL")

        api_keys = self._iter_api_keys()
        if not api_keys:
            return TavilyExtractOutcome(error="Tavily API key 未配置")

        safe_depth = extract_depth if extract_depth in {"basic", "advanced"} else "basic"
        safe_chunks = min(max(int(chunks_per_source), 1), 5)
        safe_timeout = min(max(int(timeout), 1), 60)
        retry_count = max(int(max_retries), 0)
        payload: Dict[str, Any] = {
            "urls": normalized_urls,
            "query": query,
            "chunks_per_source": safe_chunks,
            "extract_depth": safe_depth,
            "format": "markdown",
            "include_images": False,
            "timeout": safe_timeout,
        }
        last_error = "Tavily Extract 未返回有效结果"

        for retry_index in range(retry_count + 1):
            should_retry = False
            for api_key in api_keys:
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                try:
                    client_timeout = aiohttp.ClientTimeout(total=safe_timeout)
                    async with aiohttp.ClientSession(timeout=client_timeout) as session:
                        async with session.post(
                            f"{self.BASE_URL}{self.EXTRACT_ENDPOINT}",
                            json=payload,
                            headers=headers,
                            proxy=self.proxy,
                        ) as response:
                            response_text = await response.text()
                            if response.status in {401, 403}:
                                last_error = f"Tavily Extract 鉴权失败: HTTP {response.status}"
                                logger.warning("%s, key=%s", last_error, mask_api_key(api_key))
                                continue
                            if response.status == 429 or response.status >= 500:
                                last_error = f"Tavily Extract 临时失败: HTTP {response.status}"
                                logger.warning("%s, key=%s", last_error, mask_api_key(api_key))
                                should_retry = True
                                continue
                            if response.status >= 400:
                                error = f"Tavily Extract 请求失败: HTTP {response.status}: {response_text}"
                                logger.error("%s", error)
                                return TavilyExtractOutcome(error=error)
                            if not response_text:
                                last_error = "Tavily Extract 返回空响应"
                                should_retry = True
                                continue
                            try:
                                data = json.loads(response_text)
                            except json.JSONDecodeError:
                                last_error = "Tavily Extract 返回无法解析的 JSON"
                                should_retry = True
                                continue
                except (TimeoutError, aiohttp.ClientError) as exc:
                    last_error = f"Tavily Extract 请求异常: {exc}"
                    logger.warning("%s, key=%s", last_error, mask_api_key(api_key))
                    should_retry = True
                    continue

                outcome = self._parse_extract_response(data, max_content_length=max_content_length)
                if outcome.contents or outcome.failed_urls:
                    return outcome
                last_error = outcome.error or last_error
                should_retry = True

            if retry_index < retry_count and should_retry:
                await asyncio.sleep(0.5 * (2**retry_index))
                continue
            break

        return TavilyExtractOutcome(error=last_error)

    @staticmethod
    def _parse_extract_response(data: Any, *, max_content_length: int) -> TavilyExtractOutcome:
        """解析 Extract API 响应并限制单个来源的正文长度。"""
        if not isinstance(data, dict):
            return TavilyExtractOutcome(error="Tavily Extract 响应不是对象")

        contents: Dict[str, str] = {}
        failed_urls: Dict[str, str] = {}
        safe_max_length = max(int(max_content_length), 1)
        results_data = data.get("results", [])
        if isinstance(results_data, list):
            for item in results_data:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                raw_content = str(item.get("raw_content") or "").strip()
                if url and raw_content:
                    contents[url] = raw_content[:safe_max_length]

        failed_data = data.get("failed_results", [])
        if isinstance(failed_data, list):
            for item in failed_data:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                reason = str(item.get("error") or item.get("message") or "抽取失败").strip()
                failed_urls[url] = reason

        error = "" if contents or failed_urls else "Tavily Extract 响应没有可用内容"
        return TavilyExtractOutcome(contents=contents, failed_urls=failed_urls, error=error)

    def has_api_keys(self) -> bool:
        """Return True when at least one Tavily API key is available."""
        return bool(self.api_keys)
