import json
from typing import Protocol

import httpx

from app.llm.cache import ResponseCache
from app.pipeline.models import Claim, Evidence, classify_source_tier


class Retriever(Protocol):
    async def search(self, query: str, top_k: int) -> list[Evidence]: ...


def _parse_tavily(raw: dict, top_k: int) -> list[Evidence]:
    out = []
    for r in (raw.get("results") or [])[:top_k]:
        url = r.get("url", "")
        try:
            score = float(r.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        out.append(
            Evidence(
                id="",
                title=r.get("title", ""),
                url=url,
                content=r.get("content", ""),
                source_tier=classify_source_tier(url),
                score=score,
            )
        )
    return out


class TavilyRetriever:
    """Tavily 搜索 API（免费档）。响应缓存复用 LLM 缓存机制，支持离线回放。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.tavily.com",
        cache: ResponseCache | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.timeout = timeout

    async def search(self, query: str, top_k: int) -> list[Evidence]:
        key = ResponseCache.key(
            "tavily", "search", [{"role": "user", "content": query}], {"top_k": top_k}
        )
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return _parse_tavily(json.loads(hit["content"]), top_k)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "max_results": top_k, "search_depth": "basic"},
            )
            resp.raise_for_status()
            raw = resp.json()
        if self.cache is not None:
            self.cache.put(key, json.dumps(raw, ensure_ascii=False), {}, "tavily-search")
        return _parse_tavily(raw, top_k)


class FakeRetriever:
    """离线测试用：返回预设证据，记录查询。"""

    def __init__(self, results: list[Evidence] | None = None):
        self.results = results or []
        self.queries: list[str] = []

    async def search(self, query: str, top_k: int) -> list[Evidence]:
        self.queries.append(query)
        return [e.model_copy() for e in self.results[:top_k]]


async def retrieve_evidence(
    retriever: Retriever, claims: list[Claim], top_k: int, hints: dict[str, str] | None = None
) -> tuple[list[Evidence], dict[str, list[str]]]:
    """逐断言检索，按 URL 去重并分配全局证据 id，返回 (证据列表, 断言->证据id 映射)。

    hints 按 claim_id 追加检索关键词（图像 OCR/描述），避免只拿断言原文检索出
    关于事件的网页冒充支持证据。
    """
    seen: dict[str, str] = {}
    evidences: list[Evidence] = []
    claim_map: dict[str, list[str]] = {}
    for c in claims:
        ids = []
        query = c.text
        hint = (hints or {}).get(c.id, "").strip()
        if hint:
            query = f"{c.text} {hint}".strip()
        for e in await retriever.search(query, top_k):
            dedup = e.url or e.title
            if dedup in seen:
                ids.append(seen[dedup])
                continue
            eid = f"e{len(evidences) + 1}"
            evidences.append(e.model_copy(update={"id": eid}))
            seen[dedup] = eid
            ids.append(eid)
        claim_map[c.id] = ids
    return evidences, claim_map