import json

import httpx
import pytest

from app.config import Settings
from app.llm.cache import ResponseCache
from app.llm.client import LLMClient
from app.pipeline.decompose import decompose, parse_claims
from app.pipeline.models import Claim, Evidence
from app.pipeline.retrieve import (
    FakeRetriever,
    TavilyRetriever,
    _parse_tavily,
    retrieve_evidence,
)


def test_parse_claims_json_array():
    claims = parse_claims('[{"text":"甲"},{"text":"乙"}]', 5)
    assert [c.text for c in claims] == ["甲", "乙"]
    assert [c.id for c in claims] == ["c1", "c2"]


def test_parse_claims_fences_dedup_limit():
    content = '```json\n[{"text":"甲"},{"text":"甲"},{"text":"乙"},{"text":"丙"}]\n```'
    claims = parse_claims(content, 2)
    assert [c.text for c in claims] == ["甲", "乙"]


def test_parse_claims_bad_json_empty():
    assert parse_claims("not json", 5) == []


async def test_decompose_with_mock_llm(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)

    async def fake(provider, model, messages, params):
        return '[{"text":"断言A"},{"text":"断言B"}]', {}

    monkeypatch.setattr(client, "_call_once", fake)
    claims, result = await decompose(client, "某帖子", max_claims=5)
    assert [c.text for c in claims] == ["断言A", "断言B"]


async def test_decompose_fallback_to_whole_text(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)

    async def fake(provider, model, messages, params):
        return "[]", {}

    monkeypatch.setattr(client, "_call_once", fake)
    claims, _ = await decompose(client, "  单一事实帖子  ", max_claims=5)
    assert len(claims) == 1
    assert claims[0].text == "单一事实帖子"


def test_parse_tavily_tiers_and_score():
    raw = {
        "results": [
            {"title": "官方", "url": "https://www.gov.cn/a", "content": "x", "score": 0.9},
            {"title": "社交", "url": "https://reddit.com/b", "content": "y", "score": "bad"},
        ]
    }
    ev = _parse_tavily(raw, 3)
    assert ev[0].source_tier == "official"
    assert ev[0].score == pytest.approx(0.9)
    assert ev[1].source_tier == "social"
    assert ev[1].score == 0.0


async def test_retrieve_evidence_dedup_and_ids():
    preset = [
        Evidence(id="", title="共享", url="https://x.com/a", content="a"),
        Evidence(id="", title="独有", url="https://y.com/b", content="b"),
    ]
    retriever = FakeRetriever(preset)
    claims = [Claim(id="c1", text="q1"), Claim(id="c2", text="q2")]
    evidences, claim_map = await retrieve_evidence(retriever, claims, top_k=3)
    # 两条断言检索到相同 URL，去重后共 2 条证据
    assert len(evidences) == 2
    assert [e.id for e in evidences] == ["e1", "e2"]
    assert claim_map["c1"] == ["e1", "e2"]
    assert claim_map["c2"] == ["e1", "e2"]
    assert retriever.queries == ["q1", "q2"]


async def test_retrieve_evidence_appends_image_hint():
    retriever = FakeRetriever([])
    claims = [Claim(id="c1", text="q1"), Claim(id="c2", text="q2")]
    await retrieve_evidence(retriever, claims, top_k=3, hints={"c1": "炒鸡肉"})
    assert retriever.queries == ["q1 炒鸡肉", "q2"]


async def test_tavily_retriever_cache_hit(tmp_path):
    cache = ResponseCache(str(tmp_path), "readwrite")
    raw = {"results": [{"title": "T", "url": "https://gov.cn/x", "content": "C", "score": 0.9}]}
    key = ResponseCache.key("tavily", "search", [{"role": "user", "content": "q"}], {"top_k": 3})
    cache.put(key, json.dumps(raw), {}, "tavily-search")
    ev = await TavilyRetriever("k", cache=cache).search("q", 3)
    assert ev[0].source_tier == "official"


async def test_tavily_retriever_network_then_cache(tmp_path, monkeypatch):
    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"title": "T", "url": "https://bbc.com/a", "content": "C", "score": 0.5}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    r = TavilyRetriever("k", cache=ResponseCache(str(tmp_path), "readwrite"))
    ev = await r.search("q", 3)
    assert ev[0].source_tier == "mainstream"
    ev2 = await r.search("q", 3)  # 命中缓存，不再走网络
    assert ev2[0].url == ev[0].url