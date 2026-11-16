import pytest

from app.config import Settings
from app.llm.client import LLMClient, LLMResult
from app.pipeline.adjudicate import (
    _graph_context,
    adjudicate,
    parse_judge_output,
)
from app.pipeline.evidence_graph import build_evidence_graph, parse_relations
from app.pipeline.graph import build_graph
from app.pipeline.models import Claim, Evidence, Relation


def test_parse_relations_filters_and_dedup():
    content = '[{"evidence_id":"e1","relation":"support"},{"evidence_id":"e1","relation":"contradict"},{"evidence_id":"e9","relation":"support"},{"evidence_id":"e2","relation":"bogus"}]'
    rels = parse_relations(content, "c1", {"e1", "e2"})
    assert len(rels) == 1
    assert rels[0].evidence_id == "e1"
    assert rels[0].relation == "support"


def test_parse_relations_bad_json():
    assert parse_relations("nope", "c1", {"e1"}) == []


def _graph():
    claims = [Claim(id="c1", text="某断言")]
    evidences = [
        Evidence(id="e1", title="反驳", url="https://gov.cn/a", content="反证", source_tier="official"),
        Evidence(id="e2", title="支持", url="https://reddit.com/b", content="佐证", source_tier="social"),
    ]
    relations = [
        Relation(claim_id="c1", evidence_id="e1", relation="contradict"),
        Relation(claim_id="c1", evidence_id="e2", relation="support"),
    ]
    return claims, build_graph(claims, evidences, relations)


def test_graph_context_empty():
    claims = [Claim(id="c1", text="x")]
    g = build_graph(claims, [], [])
    assert "没有任何" in _graph_context(g)


def test_graph_context_has_relations():
    _, g = _graph()
    ctx = _graph_context(g)
    assert "矛盾" in ctx and "支持" in ctx
    assert "official" in ctx


def test_parse_judge_output():
    out = parse_judge_output('{"verdict":"refuted","confidence":0.8,"reason":"r","evidence_chain":["e1"]}')
    assert out["verdict"] == "refuted"
    assert out["confidence"] == pytest.approx(0.8)
    assert out["evidence_chain"] == ["e1"]


def test_parse_judge_output_bad_abstains():
    out = parse_judge_output("garbage")
    assert out["verdict"] == "unverifiable"
    assert out["confidence"] is None


def _client(tmp_path, monkeypatch, judge_content):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)
    calls = []

    async def fake_chat(role, messages, **kw):
        calls.append(role)
        if role == "prosecutor":
            return LLMResult(content='{"argument":"内容虚假","key_evidence":["e1"]}', model="m")
        if role == "defender":
            return LLMResult(content='{"argument":"内容真实","key_evidence":["e2"]}', model="m")
        if role == "judge":
            return LLMResult(content=judge_content, model="m")
        return LLMResult(content="[]", model="m")

    monkeypatch.setattr(client, "chat", fake_chat)
    return client, calls


async def test_adjudicate_three_roles(tmp_path, monkeypatch):
    claims, g = _graph()
    client, calls = _client(
        tmp_path, monkeypatch, '{"verdict":"refuted","confidence":0.7,"reason":"反证","evidence_chain":["e1"]}'
    )
    verdict, results = await adjudicate(client, claims, g, enable_debate=True)
    assert calls == ["prosecutor", "defender", "judge"]
    assert verdict["verdict"] == "refuted"
    assert verdict["evidence_chain"] == ["e1"]
    assert "虚假" in verdict["prosecutor"]
    assert len(results) == 3


async def test_adjudicate_no_debate_ablation(tmp_path, monkeypatch):
    claims, g = _graph()
    client, calls = _client(
        tmp_path, monkeypatch, '{"verdict":"unverifiable","confidence":0.5,"reason":"均衡"}'
    )
    client.settings.enable_escalation = False  # 隔离辩论消融，不触发置信度升级
    verdict, results = await adjudicate(client, claims, g, enable_debate=False)
    assert calls == ["judge"]
    assert verdict["verdict"] == "unverifiable"
    assert len(results) == 1


async def test_build_evidence_graph_stage(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)

    async def fake_chat(role, messages, **kw):
        return LLMResult(content='[{"evidence_id":"e1","relation":"contradict"}]', model="m")

    monkeypatch.setattr(client, "chat", fake_chat)
    claims = [Claim(id="c1", text="x")]
    evidences = [Evidence(id="e1", title="t", url="https://gov.cn/a", source_tier="official")]
    g, rels, results = await build_evidence_graph(client, claims, evidences, {"c1": ["e1"]})
    assert len(rels) == 1
    assert g.edges["c1", "e1"]["relation"] == "contradict"
    assert len(results) == 1


def _bad_request_error():
    import httpx
    from openai import BadRequestError

    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    return BadRequestError("content policy", response=resp, body=None)


async def test_adjudicate_content_policy_degrades(tmp_path, monkeypatch):
    claims, g = _graph()
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)

    async def fake_chat(role, messages, **kw):
        raise _bad_request_error()

    monkeypatch.setattr(client, "chat", fake_chat)
    verdict, results = await adjudicate(client, claims, g, enable_debate=True)
    assert verdict["verdict"] == "unverifiable"
    assert len(results) == 3  # 控辩裁各降级一次，不崩溃


async def test_classify_relations_content_policy_empty(tmp_path, monkeypatch):
    from app.pipeline.evidence_graph import classify_relations

    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)

    async def fake_chat(role, messages, **kw):
        raise _bad_request_error()

    monkeypatch.setattr(client, "chat", fake_chat)
    rels, res = await classify_relations(
        client, Claim(id="c1", text="x"), [Evidence(id="e1", title="t", url="https://x.com")]
    )
    assert rels == []
    assert res is None


def _routing_client(tmp_path, monkeypatch, first_content, strong_content, calls, **overrides):
    base = dict(
        model_judge="zhipu:glm-4-flash",
        model_judge_strong="zhipu:glm-4.5-flash",
        judge_confidence_threshold=0.6,
    )
    base.update(overrides)
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path), **base)
    client = LLMClient(settings)

    async def fake_chat(role, messages, **kw):
        calls.append((role, kw.get("model_override")))
        if role == "judge":
            if kw.get("model_override"):
                return LLMResult(content=strong_content, model="glm-4.5-flash")
            return LLMResult(content=first_content, model="glm-4-flash")
        return LLMResult(content='{"argument":"a"}', model="m")

    monkeypatch.setattr(client, "chat", fake_chat)
    return client


_LOW = '{"verdict":"unverifiable","confidence":0.4,"reason":"弱模型不确定"}'
_STRONG = '{"verdict":"refuted","confidence":0.9,"reason":"强模型反证","evidence_chain":["e1"]}'


async def test_routing_escalates_on_low_confidence(tmp_path, monkeypatch):
    claims, g = _graph()
    calls = []
    client = _routing_client(tmp_path, monkeypatch, _LOW, _STRONG, calls)
    verdict, results = await adjudicate(client, claims, g, enable_debate=False)
    assert verdict["escalated"] is True
    assert verdict["verdict"] == "refuted"
    assert verdict["confidence"] == pytest.approx(0.9)
    assert verdict["judge_model"] == "glm-4.5-flash"
    assert calls == [("judge", None), ("judge", "zhipu:glm-4.5-flash")]
    assert len(results) == 2


async def test_routing_no_escalation_on_high_confidence(tmp_path, monkeypatch):
    claims, g = _graph()
    calls = []
    high = '{"verdict":"supported","confidence":0.9,"reason":"确信"}'
    client = _routing_client(tmp_path, monkeypatch, high, _STRONG, calls)
    verdict, results = await adjudicate(client, claims, g, enable_debate=False)
    assert verdict["escalated"] is False
    assert verdict["verdict"] == "supported"
    assert calls == [("judge", None)]
    assert len(results) == 1


async def test_routing_skips_when_same_model(tmp_path, monkeypatch):
    claims, g = _graph()
    calls = []
    client = _routing_client(
        tmp_path, monkeypatch, _LOW, _STRONG, calls, model_judge_strong="zhipu:glm-4-flash"
    )
    verdict, results = await adjudicate(client, claims, g, enable_debate=False)
    assert verdict["escalated"] is False
    assert calls == [("judge", None)]  # 强弱同模型，不重复调用
    assert len(results) == 1


async def test_routing_disabled(tmp_path, monkeypatch):
    claims, g = _graph()
    calls = []
    client = _routing_client(
        tmp_path, monkeypatch, _LOW, _STRONG, calls, enable_escalation=False
    )
    verdict, results = await adjudicate(client, claims, g, enable_debate=False)
    assert verdict["escalated"] is False
    assert verdict["verdict"] == "unverifiable"
    assert len(results) == 1