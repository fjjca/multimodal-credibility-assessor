import base64

import httpx
import pytest
from openai import BadRequestError

from app.config import Settings
from app.llm.client import LLMClient, LLMResult
from app.pipeline.evidence_graph import build_evidence_graph
from app.pipeline.forensics import (
    IMAGE_EVIDENCE_ID,
    forensics,
    image_evidence,
    image_relations,
    load_image_url,
    parse_forensics,
)
from app.pipeline.models import Claim, Evidence, ForensicsResult, normalize_consistency

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _bad_request_error():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    return BadRequestError("content policy", response=resp, body=None)


def test_normalize_consistency_synonyms():
    assert normalize_consistency("consistent") == "consistent"
    assert normalize_consistency("矛盾") == "inconsistent"
    assert normalize_consistency("contradicts") == "inconsistent"
    assert normalize_consistency("无法判断") == "uncertain"
    assert normalize_consistency("garbage") == "uncertain"
    assert normalize_consistency("") == "uncertain"


def test_parse_forensics_valid():
    claims = [Claim(id="c1", text="a"), Claim(id="c2", text="b")]
    content = (
        '{"ocr_text":"图中文字","description":"一只猫","overall":"inconsistent",'
        '"claims":[{"claim_id":"c1","consistency":"inconsistent","note":"地点不符"},'
        '{"claim_id":"c2","consistency":"consistent"},{"claim_id":"c9","consistency":"consistent"}],'
        '"notes":"疑似拼接"}'
    )
    fr = parse_forensics(content, claims)
    assert fr.ocr_text == "图中文字"
    assert fr.overall == "inconsistent"
    assert fr.notes == "疑似拼接"
    assert len(fr.claims) == 2  # c9 被过滤
    assert fr.claims[0].consistency == "inconsistent"
    assert fr.claims[1].consistency == "consistent"


def test_parse_forensics_bad_json_defaults():
    fr = parse_forensics("not json", [Claim(id="c1", text="a")])
    assert fr.overall == "uncertain"
    assert fr.claims == []
    assert fr.ocr_text == ""


def test_llm_parsers_tolerate_raw_control_chars():
    # VLM 会在 JSON 字符串值里输出原始换行等控制字符；strict=False 必须容忍，
    # 否则解析静默降级（mm05/mm06 取证失败的根因）。覆盖全部 5 个 LLM 输出解析器。
    from app.eval.schema import parse_baseline_output
    from app.pipeline.adjudicate import parse_judge_output
    from app.pipeline.decompose import parse_claims
    from app.pipeline.evidence_graph import parse_relations

    claims = [Claim(id="c1", text="限速60")]
    fr = parse_forensics(
        '{"ocr_text":"SPEED LIMIT\n60","overall":"inconsistent",'
        '"claims":[{"claim_id":"c1","consistency":"inconsistent"}]}',
        claims,
    )
    assert fr.ocr_text == "SPEED LIMIT\n60"
    assert fr.overall == "inconsistent"
    assert fr.claims[0].consistency == "inconsistent"

    assert parse_claims('[{"text":"限速\n60"}]', 5)[0].text == "限速\n60"

    rels = parse_relations('[{"evidence_id":"e1","relation":"support","note":"a\nb"}]', "c1", {"e1"})
    assert len(rels) == 1 and rels[0].relation == "support"

    judge = parse_judge_output('{"verdict":"refuted","confidence":0.8,"reason":"r\ns"}')
    assert judge["verdict"] == "refuted" and judge["confidence"] == pytest.approx(0.8)

    verdict, conf, reason = parse_baseline_output('{"verdict":"refuted","confidence":0.7,"reason":"r\ns"}')
    assert verdict == "refuted" and conf == pytest.approx(0.7) and reason == "r\ns"


def test_image_evidence_shape():
    fr = ForensicsResult(ocr_text="文字", description="描述", notes="备注")
    ev = image_evidence(fr)
    assert ev.id == IMAGE_EVIDENCE_ID
    assert ev.source_tier == "social"
    assert ev.url == ""
    assert "文字" in ev.content and "描述" in ev.content and "备注" in ev.content


def test_image_relations_mapping():
    claims = [Claim(id="c1", text="a"), Claim(id="c2", text="b"), Claim(id="c3", text="c")]
    fr = ForensicsResult(claims=[
        {"claim_id": "c1", "consistency": "inconsistent"},
        {"claim_id": "c2", "consistency": "consistent"},
    ])
    rels = image_relations(fr, claims)
    by_claim = {r.claim_id: r.relation for r in rels}
    assert by_claim == {"c1": "contradict", "c2": "support", "c3": "neutral"}
    assert all(r.evidence_id == IMAGE_EVIDENCE_ID for r in rels)


def test_load_image_url_local_png(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(PNG_1PX)
    url = load_image_url(str(p))
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_1PX


def test_load_image_url_passthrough_and_missing(tmp_path):
    assert load_image_url("https://example.com/a.jpg") == "https://example.com/a.jpg"
    assert load_image_url("data:image/png;base64,abc") == "data:image/png;base64,abc"
    assert load_image_url(str(tmp_path / "nope.png")) is None
    assert load_image_url(None) is None
    txt = tmp_path / "x.txt"
    txt.write_text("hi")
    assert load_image_url(str(txt)) is None


def _client(tmp_path, monkeypatch, chat_fn):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)
    monkeypatch.setattr(client, "chat", chat_fn)
    return client


async def test_forensics_no_image_returns_none(tmp_path, monkeypatch):
    async def chat_fn(role, messages, **kw):
        raise AssertionError("无图不应调用 VLM")

    client = _client(tmp_path, monkeypatch, chat_fn)
    fr, res = await forensics(client, None, "text", [Claim(id="c1", text="a")])
    assert fr is None and res is None


async def test_forensics_success_builds_vision_message(tmp_path, monkeypatch):
    seen = {}

    async def chat_fn(role, messages, **kw):
        seen["role"] = role
        seen["messages"] = messages
        seen["extra_body"] = kw.get("extra_body")
        return LLMResult(
            content='{"ocr_text":"o","description":"d","overall":"inconsistent","claims":[{"claim_id":"c1","consistency":"inconsistent"}]}',
            model="glm-4.6v-flash",
            usage={"total_tokens": 50},
        )

    client = _client(tmp_path, monkeypatch, chat_fn)
    fr, res = await forensics(client, "data:image/png;base64,abc", "帖子", [Claim(id="c1", text="a")])
    assert seen["role"] == "forensics"
    assert seen["extra_body"] == {"thinking": {"type": "disabled"}}
    content = seen["messages"][1]["content"]
    assert isinstance(content, list)
    assert any(b["type"] == "image_url" for b in content)
    assert fr.overall == "inconsistent"
    assert fr.claims[0].consistency == "inconsistent"
    assert res.model == "glm-4.6v-flash"


async def test_forensics_content_policy_degrades(tmp_path, monkeypatch):
    async def chat_fn(role, messages, **kw):
        raise _bad_request_error()

    client = _client(tmp_path, monkeypatch, chat_fn)
    fr, res = await forensics(client, "data:image/png;base64,abc", "t", [Claim(id="c1", text="a")])
    assert res is None
    assert fr is not None
    assert "审核" in fr.notes
    assert fr.overall == "uncertain"


async def test_image_evidence_creates_contradict_edge(tmp_path, monkeypatch):
    async def chat_fn(role, messages, **kw):
        # 网页证据全部 neutral，矛盾只来自图像
        return LLMResult(content='[{"evidence_id":"e1","relation":"neutral"}]', model="m")

    client = _client(tmp_path, monkeypatch, chat_fn)
    claims = [Claim(id="c1", text="某断言")]
    web = [Evidence(id="e1", title="t", url="https://x.com/a", source_tier="social")]
    fr = ForensicsResult(overall="inconsistent", claims=[{"claim_id": "c1", "consistency": "inconsistent"}])
    g, rels, _ = await build_evidence_graph(
        client, claims, web, {"c1": ["e1"]},
        image_evidence=image_evidence(fr), image_relations=image_relations(fr, claims),
    )
    assert g.has_edge("c1", IMAGE_EVIDENCE_ID)
    assert g.edges["c1", IMAGE_EVIDENCE_ID]["relation"] == "contradict"
    assert g.nodes[IMAGE_EVIDENCE_ID]["kind"] == "evidence"
    # 图像 url 为空，不应产生同源边
    assert not g.has_edge("e1", IMAGE_EVIDENCE_ID)