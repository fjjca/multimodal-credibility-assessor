import json
import logging

import httpx
import pytest
from prometheus_client import generate_latest

from app import obs
from app.config import Settings
from app.llm.client import LLMClient, LLMResult
from app.main import app
from app.pipeline.retrieve import FakeRetriever
from app.pipeline.runner import run_pipeline


def _hist_value(hist, suffix, labels=None):
    """从 collect() 公开样本里读直方图的 _count/_sum，避免依赖私有桶结构。"""
    labels = labels or {}
    for metric in hist.collect():
        for s in metric.samples:
            if s.name == metric.name + suffix and s.labels == labels:
                return s.value
    return 0.0


def _make_record(msg="hello", **extra):
    record = logging.LogRecord("credibility", logging.INFO, __file__, 1, msg, None, None)
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def test_json_formatter_valid_json_with_extra():
    line = obs.JsonFormatter().format(_make_record("stage done", stage="decompose", latency_s=1.5))
    payload = json.loads(line)
    assert payload["msg"] == "stage done"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "credibility"
    assert payload["stage"] == "decompose"
    assert payload["latency_s"] == 1.5
    assert "ts" in payload


def test_json_formatter_keeps_unicode():
    line = obs.JsonFormatter().format(_make_record("断言拆解"))
    assert "断言拆解" in line  # ensure_ascii=False，中文不落盘成 \uXXXX
    assert json.loads(line)["msg"] == "断言拆解"


def test_setup_logging_idempotent():
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    try:
        obs.setup_logging()
        obs.setup_logging()
        tagged = [h for h in root.handlers if getattr(h, obs._HANDLER_TAG, False)]
        assert len(tagged) == 1
        assert isinstance(tagged[0].formatter, obs.JsonFormatter)
    finally:
        for h in list(root.handlers):
            if getattr(h, obs._HANDLER_TAG, False):
                root.removeHandler(h)
        root.handlers[:] = before_handlers
        root.setLevel(before_level)


def test_record_llm_call_live_counts_tokens_and_latency():
    labels = {"role": "judge", "model": "m-live", "cached": "false"}
    calls0 = obs.LLM_CALLS.labels(**labels)._value.get()
    tokens0 = obs.LLM_TOKENS.labels(role="judge")._value.get()
    lat0 = _hist_value(obs.LLM_LATENCY, "_count", {"role": "judge"})
    sum0 = _hist_value(obs.LLM_LATENCY, "_sum", {"role": "judge"})
    obs.record_llm_call("judge", "m-live", cached=False, tokens=42, latency_ms=1500)
    assert obs.LLM_CALLS.labels(**labels)._value.get() == calls0 + 1
    assert obs.LLM_TOKENS.labels(role="judge")._value.get() == tokens0 + 42
    assert _hist_value(obs.LLM_LATENCY, "_count", {"role": "judge"}) == lat0 + 1
    assert _hist_value(obs.LLM_LATENCY, "_sum", {"role": "judge"}) == pytest.approx(sum0 + 1.5)


def test_record_llm_call_cached_skips_tokens_and_latency():
    labels = {"role": "decompose", "model": "m-cache", "cached": "true"}
    calls0 = obs.LLM_CALLS.labels(**labels)._value.get()
    tokens0 = obs.LLM_TOKENS.labels(role="decompose")._value.get()
    lat0 = _hist_value(obs.LLM_LATENCY, "_count", {"role": "decompose"})
    obs.record_llm_call("decompose", "m-cache", cached=True, tokens=99, latency_ms=10)
    assert obs.LLM_CALLS.labels(**labels)._value.get() == calls0 + 1
    # 缓存命中不重复计费、不计模型延迟
    assert obs.LLM_TOKENS.labels(role="decompose")._value.get() == tokens0
    assert _hist_value(obs.LLM_LATENCY, "_count", {"role": "decompose"}) == lat0


def test_record_stage_histogram():
    c0 = _hist_value(obs.STAGE_LATENCY, "_count", {"stage": "graph"})
    obs.record_stage("graph", 0.25)
    assert _hist_value(obs.STAGE_LATENCY, "_count", {"stage": "graph"}) == c0 + 1


def test_record_verify_counters():
    lat0 = _hist_value(obs.VERIFY_LATENCY, "_count")
    v0 = obs.VERDICTS.labels(verdict="refuted")._value.get()
    e0 = obs.ESCALATIONS._value.get()
    obs.record_verify(latency_s=12.0, verdict="refuted", escalated=True)
    assert _hist_value(obs.VERIFY_LATENCY, "_count") == lat0 + 1
    assert obs.VERDICTS.labels(verdict="refuted")._value.get() == v0 + 1
    assert obs.ESCALATIONS._value.get() == e0 + 1
    # 未升级不增 escalation
    obs.record_verify(latency_s=1.0, verdict="supported", escalated=False)
    assert obs.ESCALATIONS._value.get() == e0 + 1


async def test_metrics_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        res = await c.get("/metrics")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    body = res.text
    assert "# HELP" in body
    for name in ("llm_calls_total", "stage_latency_seconds", "verify_requests_total", "verdicts_total"):
        assert name in body


async def test_client_chat_records_live_and_cached_metrics(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)
    _, model = settings.role_model("decompose")

    async def fake_call_once(provider, model, messages, params):
        return "ok", {"total_tokens": 5}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    live = {"role": "decompose", "model": model, "cached": "false"}
    cached = {"role": "decompose", "model": model, "cached": "true"}
    live0 = obs.LLM_CALLS.labels(**live)._value.get()
    cached0 = obs.LLM_CALLS.labels(**cached)._value.get()
    tokens0 = obs.LLM_TOKENS.labels(role="decompose")._value.get()

    msgs = [{"role": "user", "content": "obs-metrics-probe"}]
    r1 = await client.chat("decompose", msgs)
    r2 = await client.chat("decompose", msgs)
    assert not r1.cached and r2.cached
    assert obs.LLM_CALLS.labels(**live)._value.get() == live0 + 1
    assert obs.LLM_CALLS.labels(**cached)._value.get() == cached0 + 1
    assert obs.LLM_TOKENS.labels(role="decompose")._value.get() == tokens0 + 5  # 只计一次


async def test_runner_records_stage_metrics_not_serving(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path), trace_dir=str(tmp_path)
    )
    client = LLMClient(settings)

    async def fake_chat(role, messages, **kw):
        sys = messages[0]["content"]
        usage = {"total_tokens": 10}
        if "断言拆解器" in sys:
            return LLMResult(content='[{"text":"断言A"}]', model="m", usage=usage)
        if "证据关系判定器" in sys:
            return LLMResult(content='[]', model="m", usage=usage)
        if role == "prosecutor":
            return LLMResult(content='{"argument":"虚假"}', model="m", usage=usage)
        if role == "defender":
            return LLMResult(content='{"argument":"真实"}', model="m", usage=usage)
        if role == "judge":
            return LLMResult(
                content='{"verdict":"supported","confidence":0.9,"reason":"ok","evidence_chain":[]}',
                model="m", usage=usage,
            )
        return LLMResult(content="[]", model="m", usage=usage)

    monkeypatch.setattr(client, "chat", fake_chat)

    dec0 = _hist_value(obs.STAGE_LATENCY, "_count", {"stage": "decompose"})
    adj0 = _hist_value(obs.STAGE_LATENCY, "_count", {"stage": "adjudicate"})
    verify0 = _hist_value(obs.VERIFY_LATENCY, "_count")

    async def emit(event, data):
        pass

    result = await run_pipeline(
        "某帖子", None, emit, settings=settings, client=client,
        retriever=FakeRetriever([]), trace_id="obs1",
    )
    assert result["verdict"] == "supported"
    assert _hist_value(obs.STAGE_LATENCY, "_count", {"stage": "decompose"}) == dec0 + 1
    assert _hist_value(obs.STAGE_LATENCY, "_count", {"stage": "adjudicate"}) == adj0 + 1
    # 流水线内部埋点不污染 serving 级指标（record_verify 只在 api._run）
    assert _hist_value(obs.VERIFY_LATENCY, "_count") == verify0


def test_metrics_payload_matches_exposition():
    body, content_type = obs.metrics_payload()
    assert isinstance(body, bytes)
    assert content_type.startswith("text/plain")
    assert body == generate_latest()