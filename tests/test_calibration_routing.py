import json

import pytest

from app.config import Settings
from app.eval.calibration import load_calibrator
from app.llm.client import LLMClient, LLMResult
from app.pipeline.adjudicate import adjudicate
from app.pipeline.graph import build_graph
from app.pipeline.models import Claim


def _artifact(path, configured="zhipu:glm-4-flash", actual=None):
    path.write_text(json.dumps({
        "temperature": 3.0,
        "model": {"configured": configured, "actual_models": actual or ["glm-4-flash"]},
    }), encoding="utf-8")


def test_versioned_calibrator_ready_and_mismatch(tmp_path):
    p = tmp_path / "cal.json"
    _artifact(p)
    ready = load_calibrator(str(p), configured_model="zhipu:glm-4-flash", actual_model="glm-4-flash")
    assert ready["status"] == "ready" and ready["temperature"] == 3.0
    mismatch = load_calibrator(str(p), configured_model="zhipu:other", actual_model="glm-4-flash")
    assert mismatch["status"] == "model_mismatch" and mismatch["temperature"] == 1.0
    fallback = load_calibrator(str(p), configured_model="zhipu:glm-4-flash", actual_model="fallback-model")
    assert fallback["status"] == "model_mismatch"


def test_legacy_calibrator_is_not_silently_used(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text('{"temperature": 2.5}', encoding="utf-8")
    result = load_calibrator(str(p), configured_model="zhipu:glm-4-flash")
    assert result["status"] == "legacy_unversioned"
    assert result["temperature"] == 1.0


async def test_calibrated_confidence_drives_escalation(tmp_path, monkeypatch):
    artifact = tmp_path / "cal.json"
    _artifact(artifact)
    settings = Settings(
        _env_file=None,
        llm_cache_mode="off",
        llm_cache_dir=str(tmp_path),
        enable_calibration=True,
        calibrator_path=str(artifact),
        model_judge="zhipu:glm-4-flash",
        model_judge_strong="zhipu:glm-4.5-flash",
        judge_confidence_threshold=0.8,
        enable_debate=False,
    )
    client = LLMClient(settings)
    calls = []

    async def fake_chat(role, messages, **kw):
        calls.append(kw.get("model_override"))
        if kw.get("model_override"):
            return LLMResult(content='{"verdict":"supported","confidence":0.9,"reason":"强模型"}', model="glm-4.5-flash")
        return LLMResult(content='{"verdict":"supported","confidence":0.9,"reason":"原始高置信"}', model="glm-4-flash")

    monkeypatch.setattr(client, "chat", fake_chat)
    claims = [Claim(id="c1", text="x")]
    g = build_graph(claims, [], [])
    verdict, results = await adjudicate(client, claims, g, enable_debate=False)
    assert calls == [None, "zhipu:glm-4.5-flash"]
    assert verdict["escalated"] is True
    assert verdict["confidence_raw"] == pytest.approx(0.9)
    assert verdict["calibration_status"] == "model_mismatch"
    assert len(results) == 2


async def test_model_mismatch_does_not_change_routing(tmp_path, monkeypatch):
    artifact = tmp_path / "cal.json"
    _artifact(artifact, configured="zhipu:other")
    settings = Settings(
        _env_file=None,
        llm_cache_mode="off",
        llm_cache_dir=str(tmp_path),
        enable_calibration=True,
        calibrator_path=str(artifact),
        model_judge="zhipu:glm-4-flash",
        model_judge_strong="zhipu:glm-4.5-flash",
        judge_confidence_threshold=0.8,
    )
    client = LLMClient(settings)
    calls = []

    async def fake_chat(role, messages, **kw):
        calls.append(role)
        return LLMResult(content='{"verdict":"supported","confidence":0.9,"reason":"高置信"}', model="glm-4-flash")

    monkeypatch.setattr(client, "chat", fake_chat)
    verdict, _ = await adjudicate(
        client, [Claim(id="c1", text="x")], build_graph([Claim(id="c1", text="x")], [], []),
        enable_debate=False,
    )
    assert calls == ["judge"]
    assert verdict["confidence"] == pytest.approx(0.9)
    assert verdict["calibration_status"] == "model_mismatch"
