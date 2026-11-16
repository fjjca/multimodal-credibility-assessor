import json

import pytest

from app.config import Settings
from app.eval.baseline import SYSTEM_FORCE, BaselinePredictor
from app.eval.dataset import load_jsonl
from app.eval.harness import compute_metrics, run_eval
from app.eval.schema import ABSTAIN, Prediction, Sample, parse_baseline_output, parse_verdict
from app.llm.client import LLMClient


def test_parse_verdict_synonyms():
    assert parse_verdict("supported") == "supported"
    assert parse_verdict("真实") == "supported"
    assert parse_verdict("FALSE") == "refuted"
    assert parse_verdict("谣言") == "refuted"
    assert parse_verdict("out-of-context") == "misleading"
    assert parse_verdict("误导") == "misleading"
    assert parse_verdict("无法验证") == ABSTAIN
    assert parse_verdict("garbage-unknown") == ABSTAIN


def test_parse_baseline_output_handles_fences():
    content = '```json\n{"verdict":"refuted","confidence":0.8,"reason":"r"}\n```'
    verdict, conf, reason = parse_baseline_output(content)
    assert verdict == "refuted" and conf == 0.8 and reason == "r"


def test_parse_baseline_output_bad_json_abstains():
    verdict, conf, reason = parse_baseline_output("not json at all")
    assert verdict == ABSTAIN and conf is None


def test_compute_metrics_hand_case():
    records = [
        {"gold": "supported", "verdict": "supported", "usage": {}, "latency_ms": 0, "cached": False},
        {"gold": "refuted", "verdict": "refuted", "usage": {}, "latency_ms": 0, "cached": False},
        {"gold": "misleading", "verdict": ABSTAIN, "usage": {}, "latency_ms": 0, "cached": False},
        {"gold": "supported", "verdict": "refuted", "usage": {}, "latency_ms": 0, "cached": False},
    ]
    m = compute_metrics(records)
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["abstention_rate"] == pytest.approx(0.25)
    assert m["accuracy_on_attempted"] == pytest.approx(2 / 3)
    assert m["per_class"]["supported"]["recall"] == pytest.approx(0.5)
    assert m["per_class"]["refuted"]["precision"] == pytest.approx(0.5)
    assert m["per_class"]["misleading"]["f1"] == 0.0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)


def test_compute_metrics_macro_over_present_classes():
    records = [
        {"gold": "supported", "verdict": "supported", "usage": {}, "latency_ms": 0, "cached": False},
        {"gold": "refuted", "verdict": "refuted", "usage": {}, "latency_ms": 0, "cached": False},
    ]
    m = compute_metrics(records)
    # misleading 类在 gold 中缺席，不应拉低 macro-F1
    assert m["macro_f1"] == pytest.approx(1.0)


def test_load_seed_dataset():
    samples = load_jsonl("data/eval_seed.jsonl")
    assert len(samples) >= 10
    assert all(s.label in ("supported", "refuted", "misleading") for s in samples)


async def test_harness_end_to_end(tmp_path):
    samples = load_jsonl("data/eval_seed.jsonl")[:5]

    async def perfect(sample):
        return Prediction(verdict=sample.label)

    summary = await run_eval(
        samples, perfect, system_name="test", dataset_name="seed", out_dir=tmp_path
    )
    assert summary["metrics"]["accuracy"] == 1.0
    assert summary["metrics"]["abstention_rate"] == 0.0
    files = list(tmp_path.iterdir())
    assert any(f.suffix == ".json" for f in files)
    assert any(f.suffix == ".jsonl" for f in files)
    assert any(f.suffix == ".md" for f in files)


async def test_harness_abstention(tmp_path):
    samples = load_jsonl("data/eval_seed.jsonl")[:3]

    async def partial(sample):
        return Prediction(verdict=ABSTAIN if sample.id == "s03" else sample.label)

    summary = await run_eval(
        samples, partial, system_name="test", dataset_name="seed", out_dir=tmp_path
    )
    assert summary["metrics"]["abstention_rate"] == pytest.approx(1 / 3)


async def test_harness_failure_does_not_abort(tmp_path):
    samples = load_jsonl("data/eval_seed.jsonl")[:4]

    async def flaky(sample):
        if sample.id == "s02":
            raise RuntimeError("429 rate limit")
        return Prediction(verdict=sample.label)

    summary = await run_eval(
        samples, flaky, system_name="test", dataset_name="seed", out_dir=tmp_path
    )
    assert summary["n_failed"] == 1
    assert summary["failed"][0]["id"] == "s02"
    assert summary["metrics"]["n"] == 3
    assert summary["metrics"]["accuracy"] == 1.0


async def test_harness_resume_skips_done(tmp_path):
    samples = load_jsonl("data/eval_seed.jsonl")[:4]
    calls = []

    async def flaky(sample):
        calls.append(sample.id)
        if sample.id == "s02":
            raise RuntimeError("429 rate limit")
        return Prediction(verdict=sample.label)

    await run_eval(samples, flaky, system_name="test", dataset_name="seed", out_dir=tmp_path)
    assert len(calls) == 4

    calls.clear()

    async def fixed(sample):
        calls.append(sample.id)
        return Prediction(verdict=sample.label)

    summary = await run_eval(
        samples, fixed, system_name="test", dataset_name="seed", out_dir=tmp_path
    )
    # 只补跑上一轮失败的 s02，其余从 checkpoint 读取
    assert calls == ["s02"]
    assert summary["n_failed"] == 0
    assert summary["metrics"]["n"] == 4
    assert summary["metrics"]["accuracy"] == 1.0


async def test_baseline_predictor_with_mock_llm(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)

    async def fake_call_once(provider, model, messages, params):
        return '{"verdict":"supported","confidence":0.9,"reason":"ok"}', {"total_tokens": 42}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    sample = Sample(id="t", text="x", label="supported")
    pred = await BaselinePredictor(client)(sample)
    assert pred.verdict == "supported"
    assert pred.confidence == 0.9
    assert pred.usage["total_tokens"] == 42


async def test_force_choice_prompt_forbids_abstention(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path))
    client = LLMClient(settings)
    seen = {}

    async def fake_call_once(provider, model, messages, params):
        seen["system"] = messages[0]["content"]
        return '{"verdict":"refuted","confidence":0.7,"reason":"r"}', {}

    monkeypatch.setattr(client, "_call_once", fake_call_once)
    pred = await BaselinePredictor(client, force_choice=True)(
        Sample(id="t", text="x", label="refuted")
    )
    assert seen["system"] == SYSTEM_FORCE
    assert "禁止弃权" in seen["system"]
    assert pred.verdict == "refuted"


async def test_pipeline_predictor_offline(tmp_path, monkeypatch):
    from app.eval.pipeline import PipelinePredictor
    from app.llm.client import LLMResult
    from app.pipeline.models import Evidence
    from app.pipeline.retrieve import FakeRetriever

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
            return LLMResult(content='[{"evidence_id":"e1","relation":"support"}]', model="m", usage=usage)
        if role == "judge":
            return LLMResult(
                content='{"verdict":"supported","confidence":0.9,"reason":"r","evidence_chain":["e1"]}',
                model="m", usage=usage,
            )
        return LLMResult(content='{"argument":"x"}', model="m", usage=usage)

    monkeypatch.setattr(client, "chat", fake_chat)
    retriever = FakeRetriever([Evidence(id="", title="t", url="https://gov.cn/a", source_tier="official")])
    predictor = PipelinePredictor(settings=settings, client=client, retriever=retriever)
    pred = await predictor(Sample(id="t", text="帖子", label="supported"))
    assert pred.verdict == "supported"
    assert pred.usage["total_tokens"] > 0
    assert pred.latency_ms >= 0