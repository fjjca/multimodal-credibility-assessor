import json

import pytest

from app.config import Settings
from app.eval.calibration import apply_temperature
from app.llm.client import LLMClient, LLMResult
from app.pipeline.models import Evidence
from app.pipeline.retrieve import FakeRetriever
from app.pipeline.runner import run_pipeline


def _mock_client(tmp_path, monkeypatch, calls):
    settings = Settings(
        _env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path), trace_dir=str(tmp_path)
    )
    client = LLMClient(settings)

    async def fake_chat(role, messages, **kw):
        calls.append(role)
        sys = messages[0]["content"]
        usage = {"total_tokens": 10}
        if "断言拆解器" in sys:
            return LLMResult(content='[{"text":"断言A"}]', model="m", usage=usage)
        if "证据关系判定器" in sys:
            return LLMResult(content='[{"evidence_id":"e1","relation":"support"}]', model="m", usage=usage)
        if role == "prosecutor":
            return LLMResult(content='{"argument":"内容虚假"}', model="m", usage=usage)
        if role == "defender":
            return LLMResult(content='{"argument":"内容真实"}', model="m", usage=usage)
        if role == "judge":
            return LLMResult(
                content='{"verdict":"supported","confidence":0.9,"reason":"证据支持","evidence_chain":["e1"]}',
                model="m", usage=usage,
            )
        return LLMResult(content="[]", model="m", usage=usage)

    monkeypatch.setattr(client, "chat", fake_chat)
    return settings, client


def _retriever():
    return FakeRetriever(
        [Evidence(id="", title="官方佐证", url="https://gov.cn/a", content="支持内容", source_tier="official")]
    )


async def test_runner_end_to_end(tmp_path, monkeypatch):
    calls = []
    settings, client = _mock_client(tmp_path, monkeypatch, calls)
    events = []

    async def emit(event, data):
        events.append((event, data))

    result = await run_pipeline(
        "某帖子", None, emit, settings=settings, client=client, retriever=_retriever(), trace_id="t1"
    )
    assert result["verdict"] == "supported"
    assert result["verdict_label"] == "真实可信"
    assert result["evidence_chain"][0]["id"] == "e1"
    assert result["graph"]["n_support"] == 1
    assert result["usage"]["total_tokens"] > 0
    assert calls == ["decompose", "decompose", "prosecutor", "defender", "judge"]

    names = [n for n, _ in events]
    assert names[-1] == "result"
    statuses = {d["stage"]: d["status"] for n, d in events if n == "stage" and d["status"] != "running"}
    assert statuses["forensics"] == "skipped"
    assert statuses["adjudicate"] == "done"

    assert (tmp_path / "t1.jsonl").exists()
    assert (tmp_path / "t1_graph.json").exists()
    trace = [json.loads(x) for x in (tmp_path / "t1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {t["stage"] for t in trace} == {"decompose", "forensics", "retrieve", "graph", "adjudicate"}


async def test_runner_no_retrieval_ablation(tmp_path, monkeypatch):
    calls = []
    settings, client = _mock_client(tmp_path, monkeypatch, calls)
    settings.enable_retrieval = False
    retriever = _retriever()
    events = []

    async def emit(event, data):
        events.append((event, data))

    result = await run_pipeline(
        "某帖子", None, emit, settings=settings, client=client, retriever=retriever, trace_id="t2"
    )
    assert result["graph"]["n_evidence"] == 0
    assert result["graph"]["n_support"] == 0
    assert retriever.queries == []  # 检索被关闭，未发起查询


async def test_runner_no_debate_ablation(tmp_path, monkeypatch):
    calls = []
    settings, client = _mock_client(tmp_path, monkeypatch, calls)
    settings.enable_debate = False
    events = []

    async def emit(event, data):
        events.append((event, data))

    await run_pipeline(
        "某帖子", None, emit, settings=settings, client=client, retriever=_retriever(), trace_id="t3"
    )
    assert "prosecutor" not in calls
    assert "defender" not in calls
    assert "judge" in calls


async def test_runner_with_image_forensics(tmp_path, monkeypatch):
    import base64

    png = tmp_path / "img.png"
    png.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    ))
    settings = Settings(
        _env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path),
        trace_dir=str(tmp_path), enable_escalation=False,
    )
    client = LLMClient(settings)
    calls = []

    async def fake_chat(role, messages, **kw):
        calls.append(role)
        sys = messages[0]["content"]
        usage = {"total_tokens": 10}
        if "断言拆解器" in sys:
            return LLMResult(content='[{"text":"断言A"}]', model="m", usage=usage)
        if "多模态取证员" in sys:
            return LLMResult(
                content='{"ocr_text":"o","description":"d","overall":"inconsistent","claims":[{"claim_id":"c1","consistency":"inconsistent","note":"不符"}]}',
                model="v", usage=usage,
            )
        if "证据关系判定器" in sys:
            return LLMResult(content='[{"evidence_id":"e1","relation":"neutral"}]', model="m", usage=usage)
        if role == "prosecutor":
            return LLMResult(content='{"argument":"虚假"}', model="m", usage=usage)
        if role == "defender":
            return LLMResult(content='{"argument":"真实"}', model="m", usage=usage)
        if role == "judge":
            return LLMResult(
                content='{"verdict":"refuted","confidence":0.8,"reason":"图文矛盾","evidence_chain":["img1"]}',
                model="m", usage=usage,
            )
        return LLMResult(content="[]", model="m", usage=usage)

    monkeypatch.setattr(client, "chat", fake_chat)
    events = []

    async def emit(event, data):
        events.append((event, data))

    result = await run_pipeline(
        "某帖子", str(png), emit, settings=settings, client=client, retriever=_retriever(), trace_id="timg"
    )
    assert "forensics" in calls
    assert result["forensics"]["overall"] == "inconsistent"
    assert result["graph"]["n_contradict"] == 1
    assert result["verdict"] == "refuted"
    assert result["evidence_chain"][0]["id"] == "img1"
    statuses = {d["stage"]: d["status"] for n, d in events if n == "stage" and d["status"] != "running"}
    assert statuses["forensics"] == "done"


async def test_runner_calibration_off_keeps_raw(tmp_path, monkeypatch):
    calls = []
    settings, client = _mock_client(tmp_path, monkeypatch, calls)

    async def emit(event, data):
        pass

    result = await run_pipeline(
        "帖子", None, emit, settings=settings, client=client, retriever=_retriever(), trace_id="tcal0"
    )
    assert result["confidence"] == 0.9
    assert result["confidence_raw"] == 0.9


async def test_runner_calibration_posthoc_and_parity(tmp_path, monkeypatch):
    calls = []
    settings, client = _mock_client(tmp_path, monkeypatch, calls)
    cal_path = tmp_path / "calibrator.json"
    cal_path.write_text(json.dumps({"temperature": 3.0}), encoding="utf-8")
    settings.enable_calibration = True
    settings.calibrator_path = str(cal_path)

    async def emit(event, data):
        pass

    linear = await run_pipeline(
        "帖子", None, emit, settings=settings, client=client, retriever=_retriever(), trace_id="tcal1"
    )
    assert linear["confidence_raw"] == 0.9
    assert linear["confidence"] == pytest.approx(apply_temperature(0.9, 3.0))
    assert linear["confidence"] < 0.9

    # LangGraph runner 共用 _build_result，校准后置信度逐字节一致（parity 不被破坏）
    settings.enable_langgraph = True
    graph = await run_pipeline(
        "帖子", None, emit, settings=settings, client=client, retriever=_retriever(), trace_id="tcal2"
    )
    assert graph["confidence"] == linear["confidence"]
    assert graph["confidence_raw"] == linear["confidence_raw"]


async def test_runner_calibration_missing_artifact_is_noop(tmp_path, monkeypatch):
    calls = []
    settings, client = _mock_client(tmp_path, monkeypatch, calls)
    settings.enable_calibration = True
    settings.calibrator_path = str(tmp_path / "does_not_exist.json")

    async def emit(event, data):
        pass

    result = await run_pipeline(
        "帖子", None, emit, settings=settings, client=client, retriever=_retriever(), trace_id="tcal3"
    )
    assert result["confidence"] == 0.9  # 缺 artifact → T=1.0 不校准，流水线不中断