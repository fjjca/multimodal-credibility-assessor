import base64

import pytest

from app.config import Settings
from app.llm.client import LLMClient, LLMResult
from app.pipeline.models import Evidence
from app.pipeline.retrieve import FakeRetriever
from app.pipeline.runner import run_pipeline

_PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _settings(tmp_path, enable_langgraph, **overrides):
    return Settings(
        _env_file=None, llm_cache_mode="off", llm_cache_dir=str(tmp_path),
        trace_dir=str(tmp_path), enable_langgraph=enable_langgraph, **overrides,
    )


def _fake_chat_factory(calls):
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
            return LLMResult(content='[{"evidence_id":"e1","relation":"support"}]', model="m", usage=usage)
        if role == "prosecutor":
            return LLMResult(content='{"argument":"虚假"}', model="m", usage=usage)
        if role == "defender":
            return LLMResult(content='{"argument":"真实"}', model="m", usage=usage)
        if role == "judge":
            return LLMResult(
                content='{"verdict":"supported","confidence":0.9,"reason":"证据支持","evidence_chain":["e1"]}',
                model="m", usage=usage,
            )
        return LLMResult(content="[]", model="m", usage=usage)

    return fake_chat


def _retriever():
    return FakeRetriever(
        [Evidence(id="", title="官方佐证", url="https://gov.cn/a", content="支持内容", source_tier="official")]
    )


async def _run(tmp_path, monkeypatch, enable_langgraph, image_path=None, trace_id="parity", **overrides):
    settings = _settings(tmp_path, enable_langgraph, **overrides)
    client = LLMClient(settings)
    calls = []
    monkeypatch.setattr(client, "chat", _fake_chat_factory(calls))
    events = []

    async def emit(event, data):
        events.append((event, data))

    result = await run_pipeline(
        "某帖子", image_path, emit, settings=settings, client=client,
        retriever=_retriever(), trace_id=trace_id,
    )
    return result, events, calls


async def test_graph_parity_no_image(tmp_path, monkeypatch):
    lin_res, lin_ev, lin_calls = await _run(tmp_path, monkeypatch, False)
    gra_res, gra_ev, gra_calls = await _run(tmp_path, monkeypatch, True)
    assert lin_res == gra_res          # 返回 dict 逐项相等
    assert lin_ev == gra_ev            # SSE 事件序列逐项相等
    assert lin_calls == gra_calls      # LLM 角色调用顺序一致
    assert gra_res["verdict"] == "supported"


async def test_graph_parity_with_image(tmp_path, monkeypatch):
    png = tmp_path / "img.png"
    png.write_bytes(base64.b64decode(_PNG_1PX))
    lin_res, lin_ev, _ = await _run(
        tmp_path, monkeypatch, False, image_path=str(png), enable_escalation=False
    )
    gra_res, gra_ev, _ = await _run(
        tmp_path, monkeypatch, True, image_path=str(png), enable_escalation=False
    )
    assert lin_res == gra_res
    assert lin_ev == gra_ev
    assert gra_res["forensics"]["overall"] == "inconsistent"
    assert gra_res["graph"]["n_contradict"] == 1


@pytest.mark.parametrize(
    "overrides",
    [{"enable_debate": False}, {"enable_retrieval": False},
     {"enable_debate": False, "enable_retrieval": False}],
)
async def test_graph_parity_ablations(tmp_path, monkeypatch, overrides):
    lin_res, lin_ev, _ = await _run(tmp_path, monkeypatch, False, **overrides)
    gra_res, gra_ev, _ = await _run(tmp_path, monkeypatch, True, **overrides)
    assert lin_res == gra_res
    assert lin_ev == gra_ev


async def test_flag_dispatches_to_graph(tmp_path, monkeypatch):
    import app.pipeline.graph_runner as gr

    called = {}

    async def sentinel(text, image_path, emit, **kw):
        called["yes"] = True
        return {"verdict": "sentinel"}

    monkeypatch.setattr(gr, "run_graph_pipeline", sentinel)
    settings = _settings(tmp_path, True)
    client = LLMClient(settings)

    async def emit(event, data):
        pass

    res = await run_pipeline(
        "x", None, emit, settings=settings, client=client, retriever=FakeRetriever([]), trace_id="d1"
    )
    assert called.get("yes") is True
    assert res["verdict"] == "sentinel"


def test_graph_has_expected_nodes(tmp_path):
    from app.pipeline.graph_runner import build_pipeline_graph

    settings = _settings(tmp_path, True)
    client = LLMClient(settings)

    async def emit(event, data):
        pass

    app = build_pipeline_graph(settings, client, FakeRetriever([]), emit, "s1")
    nodes = set(app.get_graph().nodes)
    assert {
        "decompose", "forensics", "retrieve", "graph",
        "debate_start", "prosecutor", "defender", "judge", "debate_end", "finalize",
    } <= nodes


_LOW = '{"verdict":"unverifiable","confidence":0.4,"reason":"不确定"}'
_HIGH = '{"verdict":"refuted","confidence":0.9,"reason":"反证","evidence_chain":[]}'


def _debate_setup(tmp_path, monkeypatch, judge_fn, **overrides):
    settings = _settings(tmp_path, True, enable_retrieval=False, **overrides)
    client = LLMClient(settings)
    calls = []

    async def fake_chat(role, messages, **kw):
        calls.append((role, kw.get("model_override")))
        usage = {"total_tokens": 10}
        if role == "prosecutor":
            return LLMResult(content='{"argument":"虚假"}', model="m", usage=usage)
        if role == "defender":
            return LLMResult(content='{"argument":"真实"}', model="m", usage=usage)
        if role == "judge":
            return judge_fn(kw.get("model_override"), usage)
        if "断言拆解器" in messages[0]["content"]:
            return LLMResult(content='[{"text":"断言A"}]', model="m", usage=usage)
        return LLMResult(content="[]", model="m", usage=usage)

    monkeypatch.setattr(client, "chat", fake_chat)
    return settings, client, calls


async def _invoke(settings, client, trace_id):
    events = []

    async def emit(event, data):
        events.append((event, data))

    res = await run_pipeline(
        "某帖子", None, emit, settings=settings, client=client,
        retriever=FakeRetriever([]), trace_id=trace_id,
    )
    return res, events


def _adj_done(events):
    return [d for e, d in events
            if e == "stage" and d.get("stage") == "adjudicate" and d.get("status") == "done"][0]


async def test_multi_round_debate_converges(tmp_path, monkeypatch):
    n = {"judge": 0}

    def judge_fn(override, usage):
        n["judge"] += 1
        return LLMResult(content=_LOW if n["judge"] < 2 else _HIGH, model="m", usage=usage)

    settings, client, calls = _debate_setup(
        tmp_path, monkeypatch, judge_fn, debate_max_rounds=3, enable_escalation=False
    )
    res, events = await _invoke(settings, client, "mr1")
    roles = [r for r, _ in calls]
    assert roles.count("judge") == 2          # 第2轮高置信即收敛，不跑满3轮
    assert roles.count("prosecutor") == 2 and roles.count("defender") == 2
    assert res["verdict"] == "refuted" and res["confidence"] == 0.9
    assert _adj_done(events)["detail"]["rounds"] == 2


async def test_multi_round_debate_bounded_by_max(tmp_path, monkeypatch):
    def judge_fn(override, usage):
        return LLMResult(content=_LOW, model="m", usage=usage)

    settings, client, calls = _debate_setup(
        tmp_path, monkeypatch, judge_fn, debate_max_rounds=3, enable_escalation=False
    )
    res, events = await _invoke(settings, client, "mr2")
    roles = [r for r, _ in calls]
    assert roles.count("judge") == 3          # 始终低置信也受硬上限收敛，不死循环
    assert res["verdict"] == "unverifiable"
    assert _adj_done(events)["detail"]["rounds"] == 3


async def test_debate_off_no_cycle(tmp_path, monkeypatch):
    def judge_fn(override, usage):
        return LLMResult(content=_LOW, model="m", usage=usage)

    settings, client, calls = _debate_setup(
        tmp_path, monkeypatch, judge_fn,
        debate_max_rounds=3, enable_debate=False, enable_escalation=False,
    )
    res, events = await _invoke(settings, client, "mr3")
    roles = [r for r, _ in calls]
    assert "prosecutor" not in roles and "defender" not in roles
    assert roles.count("judge") == 1          # 关辩论则无循环


async def test_multi_round_then_escalate(tmp_path, monkeypatch):
    def judge_fn(override, usage):
        if override:
            return LLMResult(content=_HIGH, model="glm-4.5-flash", usage=usage)
        return LLMResult(content=_LOW, model="glm-4-flash", usage=usage)

    settings, client, calls = _debate_setup(
        tmp_path, monkeypatch, judge_fn, debate_max_rounds=2,
        model_judge="zhipu:glm-4-flash", model_judge_strong="zhipu:glm-4.5-flash",
        judge_confidence_threshold=0.6,
    )
    res, events = await _invoke(settings, client, "mr4")
    judge_overrides = [o for r, o in calls if r == "judge"]
    assert judge_overrides == [None, None, "zhipu:glm-4.5-flash"]  # 2轮循环 + 升级
    assert res["escalated"] is True and res["verdict"] == "refuted"