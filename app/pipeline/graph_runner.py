"""LangGraph 编排：flag 门控，与手写直线 runner 等价（max_rounds=1），并支持循环多轮辩论。

节点复用现有阶段函数与 runner._build_result / adjudicate 的共享助手，业务逻辑不重写、
只重排编排，从而保证返回 dict / SSE 事件 / trace 落盘契约与 linear 一致。
多轮辩论用条件边成环（prosecutor→defender→judge→next_round→prosecutor），
受 debate_max_rounds 硬上限收敛；整个辩论对外仍是单个 adjudicate 阶段（running→done）。
emit 与依赖经闭包注入，数据流经 State reducer 累积；`ENABLE_CHECKPOINT` 门控 AsyncSqliteSaver
按 thread_id 持久化每个超步，中断/崩溃后同 thread_id 续跑（已完成节点不重算）；默认关闭零开销。
"""

import operator
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app import obs
from app.eval.schema import ABSTAIN, parse_verdict
from app.pipeline.adjudicate import (
    DEFENDER_SYSTEM,
    PROSECUTOR_SYSTEM,
    _argue,
    debate_base,
    judge_once,
    needs_escalation,
)
from app.pipeline.decompose import decompose
from app.pipeline.evidence_graph import build_evidence_graph
from app.pipeline.forensics import forensics, image_evidence, image_relations, load_image_url
from app.pipeline.graph import graph_stats
from app.pipeline.retrieve import retrieve_evidence
from app.pipeline.runner import _build_result, _usage


class GraphState(TypedDict, total=False):
    text: str
    image_path: str | None
    claims: list
    image_url: str | None
    forensics_detail: dict | None
    img_ev: object | None
    img_rels: list | None
    evidences: list
    claim_map: dict
    g: object | None
    stats: dict
    verdict: dict | None
    debate_base: str
    pros_arg: str
    def_arg: str
    debate_round: int
    judge_feedback: str
    results: Annotated[list, operator.add]
    adj_results: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]
    result: dict | None


def build_pipeline_graph(settings, client, retriever, emit, trace_id, checkpointer=None):
    stage_started: dict[str, float] = {}

    async def begin(sid: str, label: str) -> None:
        stage_started[sid] = time.perf_counter()
        await emit("stage", {"stage": sid, "label": label, "status": "running"})

    async def done(sid: str, label: str, detail: dict, results: list) -> dict:
        usage = _usage(results)
        start = stage_started.pop(sid, None)
        elapsed = time.perf_counter() - start if start is not None else 0.0
        obs.record_stage(sid, elapsed)
        obs.log.info(
            "stage done",
            extra={"trace_id": trace_id, "stage": sid, "latency_s": round(elapsed, 3), **usage},
        )
        await emit("stage", {"stage": sid, "label": label, "status": "done", "detail": detail})
        return {"stage": sid, "detail": detail, "usage": usage}

    async def decompose_node(state: GraphState) -> dict:
        await begin("decompose", "断言拆解")
        claims, res = await decompose(client, state["text"], settings.retrieve_max_claims)
        detail = {"n_claims": len(claims), "claims": [c.text for c in claims]}
        entry = await done("decompose", "断言拆解", detail, [res])
        return {"claims": claims, "results": [res], "trace": [entry]}

    async def forensics_node(state: GraphState) -> dict:
        await begin("forensics", "多模态取证")
        image_path = state.get("image_path")
        image_url = load_image_url(image_path)
        fr, fr_res = await forensics(client, image_url, state["text"], state["claims"])
        if fr is None:
            await emit("stage", {
                "stage": "forensics", "label": "多模态取证", "status": "skipped",
                "detail": {"note": "无图像输入，跳过图文取证", "image": "无"},
            })
            return {
                "image_url": image_url, "forensics_detail": None,
                "img_ev": None, "img_rels": None,
                "trace": [{"stage": "forensics", "skipped": True}],
            }
        img_ev = image_evidence(fr)
        img_rels = image_relations(fr, state["claims"])
        detail = {
            "ocr_text": fr.ocr_text,
            "description": fr.description,
            "overall": fr.overall,
            "consistency": [
                {"claim_id": c.claim_id, "consistency": c.consistency, "note": c.note}
                for c in fr.claims
            ],
            "notes": fr.notes,
            "image": Path(image_path).name if image_path else "无",
        }
        results = [fr_res] if fr_res is not None else []
        entry = await done("forensics", "多模态取证", detail, results)
        return {
            "image_url": image_url, "forensics_detail": detail,
            "img_ev": img_ev, "img_rels": img_rels,
            "results": results, "trace": [entry],
        }

    async def retrieve_node(state: GraphState) -> dict:
        await begin("retrieve", "证据检索")
        if settings.enable_retrieval:
            evidences, claim_map = await retrieve_evidence(
                retriever, state["claims"], settings.retrieve_top_k
            )
        else:
            evidences, claim_map = [], {c.id: [] for c in state["claims"]}
        detail = {"n_evidence": len(evidences), "enabled": settings.enable_retrieval}
        entry = await done("retrieve", "证据检索", detail, [])
        return {"evidences": evidences, "claim_map": claim_map, "trace": [entry]}

    async def graph_node(state: GraphState) -> dict:
        await begin("graph", "证据图构建")
        g, _relations, graph_results = await build_evidence_graph(
            client, state["claims"], state["evidences"], state["claim_map"],
            image_evidence=state.get("img_ev"), image_relations=state.get("img_rels"),
        )
        stats = graph_stats(g)
        entry = await done("graph", "证据图构建", stats, graph_results)
        return {"g": g, "stats": stats, "results": graph_results, "trace": [entry]}

    # --- 裁决子图：控辩裁循环 + 置信度升级，对外仍是单个 adjudicate 阶段 ---

    def debate_input(state: GraphState) -> str:
        base = state["debate_base"]
        if state["debate_round"] > 1 and state.get("judge_feedback"):
            return (
                f"{base}\n\n上一轮裁判意见：{state['judge_feedback']}\n"
                "请据此修正并强化你的论证，不得编造图中不存在的证据。"
            )
        return base

    async def debate_start_node(state: GraphState) -> dict:
        await begin("adjudicate", "多模型裁决")
        return {
            "debate_base": debate_base(state["claims"], state["g"]),
            "pros_arg": "（未进行辩论）",
            "def_arg": "（未进行辩论）",
            "debate_round": 1,
            "judge_feedback": "",
        }

    async def prosecutor_node(state: GraphState) -> dict:
        arg, r = await _argue(client, "prosecutor", PROSECUTOR_SYSTEM, debate_input(state))
        return {"pros_arg": arg, "results": [r], "adj_results": [r]}

    async def defender_node(state: GraphState) -> dict:
        arg, r = await _argue(client, "defender", DEFENDER_SYSTEM, debate_input(state))
        return {"def_arg": arg, "results": [r], "adj_results": [r]}

    async def judge_node(state: GraphState) -> dict:
        verdict, r = await judge_once(
            client, state["debate_base"], state["pros_arg"], state["def_arg"]
        )
        feedback = f"{verdict['verdict']}（置信度 {verdict['confidence']}）：{verdict['reason']}"
        return {"verdict": verdict, "judge_feedback": feedback, "results": [r], "adj_results": [r]}

    async def next_round_node(state: GraphState) -> dict:
        return {"debate_round": state["debate_round"] + 1}

    async def escalate_node(state: GraphState) -> dict:
        strong = settings.model_judge_strong
        verdict, r = await judge_once(
            client, state["debate_base"], state["pros_arg"], state["def_arg"], model_override=strong
        )
        verdict["escalated"] = True
        verdict["judge_model"] = r.model or strong.partition(":")[2]
        return {"verdict": verdict, "results": [r], "adj_results": [r]}

    async def debate_end_node(state: GraphState) -> dict:
        verdict = dict(state["verdict"])
        verdict["prosecutor"] = state["pros_arg"]
        verdict["defender"] = state["def_arg"]
        detail = {
            "verdict": verdict["verdict"],
            "confidence": verdict["confidence"],
            "debate": settings.enable_debate,
            "escalated": verdict.get("escalated", False),
            "judge_model": verdict.get("judge_model", ""),
        }
        if settings.debate_max_rounds > 1:
            detail["rounds"] = state["debate_round"]
        entry = await done("adjudicate", "多模型裁决", detail, state["adj_results"])
        return {"verdict": verdict, "trace": [entry]}

    async def review_node(state: GraphState) -> dict:
        # HITL：弃权或低置信时 interrupt 暂停等人工复核；恢复值=人工裁决（空 dict=维持原判）
        verdict = state["verdict"]
        conf = verdict["confidence"]
        low = conf is None or conf < settings.judge_confidence_threshold
        if not (settings.enable_hitl and (verdict["verdict"] == ABSTAIN or low)):
            return {}
        payload = {
            "verdict": verdict["verdict"], "confidence": conf, "reason": verdict["reason"],
            "claims": [c.text for c in state["claims"]],
        }
        decision = interrupt(payload)
        verdict = dict(verdict)
        verdict["human_review"] = True
        if isinstance(decision, dict):
            if decision.get("verdict"):
                verdict["verdict"] = parse_verdict(str(decision["verdict"]))
            if decision.get("confidence") is not None:
                try:
                    verdict["confidence"] = float(decision["confidence"])
                except (TypeError, ValueError):
                    pass
            if decision.get("reason"):
                verdict["reason"] = str(decision["reason"])
        return {"verdict": verdict}

    def route_debate_start(state: GraphState) -> str:
        return "prosecutor" if settings.enable_debate else "judge"

    def route_after_judge(state: GraphState) -> str:
        verdict = state["verdict"]
        conf = verdict["confidence"]
        low = conf is None or conf < settings.judge_confidence_threshold
        if (
            settings.enable_debate
            and state["debate_round"] < settings.debate_max_rounds
            and low
            and verdict["judge_model"]  # 内容审核降级不循环
        ):
            return "next_round"
        if needs_escalation(settings, verdict, verdict["judge_model"]):
            return "escalate"
        return "debate_end"

    async def finalize_node(state: GraphState) -> dict:
        result = _build_result(
            claims=state["claims"], evidences=state["evidences"], img_ev=state.get("img_ev"),
            verdict=state["verdict"], stats=state["stats"], g=state["g"],
            forensics_detail=state.get("forensics_detail"), all_results=state["results"],
            trace=state["trace"], trace_id=trace_id, settings=settings,
        )
        await emit("result", result)
        return {"result": result}

    builder = StateGraph(GraphState)
    for name, fn in [
        ("decompose", decompose_node), ("forensics", forensics_node),
        ("retrieve", retrieve_node), ("graph", graph_node),
        ("debate_start", debate_start_node), ("prosecutor", prosecutor_node),
        ("defender", defender_node), ("judge", judge_node),
        ("next_round", next_round_node), ("escalate", escalate_node),
        ("debate_end", debate_end_node), ("review", review_node), ("finalize", finalize_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "forensics")
    builder.add_edge("forensics", "retrieve")
    builder.add_edge("retrieve", "graph")
    builder.add_edge("graph", "debate_start")
    builder.add_conditional_edges(
        "debate_start", route_debate_start, {"prosecutor": "prosecutor", "judge": "judge"}
    )
    builder.add_edge("prosecutor", "defender")
    builder.add_edge("defender", "judge")
    builder.add_conditional_edges(
        "judge", route_after_judge,
        {"next_round": "next_round", "escalate": "escalate", "debate_end": "debate_end"},
    )
    builder.add_edge("next_round", "prosecutor")  # 成环：多轮辩论
    builder.add_edge("escalate", "debate_end")
    builder.add_edge("debate_end", "review")
    builder.add_edge("review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def _saver(settings):
    # 懒加载：默认档不依赖 sqlite saver。state 携带 nx.Graph/dataclass/pydantic，msgpack 无法
    # 直接编码，故 pickle_fallback；仅作用于本地自有检查点文件（可信 artifact），不接外部输入。
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_path) as saver:
        saver.serde = JsonPlusSerializer(pickle_fallback=True)
        await saver.setup()
        yield saver


def _pending_review(final: dict, trace_id: str) -> dict | None:
    if "__interrupt__" not in final:
        return None
    interrupts = final["__interrupt__"]
    payload = interrupts[0].value if interrupts else {}
    v = final.get("verdict") or {}
    return {
        "pending_review": True, "review": payload, "trace_id": trace_id,
        "verdict": v.get("verdict"), "confidence": v.get("confidence"), "reason": v.get("reason"),
    }


async def run_graph_pipeline(
    text: str, image_path: str | None, emit, *, settings, client, retriever, trace_id: str
) -> dict:
    initial = {
        "text": text, "image_path": image_path,
        "results": [], "adj_results": [], "trace": [],
    }
    config = {"configurable": {"thread_id": trace_id}}
    # HITL 的 interrupt 依赖检查点持久化，故 enable_hitl 隐含启用 saver
    if settings.enable_checkpoint or settings.enable_hitl:
        async with _saver(settings) as saver:
            app = build_pipeline_graph(
                settings, client, retriever, emit, trace_id, checkpointer=saver
            )
            # 已有检查点=中断/崩溃后的续跑：传 None 恢复待执行节点，不重算已完成超步
            existing = await saver.aget(config)
            final = await app.ainvoke(None if existing else initial, config)
    else:
        app = build_pipeline_graph(settings, client, retriever, emit, trace_id)
        final = await app.ainvoke(initial, config)
    pending = _pending_review(final, trace_id)
    if pending is not None:
        return pending
    return final["result"]


async def run_graph_resume(
    text: str, image_path: str | None, emit, *, settings, client, retriever, trace_id: str,
    decision: dict,
) -> dict:
    """HITL 恢复：把人工裁决作为 interrupt 的返回注入 review 节点，继续到 finalize。"""
    config = {"configurable": {"thread_id": trace_id}}
    # 空 dict 会被 LangGraph 判为「中断ID→值」空映射（空键 all() 为真），不注入恢复值而再次
    # interrupt；维持原判改用显式标记，review 节点对无 override 键的 dict 一律保持原裁决
    resume_val = decision if decision else {"approve": True}
    async with _saver(settings) as saver:
        app = build_pipeline_graph(settings, client, retriever, emit, trace_id, checkpointer=saver)
        final = await app.ainvoke(Command(resume=resume_val), config)
    return final["result"]