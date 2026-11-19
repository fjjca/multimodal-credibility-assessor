import json
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from app import obs
from app.config import Settings, get_settings
from app.eval.calibration import apply_temperature, load_temperature
from app.llm.cache import ResponseCache
from app.llm.client import LLMClient, LLMResult
from app.pipeline.adjudicate import adjudicate
from app.pipeline.decompose import decompose
from app.pipeline.evidence_graph import build_evidence_graph
from app.pipeline.forensics import forensics, image_evidence, image_relations, load_image_url
from app.pipeline.graph import claim_evidence_summary, graph_elements, graph_stats, save_graph
from app.pipeline.retrieve import FakeRetriever, TavilyRetriever, retrieve_evidence

Emit = Callable[[str, dict], Awaitable[None]]

STAGES = [
    ("decompose", "断言拆解"),
    ("forensics", "多模态取证"),
    ("retrieve", "证据检索"),
    ("graph", "证据图构建"),
    ("adjudicate", "多模型裁决"),
]

VERDICT_LABELS = {
    "supported": "真实可信",
    "refuted": "虚假不实",
    "misleading": "部分误导",
    "unverifiable": "无法验证",
}


def _default_retriever(settings: Settings):
    if settings.enable_retrieval and settings.tavily_api_key:
        cache = ResponseCache(settings.llm_cache_dir, settings.llm_cache_mode)
        return TavilyRetriever(settings.tavily_api_key, settings.tavily_base_url, cache=cache)
    return FakeRetriever([])


def _usage(results: list[LLMResult]) -> dict:
    return {
        "total_tokens": sum((r.usage or {}).get("total_tokens", 0) for r in results),
        "n_calls": len(results),
        "cached": sum(bool(r.cached) for r in results),
    }


def _build_result(
    *, claims, evidences, img_ev, verdict, stats, g, forensics_detail,
    all_results, trace, trace_id, settings,
) -> dict:
    """组装结果 dict 并落盘 trace/证据图。linear 与 LangGraph 两条编排共用，保证输出契约一致。"""
    by_id = {e.id: e for e in evidences}
    if img_ev is not None:
        by_id[img_ev.id] = img_ev
    chain = [
        {"id": i, "title": by_id[i].title, "url": by_id[i].url, "source_tier": by_id[i].source_tier}
        for i in verdict["evidence_chain"] if i in by_id
    ]
    # 事后校准：仅变换上报置信度，编排路由/升级仍用原始 conf（两条 runner 共用此函数→parity 不变）
    conf_raw = verdict["confidence"]
    conf = conf_raw
    if settings.enable_calibration and conf_raw is not None:
        conf = apply_temperature(conf_raw, load_temperature(settings.calibrator_path))
    result = {
        "verdict": verdict["verdict"],
        "verdict_label": VERDICT_LABELS.get(verdict["verdict"], verdict["verdict"]),
        "confidence": conf,
        "confidence_raw": conf_raw,
        "reason": verdict["reason"],
        "claims": [{"id": c.id, "text": c.text} for c in claims],
        "evidence_chain": chain,
        "graph": stats,
        "graph_elements": graph_elements(g),
        "graph_summary": claim_evidence_summary(g),
        "forensics": forensics_detail,
        "prosecutor": verdict["prosecutor"],
        "defender": verdict["defender"],
        "escalated": verdict.get("escalated", False),
        "judge_model": verdict.get("judge_model", ""),
        "usage": _usage(all_results),
        "trace_id": trace_id,
    }
    trace_dir = Path(settings.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    save_graph(g, trace_dir / f"{trace_id}_graph.json")
    (trace_dir / f"{trace_id}.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in trace),
        encoding="utf-8",
    )
    return result


async def run_pipeline(
    text: str,
    image_path: str | None,
    emit: Emit,
    *,
    settings: Settings | None = None,
    client: LLMClient | None = None,
    retriever=None,
    trace_id: str | None = None,
) -> dict:
    settings = settings or get_settings()
    client = client or LLMClient(settings)
    if retriever is None:
        retriever = _default_retriever(settings)
    trace_id = trace_id or uuid.uuid4().hex[:12]
    if settings.enable_langgraph:
        from app.pipeline.graph_runner import run_graph_pipeline

        return await run_graph_pipeline(
            text, image_path, emit, settings=settings, client=client,
            retriever=retriever, trace_id=trace_id,
        )
    trace: list[dict] = []
    all_results: list[LLMResult] = []
    stage_started: dict[str, float] = {}

    async def begin(stage_id: str, label: str) -> None:
        stage_started[stage_id] = time.perf_counter()
        await emit("stage", {"stage": stage_id, "label": label, "status": "running"})

    async def done(stage_id: str, label: str, detail: dict, results: list[LLMResult]) -> None:
        all_results.extend(results)
        usage = _usage(results)
        start = stage_started.pop(stage_id, None)
        elapsed = time.perf_counter() - start if start is not None else 0.0
        obs.record_stage(stage_id, elapsed)
        obs.log.info(
            "stage done",
            extra={"trace_id": trace_id, "stage": stage_id, "latency_s": round(elapsed, 3), **usage},
        )
        trace.append({"stage": stage_id, "detail": detail, "usage": usage})
        await emit("stage", {"stage": stage_id, "label": label, "status": "done", "detail": detail})

    # 1. 断言拆解
    await begin("decompose", "断言拆解")
    claims, dec_res = await decompose(client, text, settings.retrieve_max_claims)
    await done(
        "decompose", "断言拆解",
        {"n_claims": len(claims), "claims": [c.text for c in claims]}, [dec_res],
    )

    # 2. 多模态取证（VLM 图文一致性 + OCR；无图则跳过）
    await begin("forensics", "多模态取证")
    image_url = load_image_url(image_path)
    fr, fr_res = await forensics(client, image_url, text, claims)
    img_ev = img_rels = None
    forensics_detail = None
    if fr is not None:
        img_ev = image_evidence(fr)
        img_rels = image_relations(fr, claims)
        forensics_detail = {
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
        await done(
            "forensics", "多模态取证", forensics_detail, [fr_res] if fr_res is not None else []
        )
    else:
        await emit("stage", {
            "stage": "forensics", "label": "多模态取证", "status": "skipped",
            "detail": {"note": "无图像输入，跳过图文取证", "image": "无"},
        })
        trace.append({"stage": "forensics", "skipped": True})

    # 3. 证据检索
    await begin("retrieve", "证据检索")
    if settings.enable_retrieval:
        evidences, claim_map = await retrieve_evidence(retriever, claims, settings.retrieve_top_k)
    else:
        evidences, claim_map = [], {c.id: [] for c in claims}
    await done(
        "retrieve", "证据检索",
        {"n_evidence": len(evidences), "enabled": settings.enable_retrieval}, [],
    )

    # 4. 证据图构建
    await begin("graph", "证据图构建")
    g, relations, graph_results = await build_evidence_graph(
        client, claims, evidences, claim_map, image_evidence=img_ev, image_relations=img_rels
    )
    stats = graph_stats(g)
    await done("graph", "证据图构建", stats, graph_results)

    # 5. 多模型裁决
    await begin("adjudicate", "多模型裁决")
    verdict, adj_results = await adjudicate(client, claims, g, settings.enable_debate)
    await done(
        "adjudicate", "多模型裁决",
        {
            "verdict": verdict["verdict"],
            "confidence": verdict["confidence"],
            "debate": settings.enable_debate,
            "escalated": verdict.get("escalated", False),
            "judge_model": verdict.get("judge_model", ""),
        },
        adj_results,
    )

    result = _build_result(
        claims=claims, evidences=evidences, img_ev=img_ev, verdict=verdict, stats=stats,
        g=g, forensics_detail=forensics_detail, all_results=all_results, trace=trace,
        trace_id=trace_id, settings=settings,
    )
    await emit("result", result)
    return result