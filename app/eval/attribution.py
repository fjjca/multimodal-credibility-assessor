"""把错误样本归因到感知、检索、关系判定或裁决阶段。"""

import json
from collections import Counter
from pathlib import Path

CATEGORIES = ("perception", "retrieval", "relation", "adjudication")
LABELS = {
    "none": "正确",
    "perception": "感知/取证",
    "retrieval": "证据检索",
    "relation": "关系判定",
    "adjudication": "最终裁决",
    "inconclusive": "信息不足",
}


def _image_relations(summary: list[dict]) -> tuple[int, int]:
    support = contradict = 0
    for claim in summary:
        support += sum(item["evidence_id"] == "img1" for item in claim.get("support", []))
        contradict += sum(item["evidence_id"] == "img1" for item in claim.get("contradict", []))
    return support, contradict


def attribute_record(record: dict) -> dict:
    if record.get("correct") or record.get("verdict") == record.get("gold"):
        return {"category": "none", "signals": ["预测与标签一致"]}
    details = record.get("details") or {}
    if not details:
        return {"category": "inconclusive", "signals": ["记录无流水线详情（baseline 或旧格式）"]}

    forensics = details.get("forensics") or {}
    graph = details.get("graph") or {}
    summary = details.get("graph_summary") or []
    notes = str(forensics.get("notes") or "")
    crosscheck = forensics.get("ocr_crosscheck") or {}
    signals: list[str] = []

    if forensics:
        if any(word in notes for word in ("降级", "限流", "不可用", "拒答")):
            signals.append("VLM 取证降级")
        if crosscheck.get("status") == "conflict":
            signals.append(f"OCR 数字冲突 {crosscheck.get('conflicts')}")
        if record.get("gold") == "supported" and forensics.get("overall") == "inconsistent":
            signals.append("真实样本被判图文矛盾")
        if record.get("gold") == "refuted":
            _, image_contradict = _image_relations(summary)
            if forensics and not image_contradict:
                signals.append("虚假样本未形成图像矛盾边")
        if signals:
            return {"category": "perception", "signals": signals}

    external_ids = {
        item["evidence_id"]
        for claim in summary
        for key in ("support", "contradict")
        for item in claim.get(key, [])
        if item.get("evidence_id") != "img1"
    }
    n_external = int(graph.get("n_external_evidence", len(external_ids)))
    n_active = int(graph.get("n_support", 0)) + int(graph.get("n_contradict", 0))
    n_neutral = int(graph.get("n_neutral", 0))
    if details.get("retrieval_enabled") and n_external == 0 and n_active == 0:
        return {"category": "retrieval", "signals": ["检索开启但未取得任何外部证据"]}
    if n_external > 0 and n_active == 0 and n_neutral > 0:
        return {"category": "relation", "signals": [f"{n_neutral} 条候选证据全部落为 neutral"]}
    if n_active > 0:
        return {"category": "adjudication", "signals": [f"已有 {n_active} 条有效证据，最终裁决仍错误或弃权"]}
    return {"category": "inconclusive", "signals": ["无有效证据且无法定位到单一阶段"]}


def attribution_summary(records: list[dict]) -> dict:
    counts = Counter()
    examples: dict[str, list[str]] = {key: [] for key in (*CATEGORIES, "inconclusive")}
    for record in records:
        result = attribute_record(record)
        category = result["category"]
        counts[category] += 1
        if category in examples and len(examples[category]) < 10:
            examples[category].append(record.get("id", ""))
    failures = sum(counts[c] for c in (*CATEGORIES, "inconclusive"))
    return {
        "n_records": len(records),
        "n_failures": failures,
        "counts": dict(counts),
        "failure_rates": {
            c: round(counts[c] / failures, 4) if failures else 0.0
            for c in (*CATEGORIES, "inconclusive")
        },
        "examples": examples,
    }


def render_attribution_md(summary: dict) -> str:
    lines = [
        "# 失败归因",
        "",
        f"- 记录数：{summary['n_records']}",
        f"- 错误/弃权数：{summary['n_failures']}",
        "",
        "| 阶段 | 数量 | 占失败比例 | 示例 |",
        "|---|---:|---:|---|",
    ]
    for category in (*CATEGORIES, "inconclusive"):
        count = summary["counts"].get(category, 0)
        rate = summary["failure_rates"].get(category, 0.0)
        examples = "、".join(summary["examples"].get(category, [])) or "—"
        lines.append(f"| {LABELS[category]} | {count} | {rate:.1%} | {examples} |")
    lines += ["", "归因是规则化诊断，不是因果证明；用于定位下一轮实验方向。"]
    return "\n".join(lines) + "\n"


def write_attribution_report(records: list[dict], out_dir: str | Path, stem: str) -> dict:
    summary = attribution_summary(records)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"failure_attribution_{stem}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / f"failure_attribution_{stem}.md").write_text(
        render_attribution_md(summary), encoding="utf-8"
    )
    return summary
