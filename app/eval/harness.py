import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.eval.schema import ABSTAIN, VERDICTS, Prediction, Sample

Predictor = Callable[[Sample], Awaitable[Prediction]]


def compute_metrics(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    abstained = [r for r in records if r["verdict"] == ABSTAIN]
    attempted = [r for r in records if r["verdict"] != ABSTAIN]
    correct_all = sum(r["verdict"] == r["gold"] for r in records)
    correct_attempted = sum(r["verdict"] == r["gold"] for r in attempted)

    per_class = {}
    for c in VERDICTS:
        tp = sum(r["verdict"] == c and r["gold"] == c for r in records)
        fp = sum(r["verdict"] == c and r["gold"] != c for r in records)
        fn = sum(r["verdict"] != c and r["gold"] == c for r in records)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1}

    present = [c for c in VERDICTS if any(r["gold"] == c for r in records)]
    macro_f1 = sum(per_class[c]["f1"] for c in present) / len(present) if present else 0.0
    total_tokens = sum(
        (r.get("usage") or {}).get("total_tokens", 0) for r in records
    )
    return {
        "n": n,
        "accuracy": correct_all / n,
        "abstention_rate": len(abstained) / n,
        "accuracy_on_attempted": correct_attempted / len(attempted) if attempted else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "total_tokens": total_tokens,
        "mean_latency_ms": sum(r.get("latency_ms", 0) for r in records) / n,
        "cached_ratio": sum(bool(r.get("cached")) for r in records) / n,
    }


async def run_eval(
    samples: list[Sample],
    predictor: Predictor,
    *,
    system_name: str,
    dataset_name: str,
    concurrency: int = 2,
    out_dir: str | Path = "out",
    resume: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(dataset_name).stem)
    ckpt_path = out / f"checkpoint_{system_name}_{stem}.jsonl"

    done: dict[str, dict] = {}
    if resume and ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["id"]] = rec

    pending = [s for s in samples if s.id not in done]
    if resume and done:
        print(f"[eval] resume: {len(done)} cached, {len(pending)} pending -> {ckpt_path}", flush=True)

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    failures: list[dict] = []

    async def one(sample: Sample) -> None:
        async with sem:
            try:
                pred = await predictor(sample)
            except Exception as exc:  # 单条失败不拖垮整轮，留待重跑补齐
                async with lock:
                    failures.append({"id": sample.id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[eval] FAIL {sample.id}: {type(exc).__name__}", flush=True)
                return
        rec = {
            "id": sample.id,
            "gold": sample.label,
            "verdict": pred.verdict,
            "correct": pred.verdict == sample.label,
            "abstained": pred.verdict == ABSTAIN,
            "confidence": pred.confidence,
            "reason": pred.reason,
            "usage": pred.usage,
            "latency_ms": pred.latency_ms,
            "cached": pred.cached,
        }
        async with lock:
            done[sample.id] = rec
            with ckpt_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(s) for s in pending))

    records = [done[s.id] for s in samples if s.id in done]
    metrics = compute_metrics(records)
    summary = {
        "system": system_name,
        "dataset": dataset_name,
        "timestamp": time.time(),
        "n_requested": len(samples),
        "n_failed": len(failures),
        "failed": failures,
        "metrics": metrics,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"eval_{system_name}_{stamp}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / f"eval_{system_name}_{stamp}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )
    (out / f"eval_{system_name}_{stamp}.md").write_text(
        _render_md(summary), encoding="utf-8"
    )
    if failures:
        print(f"[eval] {len(failures)} failed; re-run to resume from {ckpt_path}", flush=True)
    return summary


def _render_md(summary: dict) -> str:
    m = summary["metrics"]
    lines = [
        f"# 评测 {summary['system']} @ {summary['dataset']}",
        "",
        f"- 请求样本数: {summary.get('n_requested', m.get('n', 0))}",
        f"- 失败(限流/错误, 未计入指标): {summary.get('n_failed', 0)}",
        f"- 完成样本数: {m.get('n', 0)}",
    ]
    if m.get("n", 0) == 0:
        lines += ["", "无完成样本，重跑以从 checkpoint 续跑。"]
        return "\n".join(lines) + "\n"
    lines += [
        f"- 准确率(全量): {m['accuracy']:.3f}",
        f"- 准确率(去除弃权): {m['accuracy_on_attempted']:.3f}",
        f"- 弃权率: {m['abstention_rate']:.3f}",
        f"- macro-F1: {m['macro_f1']:.3f}",
        f"- 总 tokens: {m['total_tokens']}",
        f"- 平均延迟(ms): {m['mean_latency_ms']:.0f}",
        f"- 缓存命中比例: {m['cached_ratio']:.3f}",
        "",
        "| 类别 | precision | recall | f1 |",
        "|---|---|---|---|",
    ]
    for c, v in m["per_class"].items():
        lines.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    return "\n".join(lines) + "\n"