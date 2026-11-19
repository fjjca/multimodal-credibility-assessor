"""事后置信度校准拟合入口（不重跑推理，直接读已采集的 eval 记录，省额度）。

    python -m scripts.calibrate --records out/abl_text_linear/eval_pipeline_*.jsonl

读取一条或多条 eval .jsonl（按样本 id 去重），在 attempted（非弃权）样本上网格搜索温度 T，
报告校准前/后及 LOO-CV 的 ECE/MCE/Brier，落盘:
  - .data/calibrator.json   供流水线 ENABLE_CALIBRATION=true 时加载（只含 temperature 等元数据）
  - .data/reliability.json  完整分箱与逐样本概率，供 B2 可靠性曲线 / B1 dashboard 渲染
"""

import argparse
import json
import time
from pathlib import Path

from app.eval.calibration import fit_calibrator


def _load_records(paths: list[str]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for pat in paths:
        for path in sorted(Path().glob(pat)) if any(c in pat for c in "*?[") else [Path(pat)]:
            if not path.exists():
                raise SystemExit(f"记录文件不存在: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_id.setdefault(rec["id"], rec)
    return list(by_id.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="置信度校准拟合（事后，不重跑推理）")
    parser.add_argument("--records", nargs="+", required=True, help="eval .jsonl 路径（支持 glob）")
    parser.add_argument("--out", default=".data", help="artifact 输出目录")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    records = _load_records(args.records)
    cal = fit_calibrator(records, n_bins=args.bins)
    if cal["n_attempted"] == 0:
        raise SystemExit("没有可校准的 attempted 样本（全部弃权或缺 confidence）")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "temperature": cal["temperature"],
        "n_attempted": cal["n_attempted"],
        "n_bins": cal["n_bins"],
        "n_records": len(records),
        "source_records": args.records,
        "fit_timestamp": time.time(),
    }
    (out / "calibrator.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "reliability.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"样本: {len(records)} 条记录, attempted {cal['n_attempted']} 条")
    print(f"拟合温度 T = {cal['temperature']:.3f}  (>1 软化过度自信)")
    print(f"{'指标':<8}{'raw':>10}{'calibrated':>12}{'LOO-CV':>10}")
    for k in ("ece", "mce", "brier"):
        print(f"{k:<8}{cal['raw'][k]:>10.4f}{cal['calibrated'][k]:>12.4f}{cal['loo'][k]:>10.4f}")
    print(f"\n已写入 {out / 'calibrator.json'} 与 {out / 'reliability.json'}")
    print("启用流水线校准: ENABLE_CALIBRATION=true（路由/升级仍用原始 conf，仅变换上报置信度）")
    if cal["n_attempted"] < 20:
        print(f"注意: attempted 仅 {cal['n_attempted']} 条，LOO 指标可能不稳；建议采集更多标注样本后重拟合。")


if __name__ == "__main__":
    main()