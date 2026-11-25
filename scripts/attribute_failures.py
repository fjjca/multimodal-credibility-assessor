"""对已有评测 JSONL 做阶段失败归因（不重跑推理）。

    python -m scripts.attribute_failures --records out/photo_v2/eval_pipeline_*.jsonl --out out/photo_v2
"""

import argparse
import json
from pathlib import Path

from app.eval.attribution import write_attribution_report


def _load_records(patterns: list[str]) -> list[dict]:
    records = []
    for pattern in patterns:
        paths = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        for path in paths:
            if not path.exists():
                raise SystemExit(f"记录文件不存在: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="评测失败阶段归因")
    parser.add_argument("--records", nargs="+", required=True, help="eval .jsonl 路径（支持 glob）")
    parser.add_argument("--out", default=".data/attribution")
    parser.add_argument("--stem", default="latest")
    args = parser.parse_args()
    summary = write_attribution_report(_load_records(args.records), args.out, args.stem)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已写入 {Path(args.out) / f'failure_attribution_{args.stem}.json'} 与 .md")


if __name__ == "__main__":
    main()
