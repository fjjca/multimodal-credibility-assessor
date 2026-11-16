"""从 Fakeddit 平衡测试集抽取二分类子集到统一 JSONL schema。

来源: hf-mirror.com/datasets/ams-99/fakeddit_9k (test_balanced.csv)
映射: Fakeddit 标准二分类 6_way_label==0 -> supported, 1-5 -> refuted。
保留 raw_label(6-way) 与 image(original_url) 供后续多模态阶段使用。
分层抽样保证 supported/refuted 各半。

    python -m scripts.prepare_fakeddit --n 100
"""

import argparse
import csv
import json
import random
from pathlib import Path

SOURCE = "fakeddit/ams-99/fakeddit_9k:test_balanced"


def map_label(six_way: str) -> str:
    return "supported" if six_way == "0" else "refuted"


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取 Fakeddit 二分类子集")
    parser.add_argument("--input", default="data/raw/fakeddit_test_balanced.csv")
    parser.add_argument("--out", default="data/fakeddit_subset.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [r for r in csv.DictReader(Path(args.input).open(encoding="utf-8")) if r["text"].strip()]
    by_class: dict[str, list] = {"supported": [], "refuted": []}
    for r in rows:
        by_class[map_label(r["6_way_label"])].append(r)

    rng = random.Random(args.seed)
    per = args.n // 2
    chosen = []
    for label, pool in by_class.items():
        chosen.extend(rng.sample(pool, min(per, len(pool))))
    rng.shuffle(chosen)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(chosen):
            f.write(json.dumps({
                "id": f"fk{i:04d}",
                "text": r["text"],
                "label": map_label(r["6_way_label"]),
                "image": r.get("original_url") or None,
                "source": SOURCE,
                "raw_label": r["6_way_label"],
            }, ensure_ascii=False) + "\n")

    counts = {c: sum(1 for r in chosen if map_label(r["6_way_label"]) == c) for c in by_class}
    print(f"wrote {len(chosen)} samples to {out}; class counts={counts}")


if __name__ == "__main__":
    main()