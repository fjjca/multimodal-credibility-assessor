import json
from pathlib import Path

from app.eval.schema import Sample


def load_jsonl(path: str | Path) -> list[Sample]:
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        samples.append(Sample.model_validate(json.loads(line)))
    return samples