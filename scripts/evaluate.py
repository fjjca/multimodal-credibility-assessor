"""评测入口。填好 .env 的 key 后运行:

    python -m scripts.evaluate --data data/eval_seed.jsonl --system baseline

无 key 时可用缓存 replay 模式回放已录制响应（LLM_CACHE_MODE=replay）。
"""

import argparse
import asyncio
import json

from app import obs
from app.config import get_settings
from app.eval.baseline import BaselinePredictor
from app.eval.dataset import load_jsonl
from app.eval.harness import run_eval
from app.llm.client import LLMClient


def build_predictor(system: str, client: LLMClient):
    if system == "baseline":
        return BaselinePredictor(client)
    if system == "baseline_fc":
        return BaselinePredictor(client, force_choice=True)
    if system == "pipeline":
        from app.eval.pipeline import PipelinePredictor
        from app.pipeline.runner import _default_retriever

        settings = client.settings
        return PipelinePredictor(
            settings=settings, client=client, retriever=_default_retriever(settings)
        )
    raise ValueError(f"unknown system: {system} (支持 baseline / baseline_fc / pipeline)")


async def main() -> None:
    obs.setup_logging()
    parser = argparse.ArgumentParser(description="事实核查评测")
    parser.add_argument("--data", default="data/eval_seed.jsonl")
    parser.add_argument("--system", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="out")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    samples = load_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]
    client = LLMClient(settings)
    predictor = build_predictor(args.system, client)

    summary = await run_eval(
        samples,
        predictor,
        system_name=args.system,
        dataset_name=args.data,
        concurrency=args.concurrency or settings.llm_max_concurrency,
        out_dir=args.out,
    )
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())