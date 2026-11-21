import time

from app.config import Settings, get_settings
from app.eval.schema import Prediction, Sample
from app.llm.client import LLMClient
from app.pipeline.runner import run_pipeline


class PipelinePredictor:
    """把完整流水线包装成评测 Predictor，与单模型基线同口径对比。

    accuracy 分母含弃权（与 baseline_fc 一致）；另报 abstention_rate 体现证据不足时的拒答。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: LLMClient | None = None,
        retriever=None,
    ):
        self.settings = settings or get_settings()
        self.client = client
        self.retriever = retriever

    async def __call__(self, sample: Sample) -> Prediction:
        async def emit(event: str, data: dict) -> None:
            pass

        started = time.perf_counter()
        result = await run_pipeline(
            sample.text,
            sample.image,
            emit,
            settings=self.settings,
            client=self.client,
            retriever=self.retriever,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        if result.get("pending_review"):
            raise RuntimeError(
                "HITL 人工复核已触发，流水线暂停等待裁决；评测需关闭 ENABLE_HITL"
            )
        usage = {
            "total_tokens": result["usage"]["total_tokens"],
            "n_calls": result["usage"]["n_calls"],
        }
        return Prediction(
            verdict=result["verdict"],
            confidence=result["confidence"],
            reason=result["reason"],
            usage=usage,
            latency_ms=latency_ms,
            cached=result["usage"]["cached"] > 0,
        )