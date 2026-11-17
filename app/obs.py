"""轻量可观测性：结构化 JSON 日志 + Prometheus 指标，零外部服务。

日志走 stdout（容器友好），指标在根 /metrics 暴露供 Prometheus 抓取。
token 只统计实时调用，缓存命中不重复计费；延迟直方图用适配 LLM 秒级耗时的桶。
"""

import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# --- 结构化 JSON 日志 -------------------------------------------------------

_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)
_HANDLER_TAG = "_credibility_json_handler"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """幂等地给根 logger 挂一个 JSON stdout handler（由 lifespan 调用）。"""
    root = logging.getLogger()
    if any(getattr(h, _HANDLER_TAG, False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_TAG, True)
    root.addHandler(handler)
    root.setLevel(level)


# --- Prometheus 指标 --------------------------------------------------------

_LLM_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120)
_VERIFY_BUCKETS = (5, 10, 30, 60, 120, 300, 600)

LLM_CALLS = Counter("llm_calls_total", "LLM 调用次数", ["role", "model", "cached"])
LLM_TOKENS = Counter("llm_tokens_total", "实时调用消耗的 token（缓存命中不计）", ["role"])
LLM_LATENCY = Histogram("llm_latency_seconds", "实时 LLM 调用耗时", ["role"], buckets=_LLM_BUCKETS)
STAGE_LATENCY = Histogram("stage_latency_seconds", "流水线各阶段耗时", ["stage"], buckets=_LLM_BUCKETS)

VERIFY_REQUESTS = Counter("verify_requests_total", "收到的核查请求", ["has_image"])
VERIFY_LATENCY = Histogram("verify_latency_seconds", "整条核查流水线耗时", buckets=_VERIFY_BUCKETS)
VERIFY_ERRORS = Counter("verify_errors_total", "核查失败次数")
VERDICTS = Counter("verdicts_total", "裁决结论分布", ["verdict"])
ESCALATIONS = Counter("escalations_total", "低置信度升级到强裁判的次数")

log = logging.getLogger("credibility")


def record_llm_call(role: str, model: str, *, cached: bool, tokens: int = 0, latency_ms: float = 0.0) -> None:
    LLM_CALLS.labels(role=role, model=model, cached="true" if cached else "false").inc()
    if not cached:
        if tokens:
            LLM_TOKENS.labels(role=role).inc(tokens)
        LLM_LATENCY.labels(role=role).observe(max(latency_ms, 0.0) / 1000.0)


def record_stage(stage: str, seconds: float) -> None:
    STAGE_LATENCY.labels(stage=stage).observe(max(seconds, 0.0))


def record_verify_request(has_image: bool) -> None:
    VERIFY_REQUESTS.labels(has_image="true" if has_image else "false").inc()


def record_verify(*, latency_s: float, verdict: str, escalated: bool) -> None:
    VERIFY_LATENCY.observe(max(latency_s, 0.0))
    VERDICTS.labels(verdict=verdict).inc()
    if escalated:
        ESCALATIONS.inc()


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST