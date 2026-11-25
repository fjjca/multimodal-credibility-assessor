import json
import re

from pydantic import BaseModel, field_validator

# 真实标签三类；模型可额外输出 unverifiable 表示弃权
VERDICTS = ("supported", "refuted", "misleading")
ABSTAIN = "unverifiable"
ALL_VERDICTS = VERDICTS + (ABSTAIN,)

_SYNONYMS = {
    "supported": ["supported", "true", "real", "accurate", "可信", "真实", "支持", "正确"],
    "refuted": ["refuted", "false", "fake", "不实", "虚假", "谣言", "错误"],
    "misleading": ["misleading", "out-of-context", "out of context", "存疑", "误导", "部分真实", "移花接木"],
    "unverifiable": ["unverifiable", "unknown", "uncertain", "abstain", "无法验证", "不确定", "弃权"],
}


def parse_verdict(raw: str) -> str:
    text = (raw or "").strip().lower()
    for verdict, words in _SYNONYMS.items():
        if text in words:
            return verdict
    for verdict, words in _SYNONYMS.items():
        if any(w in text for w in words):
            return verdict
    return ABSTAIN


def parse_baseline_output(content: str) -> tuple[str, float | None, str]:
    match = re.search(r"\{.*\}", content or "", re.S)
    obj = {}
    if match:
        try:
            obj = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            obj = {}
    verdict = parse_verdict(str(obj.get("verdict", "")))
    conf = obj.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    reason = str(obj.get("reason", ""))
    return verdict, conf, reason


class Sample(BaseModel):
    id: str
    text: str
    label: str
    image: str | None = None
    source: str = "seed"
    raw_label: str | None = None

    @field_validator("label")
    @classmethod
    def label_in_verdicts(cls, v: str) -> str:
        if v not in VERDICTS:
            raise ValueError(f"label must be one of {VERDICTS}, got {v}")
        return v


class Prediction(BaseModel):
    verdict: str
    confidence: float | None = None
    reason: str = ""
    raw: str = ""
    usage: dict = {}
    latency_ms: float = 0.0
    cached: bool = False
    details: dict = {}