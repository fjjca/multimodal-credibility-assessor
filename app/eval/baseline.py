from app.eval.schema import Prediction, Sample, parse_baseline_output
from app.llm.client import LLMClient

SYSTEM = (
    "你是一个事实核查裁决器。仅凭给定帖子文本，零样本判断其可信度，不要检索外部证据。"
    "社交媒体帖子往往简短，你仍需结合文本的具体性、措辞与常识做出最可能的判断，"
    "尽量给出 supported 或 refuted，不要轻易弃权。"
    "只输出一个 JSON 对象，字段："
    'verdict（supported/refuted/misleading/unverifiable 之一）、'
    'confidence（0到1的小数）、reason（一句话理由）。'
    "supported=内容真实可信；refuted=内容虚假或捏造；misleading=部分真实但误导（如旧图新用、断章取义）；"
    "unverifiable=仅在文本完全无法判断时才使用。"
)

SYSTEM_FORCE = (
    "你是一个事实核查二分类器。仅凭给定帖子文本，零样本判断其可信度，不要检索外部证据。"
    "必须在 supported 与 refuted 之间二选一，禁止弃权，禁止输出 unverifiable 或 misleading。"
    "社交媒体帖子往往简短，你仍需结合文本的具体性、措辞与常识给出最可能的一项。"
    "supported=内容真实可信；refuted=内容虚假或捏造。"
    "只输出一个 JSON 对象，字段："
    'verdict（supported 或 refuted）、confidence（0到1的小数）、reason（一句话理由）。'
)


class BaselinePredictor:
    """单模型 zero-shot 直判，作为后续完整流水线要打败的基线。

    force_choice=True 时为 Fakeddit 标准二分类协议：强制 supported/refuted，
    仍弃权则按错计（accuracy 分母含弃权）。
    """

    def __init__(self, client: LLMClient, force_choice: bool = False):
        self.client = client
        self.system = SYSTEM_FORCE if force_choice else SYSTEM

    async def __call__(self, sample: Sample) -> Prediction:
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": sample.text},
        ]
        result = await self.client.chat(
            "judge",
            messages,
            temperature=0.0,
            max_tokens=512,
            extra_body={"thinking": {"type": "disabled"}},
        )
        verdict, confidence, reason = parse_baseline_output(result.content)
        return Prediction(
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            raw=result.content,
            usage=result.usage,
            latency_ms=result.latency_ms,
            cached=result.cached,
        )