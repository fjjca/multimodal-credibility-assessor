from urllib.parse import urlparse

from pydantic import BaseModel

SOURCE_TIERS = ("uploaded", "official", "mainstream", "social", "unknown")
TIER_WEIGHTS = {"uploaded": 1.0, "official": 1.0, "mainstream": 0.8, "social": 0.4, "unknown": 0.5}

RELATIONS = ("support", "contradict", "neutral")

_OFFICIAL = ("gov.cn", ".gov", "edu.cn", ".edu", "who.int", "un.org", "xinhuanet.com", "people.com.cn")
_MAINSTREAM = (
    "bbc.", "reuters.", "apnews.", "nytimes.", "theguardian.", "cnn.",
    "caixin.", "yicai.", "thepaper.cn", "chinadaily.", "infzm.",
)
_SOCIAL = (
    "reddit.", "weibo.", "twitter.", "x.com", "facebook.", "tiktok.",
    "zhihu.", "bilibili.", "youtube.", "instagram.", "douyin.",
)


def classify_source_tier(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    if not host:
        return "unknown"
    if any(d in host for d in _OFFICIAL):
        return "official"
    if any(d in host for d in _MAINSTREAM):
        return "mainstream"
    if any(d in host for d in _SOCIAL):
        return "social"
    return "unknown"


def domain_of(url: str) -> str:
    return urlparse(url or "").netloc.lower()


class Claim(BaseModel):
    id: str
    text: str


class Evidence(BaseModel):
    id: str
    title: str = ""
    url: str = ""
    content: str = ""
    source_tier: str = "unknown"
    score: float = 0.0


class Relation(BaseModel):
    claim_id: str
    evidence_id: str
    relation: str


CONSISTENCY = ("consistent", "inconsistent", "uncertain")

_CONSISTENCY_SYNONYMS = {
    "consistent": ["consistent", "support", "supports", "match", "matches", "一致", "支持", "相符"],
    "inconsistent": ["inconsistent", "contradict", "contradicts", "mismatch", "conflict", "矛盾", "不符", "不一致", "冲突"],
    "uncertain": ["uncertain", "unknown", "unclear", "neutral", "无法判断", "不确定", "未知"],
}


def normalize_consistency(raw: str) -> str:
    text = (raw or "").strip().lower()
    for value, words in _CONSISTENCY_SYNONYMS.items():
        if text in words:
            return value
    for value, words in _CONSISTENCY_SYNONYMS.items():
        if any(w in text for w in words):
            return value
    return "uncertain"


class ClaimConsistency(BaseModel):
    claim_id: str
    consistency: str = "uncertain"
    note: str = ""


class ForensicsResult(BaseModel):
    ocr_text: str = ""
    description: str = ""
    overall: str = "uncertain"
    claims: list[ClaimConsistency] = []
    notes: str = ""