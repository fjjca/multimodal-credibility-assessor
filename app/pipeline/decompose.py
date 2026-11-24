import json
import re

from app.llm.client import LLMClient, LLMResult
from app.pipeline.models import Claim

DECOMPOSE_SYSTEM = (
    "你是断言拆解器。把给定社交媒体帖子拆成若干原子断言：每条只含一个可独立核查的事实主张。"
    "只拆帖子里明确写出来的具体事实，不得使用「某种」「某项」「之类」这类含糊表述，"
    "也不得补充帖子没写的内容。帖子只有一句话时就只输出一条，不要为了凑数拆分。"
    "断言必须是原文的忠实摘录：尽量保留原文里的年份数字、地点、机构名、专有名词，"
    "禁止把具体事实概括成更笼统的说法（例如原文「2020年东京奥运会现场」"
    "不得写成「2020年发生了一件事情」）。"
    "忽略纯情绪宣泄与主观观点。只输出一个 JSON 数组，元素形如 {\"text\": \"断言文本\"}。"
    "若帖子没有任何可核查的事实主张，输出空数组 []。不要输出数组以外的内容。"
)

# 拆解器有时会用「某种事件将在某年发生」这类空话凑数：这样的断言不可核查，
# 检索会拉回一堆同主题但无关的页面，裁判再把它们当"支持"，图文矛盾就被冲掉了
# （mm05 曾因此把"图片写着2023、帖子说2024"判成真实可信 0.95）
VACUOUS_CLAIM = re.compile(
    r"某种|某项|某类|某个|某次|之类|若干|某些|即将到来|即将发生|即将开始|即将出现|将会发生|将会出现"
    r"|发生了一?件事?情?|有(这|那)?么?一?件事|出了一?件事|有过一?件事"
)

_CJK = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


def _bigrams(s: str) -> set[str]:
    chars = _CJK.findall(s)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _anchors(s: str) -> set[str]:
    """实词锚点：中文二元组 + 英文词。数字不算锚点——只有数字重合说明断言
    把具体事实（地点、事件、主体）全丢了，只剩一个年份。"""
    return _bigrams(s) | {w.lower() for w in _ASCII_WORD.findall(s)}


def _keeps_source_detail(claim: str, source: str) -> bool:
    """断言必须与原文有实词级重合。只剩数字重合的断言是"降级改写"：
    把「东京奥运会现场」压成「2020年发生了一件事情」，VLM 对着空话判一致，
    矛盾就消失了（mm01 曾因此把 PARIS 2024 图判成真实可信 0.75）。"""
    return bool(_anchors(claim) & _anchors(source))


def parse_claims(
    content: str, max_claims: int, source_text: str = ""
) -> list[Claim]:
    match = re.search(r"\[.*\]", content or "", re.S)
    items = []
    if match:
        try:
            items = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            items = []
    texts: list[str] = []
    for it in items if isinstance(items, list) else []:
        text = it.get("text", "") if isinstance(it, dict) else str(it)
        text = (text or "").strip()
        if not text or VACUOUS_CLAIM.search(text):
            continue
        if source_text and not _keeps_source_detail(text, source_text):
            continue
        if text not in texts:
            texts.append(text)
    return [Claim(id=f"c{i + 1}", text=t) for i, t in enumerate(texts[:max_claims])]


async def decompose(
    client: LLMClient, text: str, max_claims: int = 5
) -> tuple[list[Claim], LLMResult]:
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": f"帖子文本：\n{text}\n\n最多拆 {max_claims} 条断言。"},
    ]
    result = await client.chat(
        "decompose",
        messages,
        temperature=0.0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )
    claims = parse_claims(result.content, max_claims, source_text=text)
    if not claims and text.strip():
        claims = [Claim(id="c1", text=text.strip())]
    return claims, result