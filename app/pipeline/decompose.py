import json
import re

from app.llm.client import LLMClient, LLMResult
from app.pipeline.models import Claim

DECOMPOSE_SYSTEM = (
    "你是断言拆解器。把给定社交媒体帖子拆成若干原子断言：每条只含一个可独立核查的事实主张。"
    "忽略纯情绪宣泄与主观观点。只输出一个 JSON 数组，元素形如 {\"text\": \"断言文本\"}。"
    "若帖子没有任何可核查的事实主张，输出空数组 []。不要输出数组以外的内容。"
)


def parse_claims(content: str, max_claims: int) -> list[Claim]:
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
        if text and text not in texts:
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
    claims = parse_claims(result.content, max_claims)
    if not claims and text.strip():
        claims = [Claim(id="c1", text=text.strip())]
    return claims, result