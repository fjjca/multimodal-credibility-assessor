import base64
import json
import mimetypes
import re
from pathlib import Path

from openai import BadRequestError

from app.llm.client import EmptyResponseError, LLMClient, LLMResult
from app.pipeline.models import (
    Claim,
    ClaimConsistency,
    Evidence,
    ForensicsResult,
    Relation,
    normalize_consistency,
)

FORENSICS_SYSTEM = (
    "你是多模态取证员。给定一张图片和一段帖子文本（已拆成若干断言），完成四件事："
    "1) OCR：提取图中所有可见文字，没有则空字符串；"
    "2) 描述：客观描述图片主要内容（人物/物体/场景/文字），不要臆测；"
    "3) 图文一致性：逐条判断图片与每个断言的关系——consistent=图片支持该断言；"
    "inconsistent=图片与该断言矛盾（时间/地点/人物/事件不符、移花接木、旧图新用）；"
    "uncertain=图片不足以判断该断言；"
    "4) 篡改迹象：若发现明显拼接/PS/旧图新用迹象在 notes 说明，否则空字符串。"
    "只输出 JSON：{\"ocr_text\":\"...\",\"description\":\"...\",\"overall\":\"consistent|inconsistent|uncertain\","
    "\"claims\":[{\"claim_id\":\"c1\",\"consistency\":\"...\",\"note\":\"...\"}],\"notes\":\"...\"}。"
    "overall 综合所有断言：任一 inconsistent 则 overall=inconsistent。不要输出 JSON 以外的内容。"
)

IMAGE_EVIDENCE_ID = "img1"
_CONSISTENCY_TO_RELATION = {
    "consistent": "support",
    "inconsistent": "contradict",
    "uncertain": "neutral",
}


def load_image_url(image: str | None) -> str | None:
    """本地图路径→base64 data URI；http(s)/data URL 直传；缺失或非图→None。"""
    if not image:
        return None
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    if not path.is_file():
        return None
    mime, _ = mimetypes.guess_type(path.name)
    if not mime or not mime.startswith("image/"):
        return None
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def parse_forensics(content: str, claims: list[Claim]) -> ForensicsResult:
    match = re.search(r"\{.*\}", content or "", re.S)
    obj = {}
    if match:
        try:
            # strict=False 容忍 LLM 在 JSON 字符串里输出的原始换行等控制字符
            obj = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    valid_ids = {c.id for c in claims}
    items = obj.get("claims") or []
    parsed: list[ClaimConsistency] = []
    seen: set[str] = set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("claim_id", ""))
        if cid in valid_ids and cid not in seen:
            seen.add(cid)
            parsed.append(
                ClaimConsistency(
                    claim_id=cid,
                    consistency=normalize_consistency(str(it.get("consistency", ""))),
                    note=str(it.get("note", "")),
                )
            )
    return ForensicsResult(
        ocr_text=str(obj.get("ocr_text", "")),
        description=str(obj.get("description", "")),
        overall=normalize_consistency(str(obj.get("overall", ""))),
        claims=parsed,
        notes=str(obj.get("notes", "")),
    )


def image_evidence(fr: ForensicsResult) -> Evidence:
    content = f"OCR：{fr.ocr_text}\n图像描述：{fr.description}"
    if fr.notes:
        content += f"\n取证备注：{fr.notes}"
    return Evidence(
        id=IMAGE_EVIDENCE_ID,
        title="用户上传图像（VLM 取证）",
        url="",
        content=content,
        source_tier="social",
    )


def image_relations(fr: ForensicsResult, claims: list[Claim]) -> list[Relation]:
    by_id = {c.claim_id: c for c in fr.claims}
    relations: list[Relation] = []
    for claim in claims:
        cc = by_id.get(claim.id)
        consistency = cc.consistency if cc else "uncertain"
        relations.append(
            Relation(
                claim_id=claim.id,
                evidence_id=IMAGE_EVIDENCE_ID,
                relation=_CONSISTENCY_TO_RELATION.get(consistency, "neutral"),
            )
        )
    return relations


async def forensics(
    client: LLMClient,
    image_url: str | None,
    text: str,
    claims: list[Claim],
) -> tuple[ForensicsResult | None, LLMResult | None]:
    """image_url 为 base64 data URI（本地图）或 http URL（远程图）；无图返回 (None, None)。"""
    if not image_url:
        return None, None
    claims_text = "\n".join(f"{c.id}: {c.text}" for c in claims) or "（无断言）"
    user_text = (
        f"帖子文本：\n{text}\n\n断言：\n{claims_text}\n\n"
        f"请提取图中文字、描述图片，并对每个断言判断图文一致性。"
    )
    messages = [
        {"role": "system", "content": FORENSICS_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]
    try:
        result = await client.chat(
            "forensics",
            messages,
            temperature=0.0,
            max_tokens=1024,
            extra_body={"thinking": {"type": "disabled"}},
        )
    except BadRequestError:
        return ForensicsResult(notes="内容审核拒答，无法取证"), None
    except EmptyResponseError:
        return ForensicsResult(notes="VLM 多次返回空响应，取证降级"), None
    return parse_forensics(result.content, claims), result