import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path

from openai import BadRequestError, RateLimitError

from app import obs
from app.llm.client import (
    CircuitOpenError,
    EmptyResponseError,
    LLMClient,
    LLMResult,
    error_body,
    is_moderation_error,
    unsupported_param_detail,
)
from app.pipeline.models import (
    Claim,
    ClaimConsistency,
    Evidence,
    ForensicsResult,
    Relation,
    normalize_consistency,
)
from app.pipeline.ocr import augment_forensics_with_ocr, run_local_ocr

FORENSICS_SYSTEM = (
    "你是多模态取证员。给定一张图片和一段帖子文本（已拆成若干断言），完成四件事："
    "1) OCR：提取图中所有可见文字，没有则空字符串；"
    "2) 描述：客观描述图片主要内容（人物/物体/场景/文字），不要臆测；"
    "3) 图文一致性：逐条判断图片与每个断言的关系——consistent=图片支持该断言；"
    "inconsistent=图片与该断言矛盾（时间/地点/人物/事件不符、移花接木、旧图新用）；"
    "uncertain=图片不足以判断该断言；"
    "另外必须与帖子原文直接对照：若图片里的时间、地点、事件、机构与原文明写的内容冲突，"
    "无论断言表述是否笼统，都要判为 inconsistent；"
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

# 内容审核拒答的判定已挪到 app.llm.client（关系判定阶段也要用同一套判定）


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
        vlm_ocr_text=str(obj.get("ocr_text", "")),
        description=str(obj.get("description", "")),
        overall=normalize_consistency(str(obj.get("overall", ""))),
        claims=parsed,
        notes=str(obj.get("notes", "")),
    )


def image_evidence(fr: ForensicsResult) -> Evidence:
    content = f"OCR（{fr.ocr_engine}）：{fr.ocr_text}\n图像描述：{fr.description}"
    if fr.vlm_ocr_text and fr.vlm_ocr_text != fr.ocr_text:
        content += f"\nVLM OCR：{fr.vlm_ocr_text}"
    if fr.notes:
        content += f"\n取证备注：{fr.notes}"
    return Evidence(
        id=IMAGE_EVIDENCE_ID,
        title="用户上传图像（本地 OCR + VLM 取证）" if fr.ocr_engine != "vlm" else "用户上传图像（VLM 取证）",
        url="",
        content=content,
        source_tier="uploaded",
    )


def image_query_hints(fr: ForensicsResult | None, limit: int = 60) -> dict[str, str]:
    """给断言配检索关键词（图像 OCR/描述），只对 VLM 能与图像对照的断言生效。

    否则断言原文会检索出关于该事件的真实网页冒充支持证据（"照片拍摄于2020年东京奥运会"
    → 新华网报道），凭空抬高支持权重。
    """
    if fr is None:
        return {}
    hint = " ".join(x for x in (fr.ocr_text.strip(), fr.description.strip()) if x).strip()
    if not hint:
        return {}
    hint = hint[:limit]
    return {c.claim_id: hint for c in fr.claims if c.consistency != "uncertain"}


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
    except BadRequestError as exc:
        body = error_body(exc)
        obs.log.warning(
            "forensics request rejected",
            extra={
                "status": exc.status_code,
                "code": body.get("code"),
                "body_text": str(body.get("message") or body.get("raw") or "")[:200],
            },
        )
        detail = unsupported_param_detail(exc)
        if detail:
            return ForensicsResult(
                notes=f"VLM 请求参数不被支持（{detail}），取证降级", overall="uncertain"
            ), None
        if is_moderation_error(exc):
            return ForensicsResult(notes="内容审核拒答，无法取证"), None
        return ForensicsResult(notes="VLM 请求被拒（HTTP 400），取证降级"), None
    except EmptyResponseError:
        return ForensicsResult(notes="VLM 多次返回空响应，取证降级"), None
    except (RateLimitError, CircuitOpenError):
        # 免费档配额/限流：取证不可用，但流水线继续（文本核查照常出结论）
        return ForensicsResult(
            notes="VLM 限流或不可用，图文一致性未评估（取证降级，流水线继续）",
            overall="uncertain",
        ), None
    return parse_forensics(result.content, claims), result


async def collect_forensics(
    client: LLMClient,
    image_path: str | None,
    text: str,
    claims: list[Claim],
    settings,
) -> tuple[ForensicsResult | None, LLMResult | None]:
    """VLM 语义取证与本地 OCR 并行，再做数字级交叉校验。"""
    image_url = load_image_url(image_path)
    use_local_ocr = bool(settings.enable_local_ocr and image_path and image_url)
    vlm_task = forensics(client, image_url, text, claims)
    ocr_task = run_local_ocr(image_path) if use_local_ocr else asyncio.sleep(0, result=None)
    (fr, fr_res), ocr = await asyncio.gather(vlm_task, ocr_task)
    if fr is not None and ocr is not None:
        fr = augment_forensics_with_ocr(fr, ocr, claims)
    return fr, fr_res
