"""本地 OCR 与 VLM 的数字级交叉校验。"""

import asyncio
import re
import threading
import time
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

from app.pipeline.models import Claim, ForensicsResult, OcrLine, OcrResult

_ENGINE = "rapidocr-onnxruntime"
_NUMBER_RE = re.compile(r"(?<!\d)\d+(?:[.,]\d+)*(?!\d)")
_NUMERIC_CLAIM_RE = re.compile(r"金额|消费|价格|合计|应付|找零|数量|编号|电话|[¥￥]\s*\d|\d+(?:\.\d+)?\s*元")
_ENGINE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _run_engine(path: str) -> tuple[str, list[OcrLine]]:
    with _ENGINE_LOCK:
        result, _ = _engine()(path)
    lines = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        try:
            confidence = float(item[2]) if len(item) > 2 else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        lines.append(OcrLine(text=text, confidence=confidence))
    return "\n".join(line.text for line in lines), lines


async def run_local_ocr(image_path: str | None) -> OcrResult:
    if not image_path or not Path(image_path).is_file():
        return OcrResult(engine=_ENGINE, error="本地图像不存在，跳过 OCR")
    started = time.perf_counter()
    try:
        text, lines = await asyncio.to_thread(_run_engine, image_path)
    except Exception as exc:
        return OcrResult(
            engine=_ENGINE,
            error=f"本地 OCR 不可用：{type(exc).__name__}: {exc}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    return OcrResult(
        engine=_ENGINE,
        available=True,
        text=text,
        lines=lines,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


def extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def normalize_number(token: str) -> str:
    value = (token or "").strip().replace(",", "")
    if value.count(".") > 1:
        return value.replace(".", "")
    try:
        number = Decimal(value)
        if number == number.to_integral():
            return str(number.quantize(Decimal(1)))
        return format(number.normalize(), "f")
    except InvalidOperation:
        return value


def crosscheck_numbers(vlm_text: str, local_text: str) -> dict:
    vlm_numbers = extract_numbers(vlm_text)
    local_numbers = extract_numbers(local_text)
    local_set = {normalize_number(n) for n in local_numbers}
    agreements = [n for n in vlm_numbers if normalize_number(n) in local_set]
    conflicts = [n for n in vlm_numbers if normalize_number(n) not in local_set]
    if not local_text.strip():
        status = "vlm_only"
    elif not vlm_numbers:
        status = "ocr_only"
    elif conflicts:
        status = "conflict"
    else:
        status = "agree"
    return {
        "status": status,
        "vlm_numbers": vlm_numbers,
        "local_numbers": local_numbers,
        "agreements": agreements,
        "conflicts": conflicts,
    }


def augment_forensics_with_ocr(
    fr: ForensicsResult,
    ocr: OcrResult,
    claims: list[Claim],
) -> ForensicsResult:
    result = fr.model_copy(deep=True)
    result.vlm_ocr_text = fr.ocr_text
    result.ocr_engine = ocr.engine
    result.ocr_lines = ocr.lines
    crosscheck = crosscheck_numbers(fr.ocr_text, ocr.text)
    crosscheck.update({"engine": ocr.engine})
    result.ocr_crosscheck = crosscheck
    if not ocr.available or not ocr.text.strip():
        note = f"本地 OCR 未提供文本（{ocr.error or '无识别结果'}），数字取证仍以 VLM 为准"
        result.notes = f"{result.notes}；{note}" if result.notes else note
        return result

    result.ocr_text = ocr.text
    local_set = {normalize_number(n) for n in crosscheck["local_numbers"]}
    if crosscheck["status"] == "conflict":
        note = f"数字冲突：VLM={crosscheck['conflicts']}，本地 OCR 以其识别文本为准"
        result.notes = f"{result.notes}；{note}" if result.notes else note

    by_id = {c.claim_id: c for c in result.claims}
    for claim in claims:
        if not _NUMERIC_CLAIM_RE.search(claim.text):
            continue
        numbers = extract_numbers(claim.text)
        if not numbers or not local_set:
            continue
        matched = [n for n in numbers if normalize_number(n) in local_set]
        missing = [n for n in numbers if normalize_number(n) not in local_set]
        cc = by_id.get(claim.id)
        if cc is None:
            continue
        if missing and not matched:
            cc.consistency = "inconsistent"
            cc.note = f"{cc.note}；本地 OCR 未检出断言数字 {missing}".strip("；")
        elif not missing:
            cc.consistency = "consistent"
            cc.note = f"{cc.note}；本地 OCR 数字与断言一致".strip("；")
        else:
            cc.consistency = "uncertain"
            cc.note = f"{cc.note}；本地 OCR 仅匹配部分数字，缺 {missing}".strip("；")

    values = [c.consistency for c in result.claims]
    if "inconsistent" in values:
        result.overall = "inconsistent"
    elif "consistent" in values:
        result.overall = "consistent"
    return result
