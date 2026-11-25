from pathlib import Path

import pytest

from app.pipeline.forensics import image_evidence
from app.pipeline.models import Claim, ClaimConsistency, ForensicsResult, OcrResult
from app.pipeline.ocr import (
    augment_forensics_with_ocr,
    crosscheck_numbers,
    extract_numbers,
    normalize_number,
    run_local_ocr,
)


def test_extract_and_normalize_numbers():
    assert extract_numbers("合计 150.00 元，交易号 665328") == ["150.00", "665328"]
    assert normalize_number("150.00") == "150"
    assert normalize_number("1,500.00") == "1500"
    assert normalize_number("2023.06.25") == "20230625"


def test_crosscheck_detects_vlm_digit_error():
    result = crosscheck_numbers("合计金额 136.90", "合计金额：150.00\n应付金额：147.40")
    assert result["status"] == "conflict"
    assert result["conflicts"] == ["136.90"]
    assert result["agreements"] == []


def test_crosscheck_accepts_equivalent_decimal():
    result = crosscheck_numbers("合计金额 150 元", "合计金额：150.00")
    assert result["status"] == "agree"
    assert result["agreements"] == ["150"]


def _fr(consistency="inconsistent"):
    return ForensicsResult(
        ocr_text="合计金额 136.90",
        vlm_ocr_text="合计金额 136.90",
        description="超市小票",
        overall=consistency,
        claims=[ClaimConsistency(claim_id="c1", consistency=consistency, note="VLM 读数")],
    )


def _ocr(text):
    return OcrResult(engine="rapidocr-onnxruntime", available=True, text=text, latency_ms=42)


def test_local_ocr_corrects_amount_claim():
    claims = [Claim(id="c1", text="这张小票显示本次消费合计金额为150元。")]
    result = augment_forensics_with_ocr(_fr(), _ocr("合计金额：150.00"), claims)
    assert result.ocr_text == "合计金额：150.00"
    assert result.ocr_crosscheck["status"] == "conflict"
    assert result.claims[0].consistency == "consistent"
    assert result.overall == "consistent"
    assert "数字冲突" in result.notes


def test_local_ocr_refutes_wrong_amount_claim():
    claims = [Claim(id="c1", text="这张小票显示本次消费合计金额为1500元。")]
    result = augment_forensics_with_ocr(_fr("consistent"), _ocr("合计金额：150.00"), claims)
    assert result.claims[0].consistency == "inconsistent"
    assert result.overall == "inconsistent"


def test_numeric_agreement_does_not_override_location_claim():
    claims = [Claim(id="c1", text="这张照片拍摄于2024年上海马拉松现场。")]
    fr = _fr("inconsistent")
    result = augment_forensics_with_ocr(fr, _ocr("2024 北京马拉松"), claims)
    assert result.claims[0].consistency == "inconsistent"


def test_unavailable_local_ocr_keeps_vlm():
    claims = [Claim(id="c1", text="这张小票显示本次消费合计金额为150元。")]
    result = augment_forensics_with_ocr(_fr(), OcrResult(error="模型缺失"), claims)
    assert result.ocr_text == "合计金额 136.90"
    assert result.claims[0].consistency == "inconsistent"
    assert "本地 OCR 未提供文本" in result.notes


def test_image_evidence_names_ocr_source():
    fr = augment_forensics_with_ocr(
        _fr(), _ocr("合计金额：150.00"),
        [Claim(id="c1", text="这张小票显示本次消费合计金额为150元。")],
    )
    evidence = image_evidence(fr)
    assert "本地 OCR + VLM" in evidence.title
    assert "OCR（rapidocr-onnxruntime）" in evidence.content
    assert "VLM OCR" in evidence.content


@pytest.mark.skipif(
    not Path("data/demo_photos/supermarket_receipt.jpg").exists(),
    reason="demo photo is gitignored; run scripts.fetch_demo_photos first",
)
async def test_real_receipt_ocr_reads_total():
    result = await run_local_ocr("data/demo_photos/supermarket_receipt.jpg")
    assert result.available is True
    assert "150.00" in result.text
    assert result.latency_ms >= 0
