import json

from app.eval.attribution import attribute_record, attribution_summary, write_attribution_report


def _record(**overrides):
    base = {
        "id": "x",
        "gold": "supported",
        "verdict": "refuted",
        "correct": False,
        "details": {
            "retrieval_enabled": True,
            "forensics": None,
            "graph": {"n_evidence": 2, "n_external_evidence": 2, "n_support": 1, "n_contradict": 1, "n_neutral": 0},
            "graph_summary": [],
        },
    }
    base.update(overrides)
    return base


def test_correct_record_has_no_failure():
    assert attribute_record(_record(correct=True, verdict="supported"))["category"] == "none"


def test_old_or_baseline_record_is_inconclusive():
    assert attribute_record(_record(details={}))["category"] == "inconclusive"


def test_ocr_conflict_is_perception_failure():
    record = _record()
    record["details"]["forensics"] = {"notes": "", "ocr_crosscheck": {"status": "conflict", "conflicts": ["136.90"]}}
    result = attribute_record(record)
    assert result["category"] == "perception"
    assert "OCR 数字冲突" in result["signals"][0]


def test_no_retrieved_evidence_is_retrieval_failure():
    record = _record()
    record["details"]["graph"] = {"n_evidence": 0, "n_support": 0, "n_contradict": 0, "n_neutral": 0}
    assert attribute_record(record)["category"] == "retrieval"


def test_image_only_active_evidence_is_adjudication_failure():
    record = _record()
    record["details"]["graph"] = {
        "n_evidence": 1, "n_external_evidence": 0,
        "n_support": 1, "n_contradict": 0, "n_neutral": 0,
    }
    record["details"]["forensics"] = {"notes": "", "ocr_crosscheck": {"status": "agree"}, "overall": "consistent"}
    assert attribute_record(record)["category"] == "adjudication"


def test_all_neutral_is_relation_failure():
    record = _record()
    record["details"]["graph"] = {"n_evidence": 3, "n_external_evidence": 3, "n_support": 0, "n_contradict": 0, "n_neutral": 3}
    assert attribute_record(record)["category"] == "relation"


def test_active_evidence_wrong_verdict_is_adjudication_failure():
    assert attribute_record(_record())["category"] == "adjudication"


def test_summary_and_files(tmp_path):
    records = [_record(), _record(id="ok", correct=True, verdict="supported"), _record(id="old", details={})]
    summary = write_attribution_report(records, tmp_path, "demo")
    assert summary["n_records"] == 3
    assert summary["n_failures"] == 2
    assert summary["counts"]["adjudication"] == 1
    assert summary["counts"]["inconclusive"] == 1
    assert (tmp_path / "failure_attribution_demo.json").exists()
    assert (tmp_path / "failure_attribution_demo.md").exists()
    saved = json.loads((tmp_path / "failure_attribution_demo.json").read_text(encoding="utf-8"))
    assert saved == summary


def test_rates_sum_to_one_when_failures_exist():
    records = [_record(), _record(id="old", details={})]
    summary = attribution_summary(records)
    assert sum(summary["failure_rates"].values()) == 1.0
