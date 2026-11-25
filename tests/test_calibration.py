import json

import pytest

from app.eval.calibration import (
    apply_temperature,
    brier,
    ece_mce,
    fit_calibrator,
    fit_temperature,
    load_temperature,
    loo_calibrated,
    reliability_bins,
)
from app.eval.harness import compute_metrics


def test_apply_temperature_identity_and_direction():
    assert apply_temperature(0.8, 1.0) == pytest.approx(0.8)
    # T>1 把高置信压向 0.5，T<1 抬向 1
    assert apply_temperature(0.9, 3.0) < 0.9
    assert apply_temperature(0.9, 3.0) > 0.5
    assert apply_temperature(0.6, 0.5) > 0.6
    # 0.5 是不动点
    assert apply_temperature(0.5, 4.0) == pytest.approx(0.5)


def test_brier_hand_case():
    assert brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0, abs=1e-9)
    assert brier([0.9, 0.9], [1, 0]) == pytest.approx((0.01 + 0.81) / 2)


def test_ece_perfectly_calibrated_is_zero():
    # 同一分箱内 avg_conf == accuracy（0.9 置信、90% 正确）
    conf = [0.9] * 10
    corr = [1] * 9 + [0]
    ece, mce = ece_mce(conf, corr, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert mce == pytest.approx(0.0, abs=1e-9)


def test_ece_overconfident_positive():
    conf = [0.95, 0.95, 0.95, 0.95]
    corr = [1, 1, 0, 0]  # accuracy 0.5 但 conf 0.95
    ece, mce = ece_mce(conf, corr, n_bins=10)
    assert ece == pytest.approx(0.45, abs=1e-6)
    assert mce == pytest.approx(0.45, abs=1e-6)


def test_reliability_bins_counts():
    bins = reliability_bins([0.05, 0.15, 0.95], [0, 1, 1], n_bins=10)
    assert bins[0]["count"] == 1 and bins[1]["count"] == 1 and bins[9]["count"] == 1
    assert bins[1]["accuracy"] == pytest.approx(1.0)
    assert sum(b["count"] for b in bins) == 3


def test_fit_temperature_softens_overconfidence():
    conf = [0.9] * 10
    corr = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 真实正确率 0.5
    t = fit_temperature(conf, corr)
    assert t > 1.0
    cal = [apply_temperature(p, t) for p in conf]
    ece_raw, _ = ece_mce(conf, corr)
    ece_cal, _ = ece_mce(cal, corr)
    assert ece_cal < ece_raw
    assert brier(cal, corr) < brier(conf, corr)


def test_fit_temperature_insufficient_data_returns_one():
    assert fit_temperature([0.9], [1]) == 1.0
    assert fit_temperature([0.9, 0.8], [1, 1]) == 1.0  # 单一类别无法拟合
    assert fit_temperature([], []) == 1.0


def test_loo_calibrated_length_and_range():
    conf = [0.9, 0.85, 0.95, 0.8, 0.9, 0.7]
    corr = [1, 0, 1, 0, 1, 0]
    loo = loo_calibrated(conf, corr)
    assert len(loo) == len(conf)
    assert all(0.0 < p < 1.0 for p in loo)


def test_fit_calibrator_excludes_abstain_and_none():
    records = [
        {"verdict": "supported", "correct": True, "confidence": 0.9, "abstained": False},
        {"verdict": "refuted", "correct": False, "confidence": 0.95, "abstained": False},
        {"verdict": "unverifiable", "correct": False, "confidence": 0.8, "abstained": True},
        {"verdict": "supported", "correct": True, "confidence": None, "abstained": False},
    ]
    cal = fit_calibrator(records)
    assert cal["n_attempted"] == 2
    assert len(cal["samples"]) == 2
    assert cal["temperature"] >= 0.05
    for key in ("raw", "calibrated", "loo"):
        assert set(cal[key]) == {"ece", "mce", "brier"}
    assert len(cal["reliability_raw"]) == cal["n_bins"]


def test_fit_calibrator_empty_records():
    cal = fit_calibrator([])
    assert cal["n_attempted"] == 0
    assert cal["temperature"] == 1.0
    assert cal["raw"]["ece"] == 0.0


def test_load_temperature_missing_and_valid(tmp_path):
    assert load_temperature(str(tmp_path / "nope.json")) == 1.0
    p = tmp_path / "cal.json"
    p.write_text('{"temperature": 2.5}', encoding="utf-8")
    assert load_temperature(str(p)) == 2.5
    p.write_text("not json", encoding="utf-8")
    assert load_temperature(str(p)) == 1.0  # 损坏不中断


def test_compute_metrics_includes_calibration():
    records = [
        {"gold": "supported", "verdict": "supported", "confidence": 0.9, "usage": {}, "latency_ms": 0, "cached": False},
        {"gold": "refuted", "verdict": "supported", "confidence": 0.9, "usage": {}, "latency_ms": 0, "cached": False},
        {"gold": "misleading", "verdict": "unverifiable", "confidence": 0.8, "usage": {}, "latency_ms": 0, "cached": False},
    ]
    m = compute_metrics(records)
    assert m["n_calibrated"] == 2  # 弃权样本不计入校准
    assert m["ece"] == pytest.approx(0.4, abs=1e-6)  # conf 0.9, accuracy 0.5
    assert m["brier"] == pytest.approx((0.01 + 0.81) / 2)


def test_calibrate_script_dedup_and_artifacts(tmp_path, monkeypatch):
    from scripts import calibrate

    rec_file = tmp_path / "eval.jsonl"
    rows = [
        {"id": "a", "gold": "supported", "verdict": "supported", "correct": True, "abstained": False, "confidence": 0.9},
        {"id": "b", "gold": "refuted", "verdict": "supported", "correct": False, "abstained": False, "confidence": 0.95},
        {"id": "c", "gold": "misleading", "verdict": "unverifiable", "correct": False, "abstained": True, "confidence": 0.8},
    ]
    rec_file.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "artifacts"
    monkeypatch.setattr(
        "sys.argv", ["calibrate", "--records", str(rec_file), str(rec_file), "--out", str(out)]
    )
    calibrate.main()
    meta = json.loads((out / "calibrator.json").read_text(encoding="utf-8"))
    rel = json.loads((out / "reliability.json").read_text(encoding="utf-8"))
    assert meta["n_records"] == 3  # 同文件传两次按 id 去重
    assert meta["n_attempted"] == 2
    assert meta["model"]["configured"] == "unknown"
    assert rel["temperature"] == meta["temperature"]
    assert len(rel["reliability_raw"]) == 10