"""事后置信度校准（纯 Python，无 numpy/scipy/sklearn）。

裁判输出单个 confidence 标量表示"对所选裁决正确的把握"，因此按二分类
P(correct)=confidence 处理：用温度缩放 p_cal = sigmoid(logit(p_raw)/T) 软化过度自信，
T>1 压低高置信、T<1 抬高。仅在 attempted（非弃权）且 confidence 非空的样本上拟合与评估。

小样本诚实性：网格搜索拟合 T 最小化 NLL；另报 LOO-CV 的 ECE/Brier，避免"在同一批数据上
既拟合又评估"的乐观偏差。
"""

import json
import math
from pathlib import Path

_EPS = 1e-6


def load_temperature(path: str) -> float:
    """读取校准器 artifact 的温度；缺失/损坏时返回 1.0（不校准），保证流水线不中断。"""
    p = Path(path)
    if not p.exists():
        return 1.0
    try:
        return float(json.loads(p.read_text(encoding="utf-8")).get("temperature", 1.0))
    except (json.JSONDecodeError, ValueError, TypeError, OSError):
        return 1.0


def _clamp(p: float) -> float:
    return min(max(float(p), _EPS), 1.0 - _EPS)


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def apply_temperature(conf: float, temperature: float) -> float:
    if temperature == 1.0:
        return _clamp(conf)
    return _sigmoid(_logit(conf) / temperature)


def _nll(confidences: list[float], corrects: list[int], temperature: float) -> float:
    total = 0.0
    for p, y in zip(confidences, corrects):
        q = _clamp(apply_temperature(p, temperature))
        total -= y * math.log(q) + (1 - y) * math.log(1 - q)
    return total / len(confidences) if confidences else 0.0


def brier(confidences: list[float], corrects: list[int]) -> float:
    if not confidences:
        return 0.0
    return sum((_clamp(p) - y) ** 2 for p, y in zip(confidences, corrects)) / len(confidences)


def reliability_bins(
    confidences: list[float], corrects: list[int], n_bins: int = 10
) -> list[dict]:
    bins = [
        {"lower": i / n_bins, "upper": (i + 1) / n_bins, "count": 0, "conf_sum": 0.0, "correct_sum": 0}
        for i in range(n_bins)
    ]
    for p, y in zip(confidences, corrects):
        p = _clamp(p)
        idx = min(int(p * n_bins), n_bins - 1)
        b = bins[idx]
        b["count"] += 1
        b["conf_sum"] += p
        b["correct_sum"] += y
    out = []
    for b in bins:
        c = b["count"]
        out.append(
            {
                "lower": b["lower"],
                "upper": b["upper"],
                "count": c,
                "avg_conf": b["conf_sum"] / c if c else 0.0,
                "accuracy": b["correct_sum"] / c if c else 0.0,
            }
        )
    return out


def ece_mce(confidences: list[float], corrects: list[int], n_bins: int = 10) -> tuple[float, float]:
    bins = reliability_bins(confidences, corrects, n_bins)
    n = len(confidences)
    if n == 0:
        return 0.0, 0.0
    ece = sum(b["count"] / n * abs(b["accuracy"] - b["avg_conf"]) for b in bins)
    mce = max((abs(b["accuracy"] - b["avg_conf"]) for b in bins if b["count"]), default=0.0)
    return ece, mce


def fit_temperature(
    confidences: list[float], corrects: list[int], *, t_min: float = 0.05, t_max: float = 10.0, steps: int = 200
) -> float:
    """网格搜索最小化 NLL 的温度。样本不足时返回 1.0（不校准）。"""
    if len(confidences) < 2 or len(set(corrects)) < 2:
        return 1.0
    best_t, best_loss = 1.0, _nll(confidences, corrects, 1.0)
    for i in range(steps + 1):
        t = t_min + (t_max - t_min) * i / steps
        loss = _nll(confidences, corrects, t)
        if loss < best_loss - 1e-12:
            best_t, best_loss = t, loss
    return best_t


def loo_calibrated(confidences: list[float], corrects: list[int]) -> list[float]:
    """留一法：对每个样本用其余样本拟合 T 再校准该样本，得到诚实的 out-of-fold 概率。"""
    n = len(confidences)
    if n < 3:
        t = fit_temperature(confidences, corrects)
        return [apply_temperature(p, t) for p in confidences]
    out = []
    for i in range(n):
        train_p = confidences[:i] + confidences[i + 1 :]
        train_y = corrects[:i] + corrects[i + 1 :]
        t = fit_temperature(train_p, train_y)
        out.append(apply_temperature(confidences[i], t))
    return out


def _attempted_pairs(records: list[dict]) -> tuple[list[float], list[int]]:
    confidences, corrects = [], []
    for r in records:
        if r.get("abstained") or r.get("verdict") == "unverifiable":
            continue
        p = r.get("confidence")
        if p is None:
            continue
        confidences.append(float(p))
        corrects.append(1 if r.get("correct") else 0)
    return confidences, corrects


def fit_calibrator(records: list[dict], n_bins: int = 10) -> dict:
    """从 eval 记录拟合校准器，返回可序列化 dict（含 before/after 与 LOO 指标、可靠性分箱）。"""
    confidences, corrects = _attempted_pairs(records)
    n = len(confidences)
    ece_raw, mce_raw = ece_mce(confidences, corrects, n_bins)
    brier_raw = brier(confidences, corrects)
    temperature = fit_temperature(confidences, corrects)
    cal = [apply_temperature(p, temperature) for p in confidences]
    ece_cal, mce_cal = ece_mce(cal, corrects, n_bins)
    brier_cal = brier(cal, corrects)
    loo = loo_calibrated(confidences, corrects)
    ece_loo, mce_loo = ece_mce(loo, corrects, n_bins)
    brier_loo = brier(loo, corrects)
    return {
        "temperature": temperature,
        "n_attempted": n,
        "n_bins": n_bins,
        "raw": {"ece": ece_raw, "mce": mce_raw, "brier": brier_raw},
        "calibrated": {"ece": ece_cal, "mce": mce_cal, "brier": brier_cal},
        "loo": {"ece": ece_loo, "mce": mce_loo, "brier": brier_loo},
        "reliability_raw": reliability_bins(confidences, corrects, n_bins),
        "reliability_calibrated": reliability_bins(cal, corrects, n_bins),
        "samples": [
            {"confidence": p, "correct": y, "calibrated": c, "loo": l}
            for p, y, c, l in zip(confidences, corrects, cal, loo)
        ],
    }