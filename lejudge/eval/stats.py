"""Statistics exactly as pre-registered: bootstrap CIs, paired Wilcoxon, Holm, risk differences,
classification metrics and calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

BOOT_N = 10_000
BOOT_SEED = 0


def bootstrap_ci(x: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED, stat=np.mean) -> tuple[float, float, float]:
    """(point, lo, hi) 95 % percentile bootstrap of ``stat`` over rows of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    boots = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def risk_difference(a: np.ndarray, b: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Paired risk difference mean(a) − mean(b) with a paired bootstrap CI."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> float:
    """p-value of the paired Wilcoxon signed-rank test on per-episode outcomes (ties dropped)."""
    from scipy.stats import wilcoxon

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    if np.all(d == 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def prf(y: np.ndarray, s: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(s, dtype=np.float64) > thr
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) and not np.isnan(p) and not np.isnan(r) else float("nan")
    acc = (tp + tn) / max(1, len(y))
    return {"precision": p, "recall": r, "f1": f1, "accuracy": acc, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": int(len(y))}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    ok = ~np.isnan(s)
    if ok.sum() == 0 or y[ok].min() == y[ok].max():
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def ece(y: np.ndarray, s: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    """Expected calibration error with equal-mass bins; also returns the reliability table."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(s) == 0:
        return float("nan"), []
    order = np.argsort(s)
    chunks = np.array_split(order, bins)
    total = 0.0
    table = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        conf = float(s[ch].mean())
        acc = float(y[ch].mean())
        total += len(ch) / len(s) * abs(acc - conf)
        table.append({"confidence": conf, "accuracy": acc, "n": int(len(ch))})
    return float(total), table


def summarize_rates(df: Any, by: list[str], col: str) -> Any:
    """Group and attach bootstrap CIs for a boolean/float column."""
    import pandas as pd

    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        m, lo, hi = bootstrap_ci(g[col].to_numpy(dtype=float))
        rows.append({**dict(zip(by, key)), col: m, f"{col}_lo": lo, f"{col}_hi": hi, "n": len(g)})
    return pd.DataFrame(rows)
