"""Aggregation and gate inputs — arithmetic lives here, never in a question."""

from __future__ import annotations

import math

import numpy as np


def penalty(family: str, p: list[float], smooth_temperature: float | None = None) -> float:
    """Fold per-step probabilities (or soft level probabilities) into one penalty in [0, 1].

    ``never``: ``max_t p``; ``always``: ``1 - min_t p``; ``soft``: ``1 - E[level] / 4``.
    Missing answers (NaN) are read as 0, i.e. "not present": that under-penalises ``never`` and
    over-penalises ``always``. ``JevCost`` drops unknown steps before calling this, so the planner
    never relies on the convention. The judge-only study scores a failed call with it: 0 for
    ``never``, 1 for ``always`` and 0.5 for ``soft``.
    """
    if not p:
        return 0.0
    arr = np.asarray(p, dtype=np.float64)
    if np.any(np.isnan(arr)):
        arr = np.nan_to_num(arr, nan=0.0)
    if family == "never":
        if smooth_temperature:
            w = np.exp((arr - arr.max()) / smooth_temperature)
            return float(np.sum(w * arr) / np.sum(w))
        return float(arr.max())
    if family == "always":
        return float(1.0 - arr.min())
    if family == "soft":
        levels = np.arange(len(arr))
        s = arr.sum()
        dist = arr / s if s > 0 else np.ones_like(arr) / len(arr)
        expected = float((levels * dist).sum())
        return float(1.0 - expected / (len(arr) - 1))
    if family == "temporal_before":
        # p = [pA_1..pA_H, pB_1..pB_H]
        h = len(arr) // 2
        a, b = arr[:h], arr[h:]
        fa, fb = first_true(a), first_true(b)
        return 0.0 if fa < fb else 1.0
    raise ValueError(family)


def first_true(p: np.ndarray, threshold: float = 0.5) -> int:
    idx = np.nonzero(p > threshold)[0]
    return int(idx[0]) if len(idx) else 10**9


def uncertainty_proxy(
    family: str, p: list[float], confidence: float | None, band: tuple[float, float] = (0.3, 0.7)
) -> float:
    """Fraction of step probabilities inside the uncertain band (never/always) or 1 − confidence (soft)."""
    if family == "soft":
        return (
            1.0 - float(confidence)
            if confidence is not None and not math.isnan(confidence)
            else 1.0
        )
    if not p:
        return 1.0
    arr = np.asarray(p, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return 1.0
    arr = arr[~np.isnan(arr)]
    return float(np.mean((arr >= band[0]) & (arr <= band[1])))
