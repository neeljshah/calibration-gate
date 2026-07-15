"""Calibration metrics for binary probability forecasts. numpy only."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate(p: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError(f"p and y must be 1-D arrays of equal length, got {p.shape} vs {y.shape}")
    if len(p) == 0:
        raise ValueError("empty forecast arrays")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must be 0 or 1")
    return p, y


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between forecast probability and binary outcome."""
    p, y = _validate(p, y)
    return float(np.mean((p - y) ** 2))


def brier_decomposition(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Murphy (1973) decomposition: brier = reliability - resolution + uncertainty.

    Computed on forecasts grouped into uniform bins; returns the binned Brier
    too, since the identity holds exactly only for the binned forecasts.
    """
    p, y = _validate(p, y)
    n = len(p)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    base_rate = float(y.mean())
    uncertainty = base_rate * (1.0 - base_rate)
    reliability = 0.0
    resolution = 0.0
    for k in range(bins):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = float(p[mask].mean())
        ok = float(y[mask].mean())
        reliability += (nk / n) * (pk - ok) ** 2
        resolution += (nk / n) * (ok - base_rate) ** 2
    return {
        "brier": brier_score(p, y),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        # identity holds for bin-mean forecasts, not raw p; report it explicitly
        "binned_brier": reliability - resolution + uncertainty,
    }


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10, strategy: str = "uniform") -> float:
    """Expected Calibration Error: bin-weighted |mean outcome - mean forecast|.

    strategy="uniform": equal-width bins on [0,1]; strategy="quantile": equal-count
    bins (robust when forecasts cluster, e.g. all near 0.5).
    """
    p, y = _validate(p, y)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(edges)  # collapse duplicate edges from ties
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    n = len(p)
    total = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        nb = int(mask.sum())
        if nb:
            total += (nb / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def reliability_curve(
    p: np.ndarray, y: np.ndarray, bins: int = 10, strategy: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean forecast, observed frequency, count) for a reliability plot.

    Empty bins are dropped. Perfect calibration lies on the diagonal.
    """
    p, y = _validate(p, y)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    mean_pred, obs_freq, counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == len(edges) - 2 else (p >= lo) & (p < hi)
        nb = int(mask.sum())
        if nb:
            mean_pred.append(float(p[mask].mean()))
            obs_freq.append(float(y[mask].mean()))
            counts.append(nb)
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)
