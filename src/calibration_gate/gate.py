"""Recalibrators (isotonic via PAVA, Platt via gradient steps) and the
multi-corpus acceptance gate. numpy + stdlib only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .metrics import brier_score

Recalibrator = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Recalibrators
# ---------------------------------------------------------------------------

def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Recalibrator:
    """Isotonic regression of y on p via the Pool Adjacent Violators Algorithm.

    Returns a monotone step function mapping raw probability -> recalibrated
    probability (each pooled block's mean outcome, applied from its first raw p).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]

    # PAVA: maintain blocks of (weighted mean, weight); merge while decreasing.
    means: List[float] = []
    weights: List[float] = []
    starts: List[int] = []  # start index of each block in sorted order
    for i, v in enumerate(ys):
        means.append(float(v))
        weights.append(1.0)
        starts.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            w = weights[-2] + weights[-1]
            m = (means[-2] * weights[-2] + means[-1] * weights[-1]) / w
            means[-2:] = [m]
            weights[-2:] = [w]
            starts.pop()

    # Step function: block k applies from its first raw p onward.
    thresholds = ps[np.array(starts)]
    vals = np.clip(np.array(means), 0.0, 1.0)

    def transform(q: np.ndarray) -> np.ndarray:
        i = np.searchsorted(thresholds, np.asarray(q, dtype=float), side="right") - 1
        return vals[np.clip(i, 0, len(vals) - 1)]

    return transform


def fit_platt(p: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 2000) -> Recalibrator:
    """Platt scaling: logistic regression sigmoid(a*logit(p)+b) by gradient descent.

    # ponytail: plain full-batch GD, fine for the corpus sizes a gate sees;
    # swap in Newton or sklearn LogisticRegression if fitting ever gets slow.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    x = np.log(p / (1 - p))
    a, b = 1.0, 0.0
    n = len(x)
    for _ in range(steps):
        z = 1.0 / (1.0 + np.exp(-(a * x + b)))
        g = z - y
        a -= lr * float(g @ x) / n
        b -= lr * float(g.sum()) / n

    def transform(q: np.ndarray) -> np.ndarray:
        qc = np.clip(np.asarray(q, dtype=float), 1e-6, 1 - 1e-6)
        xl = np.log(qc / (1 - qc))
        return 1.0 / (1.0 + np.exp(-(a * xl + b)))

    return transform


# Intervention rank for the least-intervention tie-break: identity < platt < isotonic.
# Platt bends the forecast with 2 parameters; isotonic can reshape it arbitrarily.
INTERVENTION_RANK: Dict[str, int] = {"raw": 0, "platt": 1, "isotonic": 2}


# ---------------------------------------------------------------------------
# Acceptance gate
# ---------------------------------------------------------------------------

@dataclass
class Corpus:
    """One independent out-of-sample evaluation set."""
    name: str
    p_raw: np.ndarray  # raw forecast probabilities
    y: np.ndarray      # binary outcomes


@dataclass
class GateVerdict:
    decision: str                      # "SHIP" | "REJECT" | "UNDERPOWERED"
    method: str                        # winning method ("raw" if REJECT)
    wins: int                          # corpora where candidate beat raw
    evidence: List[str] = field(default_factory=list)  # per-corpus lines
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)  # corpus -> {raw, candidate}


def accept(
    candidate: Recalibrator,
    corpora: Sequence[Corpus],
    method: str = "candidate",
    min_n: int = 100,
    min_corpora: int = 2,
) -> GateVerdict:
    """SHIP only if the candidate beats raw Brier on >= min_corpora independent
    corpora, each with n >= min_n. Thin corpora never count toward the gate.

    UNDERPOWERED when fewer than min_corpora corpora are big enough to judge.
    """
    evidence: List[str] = []
    scores: Dict[str, Dict[str, float]] = {}
    wins = 0
    eligible = 0
    for c in corpora:
        n = len(c.y)
        if n < min_n:
            evidence.append(f"{c.name}: n={n} < {min_n} (thin, not counted)")
            continue
        eligible += 1
        b_raw = brier_score(c.p_raw, c.y)
        b_cand = brier_score(np.clip(candidate(c.p_raw), 0.0, 1.0), c.y)
        scores[c.name] = {"raw": b_raw, "candidate": b_cand}
        beat = b_cand < b_raw
        wins += beat
        evidence.append(
            f"{c.name}: raw={b_raw:.5f} cand={b_cand:.5f} "
            f"delta={b_cand - b_raw:+.5f} ({'CAND wins' if beat else 'RAW wins'})"
        )

    if eligible < min_corpora:
        return GateVerdict("UNDERPOWERED", "raw", wins, evidence, scores)
    if wins >= min_corpora:
        return GateVerdict("SHIP", method, wins, evidence, scores)
    return GateVerdict("REJECT", "raw", wins, evidence, scores)


def accept_best(
    candidates: Dict[str, Recalibrator],
    corpora: Sequence[Corpus],
    min_n: int = 100,
    min_corpora: int = 2,
) -> GateVerdict:
    """Gate several candidates; among those that SHIP, pick the LEAST intervention
    (per INTERVENTION_RANK), breaking remaining ties by mean Brier across corpora.
    """
    verdicts: List[Tuple[str, GateVerdict]] = [
        (name, accept(fn, corpora, method=name, min_n=min_n, min_corpora=min_corpora))
        for name, fn in candidates.items()
    ]
    shipped = [(name, v) for name, v in verdicts if v.decision == "SHIP"]
    if shipped:
        def key(item: Tuple[str, GateVerdict]) -> Tuple[int, float]:
            name, v = item
            mean_b = float(np.mean([s["candidate"] for s in v.scores.values()]))
            return (INTERVENTION_RANK.get(name, 99), mean_b)
        return min(shipped, key=key)[1]
    # No candidate shipped: return the first verdict (UNDERPOWERED beats REJECT for honesty)
    for _, v in verdicts:
        if v.decision == "UNDERPOWERED":
            return v
    return verdicts[0][1] if verdicts else GateVerdict("UNDERPOWERED", "raw", 0)
