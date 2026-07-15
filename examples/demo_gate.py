"""Demo: three independent OOS corpora and the acceptance gate.

Scenario 1: the forecaster is systematically OVERCONFIDENT. Isotonic
recalibration fit on a held-out train window SHIPS -- it beats raw on 2+
corpora because the miscalibration is real and shows up everywhere.

Scenario 2: the forecaster is already well calibrated. A recalibrator fit on
one small window 'improves' that window (it fit the noise), but the gate
REJECTS it -- the other corpora disagree. This single-window overfit is
exactly what the gate exists to catch.
"""
from __future__ import annotations

import numpy as np

from calibration_gate import Corpus, accept, brier_score, ece, fit_isotonic

rng = np.random.default_rng(7)


def make_corpus(name: str, n: int, sharpen: float = 1.0) -> Corpus:
    """True probability t ~ U(0.05, 0.95). sharpen>1 makes the forecaster
    overconfident: it reports sigmoid(sharpen * logit(t)) instead of t."""
    t = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < t).astype(float)
    logit = np.log(t / (1 - t))
    p = 1.0 / (1.0 + np.exp(-sharpen * logit))
    return Corpus(name, p, y)


def show(title: str, verdict) -> None:
    print(f"\n--- {title} ---")
    for line in verdict.evidence:
        print(f"  {line}")
    print(f"  VERDICT: {verdict.decision} (method={verdict.method}, wins={verdict.wins})")


def main() -> None:
    # Scenario 1: overconfident forecaster, honest recalibration.
    train = make_corpus("train_window", 4000, sharpen=1.8)
    corpora = [make_corpus(f"corpus_{i}", 2000, sharpen=1.8) for i in (1, 2, 3)]
    print("Scenario 1 raw forecaster (overconfident) on the 3 OOS corpora:")
    for c in corpora:
        print(f"  {c.name}: brier={brier_score(c.p_raw, c.y):.5f} ece={ece(c.p_raw, c.y):.4f}")
    iso = fit_isotonic(train.p_raw, train.y)
    show("Scenario 1: isotonic fit on held-out train window", accept(iso, corpora, method="isotonic"))

    # Scenario 2: forecaster is ALREADY calibrated; recalibrator fit on one
    # small lucky window fits noise, looks like a win there, fails elsewhere.
    lucky_window = make_corpus("lucky_window", 300, sharpen=1.0)
    lucky = fit_isotonic(lucky_window.p_raw, lucky_window.y)
    b_raw = brier_score(lucky_window.p_raw, lucky_window.y)
    b_in = brier_score(lucky(lucky_window.p_raw), lucky_window.y)
    print(f"\nScenario 2: on its own window the lucky recalibrator 'improves' "
          f"brier {b_raw:.5f} -> {b_in:.5f} (noise fit)")
    oos = [make_corpus(f"corpus_{i}", 2000, sharpen=1.0) for i in (1, 2, 3)]
    show("Scenario 2: lucky single-window recalibrator, judged on 3 OOS corpora",
         accept(lucky, oos, method="lucky_isotonic"))


if __name__ == "__main__":
    main()
