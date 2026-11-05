import numpy as np
import pytest

from calibration_gate.gate import (
    Corpus, accept, accept_best, fit_isotonic, fit_platt,
)
from calibration_gate.metrics import brier_score


def test_pava_known_small_example() -> None:
    # Classic PAVA: y = [1,0,1] on increasing p -> middle violation pools to 0.5.
    p = np.array([0.1, 0.5, 0.9])
    y = np.array([1.0, 0.0, 1.0])
    f = fit_isotonic(p, y)
    # First two points pool to mean 0.5; third stays 1.0.
    out = f(p)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_pava_monotone() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    y = (rng.uniform(size=500) < p).astype(float)
    f = fit_isotonic(p, y)
    grid = np.linspace(0, 1, 200)
    assert np.all(np.diff(f(grid)) >= -1e-12)


def test_pava_pools_ties_independently_of_outcome_order() -> None:
    p = np.array([0.5, 0.5])
    for y in ([0.0, 1.0], [1.0, 0.0]):
        assert fit_isotonic(p, y)(p) == pytest.approx([0.5, 0.5])


def test_pava_weights_tied_groups_when_pooling() -> None:
    p = np.array([0.2, 0.5, 0.5, 0.5, 0.9])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert fit_isotonic(p, y)(p) == pytest.approx([0.5, 0.5, 0.5, 0.5, 1.0])


def test_platt_fixes_overconfidence() -> None:
    rng = np.random.default_rng(1)
    t = rng.uniform(0.1, 0.9, 5000)
    y = (rng.uniform(size=5000) < t).astype(float)
    logit = np.log(t / (1 - t))
    p_over = 1.0 / (1.0 + np.exp(-2.0 * logit))  # overconfident
    f = fit_platt(p_over, y)
    assert brier_score(f(p_over), y) < brier_score(p_over, y)


def _corpus(name: str, n: int, sharpen: float, seed: int) -> Corpus:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < t).astype(float)
    logit = np.log(t / (1 - t))
    p = 1.0 / (1.0 + np.exp(-sharpen * logit))
    return Corpus(name, p, y)


def test_gate_ship_path() -> None:
    train = _corpus("train", 4000, sharpen=1.8, seed=10)
    iso = fit_isotonic(train.p_raw, train.y)
    corpora = [_corpus(f"c{i}", 2000, sharpen=1.8, seed=20 + i) for i in range(3)]
    v = accept(iso, corpora, method="isotonic")
    assert v.decision == "SHIP"
    assert v.wins >= 2
    assert v.method == "isotonic"


def test_gate_reject_path() -> None:
    # Raw already calibrated; identity-worsening candidate loses everywhere.
    corpora = [_corpus(f"c{i}", 2000, sharpen=1.0, seed=30 + i) for i in range(3)]
    worse = lambda q: np.clip(q + 0.15, 0, 1)  # noqa: E731
    v = accept(worse, corpora)
    assert v.decision == "REJECT"
    assert v.method == "raw"


def test_gate_underpowered_path() -> None:
    corpora = [_corpus(f"c{i}", 50, sharpen=1.8, seed=40 + i) for i in range(3)]
    v = accept(lambda q: q, corpora, min_n=100)
    assert v.decision == "UNDERPOWERED"
    assert all("thin" in e for e in v.evidence)


def test_gate_thin_corpora_do_not_count_toward_wins() -> None:
    train = _corpus("train", 4000, sharpen=1.8, seed=50)
    iso = fit_isotonic(train.p_raw, train.y)
    # 1 big corpus + 2 thin ones: only 1 eligible -> UNDERPOWERED, not SHIP.
    corpora = [
        _corpus("big", 2000, sharpen=1.8, seed=51),
        _corpus("thin1", 20, sharpen=1.8, seed=52),
        _corpus("thin2", 20, sharpen=1.8, seed=53),
    ]
    v = accept(iso, corpora, min_n=100, min_corpora=2)
    assert v.decision == "UNDERPOWERED"


def test_least_intervention_tie_break() -> None:
    # Overconfidence is a logit-linear distortion: BOTH platt and isotonic ship.
    # accept_best must pick platt (lower intervention rank).
    train = _corpus("train", 6000, sharpen=1.8, seed=60)
    corpora = [_corpus(f"c{i}", 3000, sharpen=1.8, seed=70 + i) for i in range(3)]
    v = accept_best(
        {
            "isotonic": fit_isotonic(train.p_raw, train.y),
            "platt": fit_platt(train.p_raw, train.y),
        },
        corpora,
    )
    assert v.decision == "SHIP"
    assert v.method == "platt"
