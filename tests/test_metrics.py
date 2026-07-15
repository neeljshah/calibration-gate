import numpy as np
import pytest

from calibration_gate.metrics import brier_score, brier_decomposition, ece, reliability_curve


def test_brier_hand_computed() -> None:
    p = np.array([0.8, 0.3, 0.5, 1.0])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    # (0.04 + 0.09 + 0.25 + 0.0) / 4
    assert brier_score(p, y) == pytest.approx(0.095)


def test_brier_perfect_and_worst() -> None:
    y = np.array([1.0, 0.0, 1.0])
    assert brier_score(y.copy(), y) == 0.0
    assert brier_score(1.0 - y, y) == 1.0


def test_brier_validation() -> None:
    with pytest.raises(ValueError):
        brier_score(np.array([1.2]), np.array([1.0]))
    with pytest.raises(ValueError):
        brier_score(np.array([0.5]), np.array([0.7]))
    with pytest.raises(ValueError):
        brier_score(np.array([]), np.array([]))


def test_decomposition_identity_on_binned_forecasts() -> None:
    # When forecasts equal their bin means, brier == rel - res + unc exactly.
    rng = np.random.default_rng(0)
    p = rng.choice([0.05, 0.25, 0.45, 0.65, 0.85], size=5000)
    y = (rng.uniform(size=5000) < p * 0.8).astype(float)  # miscalibrated
    d = brier_decomposition(p, y, bins=10)
    assert d["brier"] == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-12
    )
    assert d["reliability"] > 0.0
    assert d["uncertainty"] == pytest.approx(y.mean() * (1 - y.mean()))


def test_ece_near_zero_when_perfectly_calibrated() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 50000)
    y = (rng.uniform(size=50000) < p).astype(float)
    assert ece(p, y, bins=10) < 0.01
    assert ece(p, y, bins=10, strategy="quantile") < 0.01


def test_ece_detects_miscalibration() -> None:
    p = np.full(1000, 0.9)
    y = np.zeros(1000)
    y[:500] = 1.0  # observed 50% vs forecast 90%
    assert ece(p, y) == pytest.approx(0.4)


def test_ece_bad_strategy() -> None:
    with pytest.raises(ValueError):
        ece(np.array([0.5]), np.array([1.0]), strategy="magic")


def test_reliability_curve_shapes_and_counts() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(size=1000)
    y = (rng.uniform(size=1000) < p).astype(float)
    mp, of, n = reliability_curve(p, y, bins=10)
    assert len(mp) == len(of) == len(n)
    assert n.sum() == 1000
    assert np.all((mp >= 0) & (mp <= 1)) and np.all((of >= 0) & (of <= 1))
    # bin means must be increasing for uniform bins on uniform forecasts
    assert np.all(np.diff(mp) > 0)
