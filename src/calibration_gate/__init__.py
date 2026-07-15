"""calibration_gate: calibration metrics + a multi-corpus acceptance gate."""
from .metrics import brier_score, brier_decomposition, ece, reliability_curve
from .gate import Corpus, GateVerdict, accept, accept_best, fit_isotonic, fit_platt

__all__ = [
    "brier_score", "brier_decomposition", "ece", "reliability_curve",
    "Corpus", "GateVerdict", "accept", "accept_best", "fit_isotonic", "fit_platt",
]
