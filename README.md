# calibration-gate

Calibration metrics for binary probability forecasts, plus a **multi-corpus
acceptance gate**: a recalibration ships only if it beats the raw forecaster
on at least two independent out-of-sample corpora.

Distilled from a multi-sport forecasting system, where this exact gate
repeatedly rejected recalibrations that looked like clear wins on a single
evaluation window and evaporated everywhere else.

That system is CourtVision, a four-sport forecasting system audited in public; the measured results and the retraction record are in its evidence index: https://github.com/neeljshah/court-vision/blob/master/EVIDENCE.md

**Scope note:** calibration is a statement about forecast quality, not about
any market advantage. Nothing here claims or implies an edge over anyone.

## Calibration is not accuracy

Accuracy asks "how often is the forecast on the right side?" Calibration asks
"when you say 60%, does it happen 60% of the time?" They are independent:

- A forecaster that says **60%** for every game of a league where home teams
  win 60% of the time is **perfectly calibrated** and only 60% accurate. It
  has zero resolution -- it never distinguishes one game from another.
- A forecaster that says **99%** whenever it is right and **51%** whenever it
  is wrong can be highly accurate and **badly miscalibrated** -- its stated
  probabilities are lies about its own uncertainty.

If probabilities feed anything downstream (expected values, decisions,
aggregation), calibration is the property you need.

## Metrics (`calibration_gate.metrics`)

- **`brier_score(p, y)`** -- mean squared error between forecast and outcome.
  Proper: minimized by reporting your true belief.
- **`brier_decomposition(p, y, bins=10)`** -- Murphy's decomposition
  `brier = reliability - resolution + uncertainty`.
  *Reliability* (lower is better) is miscalibration; *resolution* (higher is
  better) is how much the forecasts separate outcomes; *uncertainty* is the
  base-rate variance you cannot control. The identity holds exactly for
  bin-mean forecasts; the returned `binned_brier` makes that explicit.
- **`ece(p, y, bins=10, strategy="uniform"|"quantile")`** -- Expected
  Calibration Error: bin-weighted average of |observed frequency - mean
  forecast|. **Binning caveat:** ECE depends on the binning. Uniform bins can
  hide miscalibration in sparse regions; quantile bins fix bin counts but move
  the edges with the forecast distribution. Report the bin count and strategy
  with any ECE, and treat small differences across binnings as noise.
- **`reliability_curve(p, y, bins=10)`** -- per-bin (mean forecast, observed
  frequency, count) for plotting against the diagonal.

## Why single-window calibration "wins" are usually overfit

A recalibrator fit and evaluated on (or near) one window inherits that
window's noise. With a few hundred samples, an isotonic fit will happily bend
itself around random clumps of outcomes and show a flattering in-sample Brier
drop. That improvement is a property of the window, not the forecaster. The
failure mode is quiet: nothing looks wrong until the "improved" model is worse
in production. Independent corpora -- different seasons, different data
sources, different populations -- are the cheapest instrument that detects it:
noise does not replicate; real miscalibration does.

## The acceptance gate (`calibration_gate.gate`)

```
accept(candidate, corpora, min_n=100, min_corpora=2) -> GateVerdict
```

- **SHIP** -- the candidate beats raw Brier on `>= min_corpora` independent
  OOS corpora, each with `n >= min_n`.
- **REJECT** -- enough evidence existed and the candidate did not clear the
  bar. A reject is a *successful* measurement, not a failure.
- **UNDERPOWERED** -- fewer than `min_corpora` corpora were big enough to
  judge. Thin corpora never count toward the gate in either direction.

`accept_best(candidates, corpora)` gates several recalibrators at once and
applies the **least-intervention tie-break**: among candidates that ship,
prefer the one that distorts the forecast least (raw < Platt < isotonic),
then lower mean Brier. The prior is that smaller interventions generalize
better; you should need multi-corpus evidence to justify a bigger one.

Two recalibrators are included, both from scratch on numpy:

- **`fit_platt(p, y)`** -- Platt scaling, a 2-parameter logistic map on the
  logit, fit by gradient descent.
- **`fit_isotonic(p, y)`** -- isotonic regression via the Pool Adjacent
  Violators Algorithm; a monotone step function.

## Quick start

```bash
pip install -e ".[dev]"
python examples/demo_gate.py
python -m pytest tests/test_metrics.py -q
python -m pytest tests/test_gate.py -q
```

The demo builds synthetic corpora from a known-overconfident forecaster and
shows both gate outcomes: an honest isotonic recalibration **SHIPPING**
(the miscalibration replicates on all corpora), then a lucky single-window
recalibration getting **REJECTED** when the other corpora disagree.

## Author

Neel Shah -- neeljshah22@gmail.com -- [github.com/neeljshah](https://github.com/neeljshah)

MIT License.
