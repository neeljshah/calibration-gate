# calibration-gate

Measure binary forecast calibration and evaluate recalibration candidates across
multiple corpora. The gate requires lower Brier error on at least two eligible
evaluation corpora so a single favorable window cannot decide the result.

The library includes Brier score and decomposition, expected calibration error
(ECE), reliability curves, isotonic regression and Platt scaling. Runtime
dependencies are limited to NumPy.

## Install

Requires Python 3.9 or later.

```bash
git clone https://github.com/neeljshah/calibration-gate.git
cd calibration-gate
python -m pip install -e ".[dev]"
```

The `dev` extra installs pytest. Install `-e .` for the library alone.

## Runnable example

```bash
python examples/demo_gate.py
```

This seeded synthetic example trains an isotonic recalibrator separately from
three evaluation corpora, then shows how an in-sample improvement can fail to
transfer. Its output is:

```text
Scenario 1 raw forecaster (overconfident) on the 3 OOS corpora:
  corpus_1: brier=0.20100 ece=0.1050
  corpus_2: brier=0.19816 ece=0.0986
  corpus_3: brier=0.19162 ece=0.0912

--- Scenario 1: isotonic fit on held-out train window ---
  corpus_1: raw=0.20100 cand=0.19038 delta=-0.01061 (CAND wins)
  corpus_2: raw=0.19816 cand=0.18816 delta=-0.01000 (CAND wins)
  corpus_3: raw=0.19162 cand=0.18328 delta=-0.00834 (CAND wins)
  VERDICT: SHIP (method=isotonic, wins=3)

Scenario 2: on its own window the lucky recalibrator 'improves' brier 0.18471 -> 0.17612 (noise fit)

--- Scenario 2: lucky single-window recalibrator, judged on 3 OOS corpora ---
  corpus_1: raw=0.18152 cand=0.18408 delta=+0.00257 (RAW wins)
  corpus_2: raw=0.18538 cand=0.18752 delta=+0.00214 (RAW wins)
  corpus_3: raw=0.17823 cand=0.18087 delta=+0.00264 (RAW wins)
  VERDICT: REJECT (method=raw, wins=0)
```

These numbers describe synthetic forecast calibration, not measured market
performance.

## API and interpretation

| Function | Result |
| --- | --- |
| `brier_score(p, y)` | Mean squared forecast error; lower is better. |
| `brier_decomposition(p, y, bins=10)` | Reliability, resolution, uncertainty and the bin-mean Brier score. |
| `ece(p, y, bins=10, strategy="uniform")` | Bin-weighted absolute difference between forecast and observed frequency. |
| `reliability_curve(p, y, bins=10, strategy="uniform")` | Nonempty bins' mean forecasts, observed frequencies and counts. |
| `fit_isotonic(p, y)` | A monotone step function fitted with weighted PAVA; equal forecasts are pooled first. |
| `fit_platt(p, y)` | A two-parameter logistic recalibrator fitted by gradient descent. |

Metrics expect equal-length, nonempty one-dimensional arrays, finite forecast
probabilities in `[0, 1]` and binary outcomes. Uniform and quantile binning are
supported for ECE and reliability curves. Quantile boundaries collapse when
forecasts tie, so actual bins may be fewer than requested; constant forecasts
occupy one bin. Always report the bin count and strategy with ECE.

The decomposition identity applies to the bin-mean forecasts; it need not equal
the Brier score of the original individual forecasts. Both values are returned.

`accept(candidate, corpora, min_n=100, min_corpora=2)` returns:

- `SHIP` when the candidate lowers Brier error on enough eligible corpora.
- `REJECT` when enough corpora are eligible but too few improve.
- `UNDERPOWERED` when too few corpora meet `min_n`.

`accept_best` prefers Platt over isotonic among candidates that pass, then uses
mean candidate Brier error to break equal-rank ties.

The caller is responsible for separating training from evaluation and supplying
independent corpora. The gate does not verify independence, test statistical
significance or require improvement on every corpus. Its verdict is a
recalibration decision under these rules, not a claim of market outperformance.

## Tests

```bash
python -m pytest tests/test_metrics.py -q
python -m pytest tests/test_gate.py -q
```

The test files cover metric identities, input validation, constant quantile
forecasts, tied isotonic inputs, monotonicity, recalibration, thin corpora and all
three gate verdicts. GitHub Actions runs each file separately.

## Context and license

[The CourtVision evidence site](https://neeljshah.github.io/court-vision/)
contains the forecasting results and public retraction record. Its moneyline
models land near the devigged close without beating it; totals and tennis trail.

[MIT](LICENSE), copyright Neel Shah 2026.
