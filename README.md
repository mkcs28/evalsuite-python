# EvalSuite

[![CI](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml/badge.svg)](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![Python](https://img.shields.io/pypi/pyversions/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Unified, reproducible evaluation for machine learning and research.**

EvalSuite brings classification and regression metrics (with clinical, statistical, segmentation and
object-detection evaluation on the roadmap) into one consistent, validated, documented framework.

> **Status: stable (0.1.1).** Every item on the 0.1.0 roadmap is implemented and verified.

## Installation

```bash
pip install evalsuite-python
```

The package is installed as `evalsuite-python` and imported as `evalsuite`:

```python
import evalsuite as es
```

## Why EvalSuite

- **One consistent API.** Every metric returns a result object that behaves like a number and exports to
  JSON, pandas, Markdown and LaTeX.
- **Explicit conventions.** Averaging, label order, the positive class and zero-division behaviour are stated
  and recorded in every result, never silently assumed.
- **Validated.** Each metric is tested against scikit-learn where definitions coincide, plus property-based
  tests and edge cases.
- **Documented.** Every metric carries its definition, formula, range, input requirements and references,
  available programmatically through `metric_info()`.
- **Efficient.** `evaluate()` validates inputs once and computes the confusion matrix once for all metrics.
- **Lightweight.** Requires only NumPy, SciPy and pandas.

## Quick start

```python
import evalsuite as es

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]
y_prob = [0.1, 0.9, 0.4, 0.2, 0.8, 0.6]

result = es.evaluate(y_true, y_pred, y_prob=y_prob)
print(result.summary())

result["f1"]  # MetricResult(f1=0.666667)
f"{result['mcc']:.3f}"  # '0.333'
result.to_latex(caption="Test-set performance")
result.to_dataframe()

es.f1(y_true, y_pred)  # individual metrics
es.roc_auc(y_true, y_prob)
es.metric_info("classification.mcc").formula  # documentation
es.list_metrics("regression")
```

## Comparing models

```python
result = es.compare(
    y_true,
    {"logistic": pred_lr, "forest": pred_rf, "boosting": pred_gb},
    probabilities={"logistic": prob_lr, "forest": prob_rf, "boosting": prob_gb},
    random_state=0,
)
print(result.summary())  # estimates with 95% CIs, paired tests, Holm-adjusted p-values
result.to_latex(label="tab:models")

es.bootstrap_ci("f1", y_true, y_pred, average="macro", random_state=0)  # BCa interval for any metric
es.accuracy_ci(y_true, y_pred)  # Wilson interval
es.delong_test(y_true, prob_a, prob_b)  # two correlated AUCs
es.mcnemar_test(y_true, pred_a, pred_b)
```

Every model is evaluated on the same bootstrap resamples, so differences are paired. Accuracy is compared
with McNemar's test, binary ROC AUC with DeLong's test and other metrics with a paired bootstrap test;
p-values are adjusted for multiple comparisons (Holm by default).

## Classification report

```python
report = es.classification_report(y_true, y_pred)
print(report)  # per-class precision, recall, F1, specificity, support + averages
report.save("report.html")  # also .csv .md .tex .json .txt
```

## Plots

```bash
pip install "evalsuite-python[plot]"   # adds matplotlib; importing evalsuite never loads it
```

```python
es.plot.roc(y_true, {"logistic": prob_lr, "forest": prob_rf})  # AUC in the legend
es.plot.pr(y_true, prob)  # AP and the prevalence line
es.plot.calibration(y_true, prob)  # reliability diagram, ECE, Brier
es.plot.confusion_matrix(y_true, y_pred, normalize="true")
es.plot.residuals(y_reg, pred_reg)  # or kind="predicted"
es.plot.comparison(es.compare(...))  # forest plot with CIs
```

Each function returns a matplotlib `Axes` (pass `ax=` to draw into your own figure). The numbers shown are
computed with EvalSuite's metrics, so plots and tables always agree. Several models get distinct colours
*and* line styles, so figures stay readable in greyscale print.

## Exports

Every result (`evaluate`, `classification_report`, `compare`, single metrics) exports to `summary()`,
`to_json()`, `to_csv()`, `to_dataframe()`, `to_markdown()`, `to_latex()` and `to_html()`, and `save(path)` picks
the format from the extension. HTML pages are standalone (inline CSS, no scripts) and escape all text.

## Command line

```bash
evalsuite evaluate predictions.csv --y-true label --y-pred pred --y-prob prob
evalsuite report predictions.csv --y-true label --y-pred pred -o report.html
evalsuite compare predictions.csv --y-true label --pred lr=pred_lr --pred rf=pred_rf \
    --prob lr=p_lr --prob rf=p_rf --plot comparison.png
evalsuite plot roc predictions.csv --y-true label --y-prob prob -o roc.png
evalsuite metrics --category classification
evalsuite info classification.mcc
evalsuite benchmark --quick
```

Input files can be CSV, TSV, Parquet or JSON. Output format follows `--format` or the `-o` extension
(text, json, csv, markdown, latex, html). Errors are reported in one line with exit code 2.

## Performance

Benchmarked against scikit-learn on the same data (fastest of 5 runs; Python 3.12, NumPy 2.5,
scikit-learn 1.9, Linux x86_64). Every result agrees with scikit-learn to floating-point rounding
(largest difference 1.1e-16).

| Case | n | EvalSuite (ms) | scikit-learn (ms) | Speed-up | Peak memory EvalSuite / sklearn (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 binary label metrics via `evaluate()` | 1,000 | 0.38 | 11.70 | **31.2×** | 0.04 / 0.05 |
| 8 binary label metrics via `evaluate()` | 100,000 | 10.9 | 116.9 | **10.7×** | 3.2 / 3.1 |
| 8 binary label metrics via `evaluate()` | 1,000,000 | 108.8 | 1043.6 | **9.6×** | 31.5 / 30.5 |
| macro F1, 10 classes | 1,000,000 | 88.4 | 139.3 | **1.58×** | 30.5 / 21.8 |
| ROC AUC, binary | 1,000,000 | 247.4 | 352.5 | **1.42×** | 91.6 / 76.3 |
| MAE, MSE, RMSE, R² via `evaluate()` | 1,000 | 0.12 | 0.90 | **7.2×** | 0.03 / 0.02 |
| MAE, MSE, RMSE, R² via `evaluate()` | 1,000,000 | 29.3 | 20.1 | 0.69× | 22.9 / 15.3 |

`evaluate()` validates inputs once and builds the confusion matrix once for all metrics, which is where the
speed-up comes from. Large regression arrays are slower because EvalSuite checks every value for NaN,
infinity, shape and dtype before computing. Reproduce on your machine with `evalsuite benchmark`; full
table and notes in
[BENCHMARKS.md](https://github.com/mkcs28/evalsuite-python/blob/main/BENCHMARKS.md).

## Metrics in this release

**Classification** (binary, multiclass, multilabel; micro/macro/weighted/samples/per-class averaging;
sample weights): accuracy, balanced accuracy, precision, recall, specificity, NPV, F1, F-beta, Jaccard,
MCC, Cohen's kappa (unweighted, linear, quadratic), Hamming loss, confusion matrix, ROC AUC (binary,
one-vs-rest, one-vs-one), average precision, ROC and PR curves, log loss, Brier score, top-k accuracy,
calibration curve and expected calibration error.

**Regression** (single and multi-output; sample weights): MAE, MSE, RMSE, R², adjusted R², MAPE, sMAPE,
MSLE, RMSLE, median absolute error, explained variance, max error, mean bias error, quantile (pinball)
loss, Huber loss, relative absolute error, relative squared error.

## Conventions

- `average="auto"` resolves to `"binary"` for binary targets and `"macro"` otherwise; the resolved value
  is stored in `result.params["average"]`.
- Labels are sorted unless you pass `labels=[...]`; that order defines per-class outputs and the columns
  of 2-D `y_prob`.
- Undefined ratios (zero denominators) return 0 **with an `UndefinedMetricWarning`**; pass
  `zero_division=np.nan` to propagate NaN, or `0`/`1` to choose silently.
- Domain violations raise clear errors instead of being patched over (for example MAPE with zero targets).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=evalsuite
ruff check . && ruff format --check . && mypy
```

## Links

- PyPI: https://pypi.org/project/evalsuite-python/
- Website and documentation: https://evalsuite-nine.vercel.app
- Website source: https://github.com/mkcs28/evalsuite

## Credits

Authors and maintainers: **Manoj Kumar C S** and **Nikhil D Bharadwaj**.

## License

MIT. See [LICENSE](LICENSE).
