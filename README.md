# EvalSuite

[![CI](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml/badge.svg)](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![Python](https://img.shields.io/pypi/pyversions/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Unified, reproducible evaluation for machine learning and research.**

EvalSuite brings classification and regression metrics (with clinical, statistical, segmentation and
object-detection evaluation on the roadmap) into one consistent, validated, documented framework.

> **Status: in development (v0.1.0 in progress).** The API may change before 0.1.0.

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

result["f1"]                 # MetricResult(f1=0.666667)
f"{result['mcc']:.3f}"       # '0.333'
result.to_latex(caption="Test-set performance")
result.to_dataframe()

es.f1(y_true, y_pred)                                  # individual metrics
es.roc_auc(y_true, y_prob)
es.metric_info("classification.mcc").formula           # documentation
es.list_metrics("regression")
```

## Metrics in this release

**Classification** (binary, multiclass, multilabel; micro/macro/weighted/samples/per-class averaging;
sample weights): accuracy, balanced accuracy, precision, recall, specificity, NPV, F1, F-beta, Jaccard,
MCC, Cohen's kappa (unweighted, linear, quadratic), Hamming loss, confusion matrix, ROC AUC (binary,
one-vs-rest, one-vs-one), average precision, ROC and PR curves, log loss, Brier score, top-k accuracy.

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

## License

MIT. See [LICENSE](LICENSE).
