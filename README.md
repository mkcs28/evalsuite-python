# EvalSuite

[![CI](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml/badge.svg)](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![Python](https://img.shields.io/pypi/pyversions/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Unified, reproducible evaluation for machine learning and research.**

EvalSuite brings classification, regression, clinical, statistical, segmentation and object-detection
evaluation into one consistent, validated, documented framework.

> **Status: stable (0.3.0).** Every item on the 0.1.0, 0.2.0 and 0.3.0 roadmaps is implemented and verified.

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

## Clinical evaluation

```python
report = es.diagnostic_report(y_true, y_pred)  # binary test vs reference standard
print(report)
# Sensitivity, specificity, PPV, NPV (Wilson CIs), LR+ and LR− (log CIs, Simel 1991),
# diagnostic odds ratio (Woolf), Youden's J, accuracy and prevalence

es.lr_positive(y_true, y_pred)
es.youden_j(y_true, y_pred)

dca = es.decision_curve(y_true, {"model": y_prob})  # net benefit vs treat all / treat none
dca.useful_range()  # thresholds where the model beats both
es.plot.decision_curve(y_true, {"model": y_prob})
```

Ratios that divide by zero are `inf` or NaN with a warning, never 0.

## Calibration

```python
es.calibration_report(y_true, y_prob)  # Brier, ECE, MCE, intercept, slope, Hosmer–Lemeshow
es.calibration_slope(y_true, y_prob)  # ideal 1; < 1 means predictions are too extreme
es.calibration_intercept(y_true, y_prob)  # ideal 0 (calibration-in-the-large)
es.hosmer_lemeshow(y_true, y_prob, n_groups=10)
```

## Statistical tests

```python
es.t_test(scores_a, scores_b)  # Welch by default; mean difference with CI and Cohen's d
es.paired_t_test(fold_scores_a, fold_scores_b)
es.wilcoxon_test(fold_scores_a, fold_scores_b)  # with matched-pairs rank-biserial r
es.mann_whitney_test(a, b)  # with rank-biserial r
es.friedman_test(scores_a, scores_b, scores_c)  # with Kendall's W
es.kruskal_wallis_test(g1, g2, g3)
es.shapiro_wilk_test(residuals)
es.chi_square_test(table)  # with Cramér's V
es.fisher_exact_test([[8, 2], [1, 5]])
es.adjust_pvalues(p_values, method="hochberg")  # also holm, bonferroni, bh, by
```

Every test returns a `TestResult` with the statistic, p-value and an effect size, computed with SciPy and
checked against SciPy and statsmodels in the test suite.

## Segmentation

```python
# label masks: one 2-D image, or images on the first axis (N, H, W) / (N, D, H, W), or a list of masks
es.dice(y_true, y_pred)  # macro over classes, pixel counts summed over the dataset
es.iou(y_true, y_pred, average=None)  # per class; classes absent from both masks are NaN, not 0
es.miou(y_true, y_pred, ignore_index=255)
es.dice(y_true, y_pred, aggregate="image")  # mean of per-image scores (medical imaging convention)
es.boundary_iou(y_true, y_pred)  # Cheng et al. 2021
es.hausdorff_distance(y_true, y_pred, percentile=95, spacing=(0.8, 0.8))  # HD95 in mm

report = es.segmentation_report(y_true, y_pred, class_names={0: "background", 1: "liver"})
print(report)  # mIoU, Dice, pixel accuracy, Boundary IoU, HD95, ASSD + per-class table
es.plot.segmentation(image, y_true[0], y_pred[0])  # prediction fill, truth outline
es.plot.per_class(report, metric="iou")  # per-class bars (also takes a detection report)
```

## Object detection

```python
y_true = [{"boxes": [[x1, y1, x2, y2], ...], "labels": [3, ...]}, ...]  # one dict per image
y_pred = [{"boxes": [...], "labels": [...], "scores": [...]}, ...]

report = es.detection_report(y_true, y_pred)  # the 12 COCO numbers + AP per class
report["map"], report["map_50"], report["mar_100"]
es.mean_average_precision(y_true, y_pred, iou_threshold=0.5)  # mAP@.50
es.average_precision_detection(y_true, y_pred, interpolation="voc")  # per class, VOC-style
es.box_iou(boxes_a, boxes_b, box_format="xywh")
y_true, y_pred = es.from_coco("instances_val.json", "detections.json")
es.plot.detection_pr(y_true, y_pred)
```

The COCO protocol (crowd regions, area ranges, max detections, 101-point interpolation) matches
`pycocotools` to the last digit in the test suite.

Models can be compared over the same images with intervals and paired tests, as for every other task:
`es.compare(y_true, {"unet": masks_a, "deeplab": masks_b})` resamples images; for detection it compares
mAP.

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
evalsuite diagnostic predictions.csv --y-true label --y-pred pred   # sensitivity, LR+, DOR... with CIs
evalsuite calibration predictions.csv --y-true label --y-prob prob  # slope, intercept, ECE, HL
evalsuite plot decision predictions.csv --y-true label --y-prob prob -o dca.png
evalsuite segmentation true_masks.npy pred_masks.npy --ignore-index 255 --plot per_class.png
evalsuite detection instances_val.json detections.json --plot pr_curves.png
evalsuite metrics --category clinical
evalsuite info classification.mcc
evalsuite benchmark --quick
```

Input files can be CSV, TSV, Parquet or JSON. Output format follows `--format` or the `-o` extension
(text, json, csv, markdown, latex, html). Errors are reported in one line with exit code 2.

## Performance

Benchmarked against reference implementations on the same data (fastest of 5 runs; Python 3.12, NumPy 2.5,
Linux x86_64). Every result agrees with the reference to floating-point rounding (largest difference
1.4e-14).

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up |
| --- | ---: | --- | ---: | ---: | ---: |
| 8 binary label metrics via `evaluate()` | 1,000,000 | scikit-learn | 30.2 | 1020.2 | **33.8×** |
| macro F1, 10 classes | 1,000,000 | scikit-learn | 22.4 | 128.7 | **5.8×** |
| ROC AUC, binary | 1,000,000 | scikit-learn | 173.2 | 300.6 | **1.7×** |
| MAE, MSE, RMSE, R² via `evaluate()` | 1,000,000 | scikit-learn | 19.1 | 9.7 | 0.51× |
| sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 66.2 | 392.8 | **5.9×** |
| calibration slope and intercept | 1,000,000 | statsmodels | 178.0 | 1014.8 | **5.7×** |
| decision curve, 99 thresholds | 1,000,000 | NumPy loop | 155.2 | 174.3 | **1.1×** |
| diagnostic report (7 CIs) | 1,000,000 | statsmodels | 16.6 | 4.6 | 0.28× |
| Welch t-test | 1,000,000 | SciPy | 14.5 | 7.3 | 0.51× |
| Hochberg correction | 1,000,000 | statsmodels | 75.4 | 81.9 | **1.1×** |
| segmentation Dice and IoU per class | 1,000,000 px | scikit-learn | 40.2 | 282.0 | **7.0×** |
| COCO detection evaluation (12 numbers) | 1,000 images | pycocotools | 967.4 | 980.6 | **1.0×** |
| Hausdorff distance | 50 images | SciPy | 26.2 | 20.8 | 0.79× |

`evaluate()` validates inputs once and builds the confusion matrix once for all metrics, which is where most
of the speed-up comes from. Hypothesis tests use SciPy underneath, so they match its speed at best; rows
below 1× pay for input validation and the extra intervals and effect sizes EvalSuite reports. Reproduce on
your machine with `evalsuite benchmark`; full table (1k, 100k and 1M samples, peak memory) and notes in
[BENCHMARKS.md](https://github.com/mkcs28/evalsuite-python/blob/main/BENCHMARKS.md).

## Metrics in this release

**Classification** (binary, multiclass, multilabel; micro/macro/weighted/samples/per-class averaging;
sample weights): accuracy, balanced accuracy, precision, recall, specificity, NPV, F1, F-beta, Jaccard,
MCC, Cohen's kappa (unweighted, linear, quadratic), Hamming loss, confusion matrix, ROC AUC (binary,
one-vs-rest, one-vs-one), average precision, ROC and PR curves, log loss, Brier score, top-k accuracy,
calibration curve and expected calibration error.

**Clinical** (binary; `pos_label`; sample weights): sensitivity, specificity, PPV, NPV, positive and
negative likelihood ratios, diagnostic odds ratio, Youden's J, net benefit and decision curves, and a
diagnostic report with confidence intervals for all of them.

**Calibration**: calibration curve, Brier score, expected and maximum calibration error, calibration slope
and intercept, Hosmer–Lemeshow test.

**Statistics**: confidence intervals (bootstrap percentile/basic/BCa, Wilson, Clopper–Pearson, DeLong),
paired tests (McNemar, DeLong, paired bootstrap), t-tests (Welch, Student, paired), Mann–Whitney, Wilcoxon,
Kruskal–Wallis, Friedman, Shapiro–Wilk, χ², Fisher's exact; effect sizes (Cohen's d, Hedges' g, Cliff's
delta, Cramér's V); multiple-testing corrections (Bonferroni, Holm, Hochberg, Benjamini–Hochberg,
Benjamini–Yekutieli).

**Segmentation** (2-D and 3-D label masks; `ignore_index`; dataset or per-image aggregation): Dice, IoU,
mIoU, pixel accuracy, mean pixel accuracy, Boundary IoU, Hausdorff distance and HD95, average symmetric
surface distance (with pixel spacing), confusion matrix and a full report.

**Object detection**: box IoU (xyxy, xywh, cxcywh), COCO mAP@[.50:.95], mAP@.50, mAP@.75, mAP and mAR by
object size, AP per class with COCO or VOC interpolation, precision-recall curves, COCO file import.

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
