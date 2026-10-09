# Benchmarks

Reproduce with `evalsuite benchmark` (all cases) or one suite: `--suite core` (classification and
regression), `--suite clinical` (clinical, calibration and statistics) or `--suite vision` (segmentation and
object detection). Add `--format markdown -o BENCHMARKS-local.md` to save a table.

Each case is timed as the fastest of 5 runs after a warm-up; peak memory is measured with `tracemalloc`.
EvalSuite and a **reference implementation compute the same quantities on the same data**, and the last
column is the largest absolute difference between their results: every row agrees to floating-point
rounding.

| Cases | Reference |
| --- | --- |
| classification, regression, sensitivity/specificity/likelihood ratios, segmentation Dice and IoU | scikit-learn |
| diagnostic report intervals, calibration slope and intercept, Hochberg correction | statsmodels |
| t-test, Mann–Whitney, Cramér's V, Hausdorff distance | SciPy |
| COCO object detection (all 12 summary numbers) | pycocotools |
| decision curve | the textbook NumPy loop over thresholds |

For segmentation, `n` is the number of pixels (64 × 64 images with five classes); detection uses
`n / 1000` images with 1–7 objects of five classes each, plus false positives.

```text
EvalSuite 0.3.0 benchmarks | Python 3.12.3 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.255 | 9.554 | 37.42× | 0.05 | 0.05 | 0.0e+00 |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.102 | 1.319 | 12.95× | 0.05 | 0.03 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.176 | 1.471 | 8.35× | 0.09 | 0.08 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.091 | 0.648 | 7.11× | 0.03 | 0.02 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.310 | 3.449 | 11.12× | 0.05 | 0.03 | 4.4e-16 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.151 | 0.531 | 3.52× | 0.05 | 0.01 | 1.1e-14 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.487 | 2.513 | 5.16× | 0.10 | 0.61 | 2.8e-16 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.156 | 0.897 | 5.76× | 0.07 | 0.01 | 5.6e-17 |
| statistics: Welch t-test | 1,000 | SciPy | 0.819 | 0.623 | 0.76× | 0.04 | 0.02 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.649 | 0.602 | 0.93× | 0.16 | 0.14 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.260 | 0.234 | 0.90× | 0.00 | 0.00 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.046 | 0.053 | 1.16× | 0.05 | 0.05 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.232 | 3.282 | 14.15× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.548 | 0.399 | 0.73× | 0.15 | 0.03 | 0.0e+00 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 12.599 | 22.011 | 1.75× | 0.54 | 1.33 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 4.382 | 112.468 | 25.67× | 3.21 | 3.07 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 4.098 | 13.204 | 3.22× | 3.05 | 2.18 | 0.0e+00 |
| binary: ROC AUC | 100,000 | scikit-learn | 16.161 | 31.224 | 1.93× | 9.16 | 7.64 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.905 | 1.587 | 0.83× | 2.29 | 1.53 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 11.091 | 42.599 | 3.84× | 3.05 | 2.24 | 8.9e-16 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 2.895 | 0.976 | 0.34× | 3.05 | 0.29 | 2.8e-14 |
| calibration: slope and intercept | 100,000 | statsmodels | 18.032 | 100.467 | 5.57× | 8.46 | 58.00 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 12.551 | 15.825 | 1.26× | 6.87 | 0.29 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 2.086 | 1.306 | 0.63× | 3.06 | 1.53 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 31.660 | 30.400 | 0.96× | 15.45 | 13.93 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.259 | 0.262 | 1.01× | 0.00 | 0.00 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 4.123 | 4.428 | 1.07× | 4.58 | 3.97 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 3.719 | 26.436 | 7.11× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 11.498 | 9.098 | 0.79× | 0.15 | 0.03 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 81.104 | 91.893 | 1.13× | 1.15 | 4.32 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 34.020 | 1045.806 | 30.74× | 31.54 | 30.53 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 31.727 | 129.069 | 4.07× | 30.52 | 21.79 | 0.0e+00 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 203.583 | 348.552 | 1.71× | 91.56 | 76.30 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 21.052 | 10.722 | 0.51× | 22.89 | 15.26 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 75.990 | 426.237 | 5.61× | 30.52 | 22.33 | 2.8e-17 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 18.828 | 5.073 | 0.27× | 30.52 | 1.91 | 2.5e-14 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 192.705 | 1096.539 | 5.69× | 83.99 | 579.85 | 3.3e-16 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 171.516 | 198.074 | 1.15× | 68.67 | 1.97 | 5.6e-17 |
| statistics: Welch t-test | 1,000,000 | SciPy | 16.322 | 7.974 | 0.49× | 30.52 | 15.26 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 395.086 | 378.419 | 0.96× | 154.50 | 139.24 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.290 | 0.283 | 0.97× | 0.00 | 0.00 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 85.440 | 88.325 | 1.03× | 45.78 | 39.17 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 40.178 | 281.966 | 7.02× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 26.171 | 20.775 | 0.79× | 0.15 | 0.03 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 967.372 | 980.558 | 1.01× | 7.03 | 34.75 | 0.0e+00 |

## Reading the results

- **Many metrics at once is where EvalSuite is fastest.** `evaluate()` validates the inputs once and builds
  the confusion matrix once, then derives all eight label metrics from it: 26–37× faster than eight
  separate scikit-learn calls. The same applies to sensitivity, specificity, LR+ and LR− (4–11×) and to
  segmentation Dice and IoU per class (7–14×), which come from one pixel confusion matrix.
- **COCO detection evaluation matches pycocotools exactly and is as fast or faster** (1.0–1.8×). IoUs are
  computed once per image and class and reused for every area range and IoU threshold, and the greedy
  matching is vectorised over thresholds.
- **Calibration slope and intercept are 5–6× faster than statsmodels' GLM and use far less memory**
  (84 MiB vs 580 MiB at 1M samples).
- **Hausdorff distance (0.7–0.8×)** uses a Euclidean distance transform per class, which also supports
  anisotropic pixel spacing and HD95; the reference computes the exact point-set distance between boundary
  points of one class. Both give identical results.
- **Hypothesis tests call SciPy**, so they match its speed at best; the Welch t-test also computes the
  confidence interval and Cohen's d.
- **Slower rows, shown on purpose:** the diagnostic report (0.3× at large n) validates labels and reports
  ten measures where the reference computes seven intervals from counts; regression on a million values
  (0.5×) spends most of its ~20 ms checking every value for NaN, infinity, shape and dtype.

Numbers depend on the machine, Python and NumPy versions; run `evalsuite benchmark` on your own hardware.
