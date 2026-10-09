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
EvalSuite 0.3.1 benchmarks | Python 3.12.3 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 10/12 | 4.98× | 0.47×–33.31× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 16/24 | 1.54× | 0.20×–10.89× |
| Segmentation and object detection | 3 | 9 | 9/9 | 6/9 | 2.04× | 0.73×–16.64× |
| Overall | 15 | 45 | 45/45 | 32/45 | 2.22× | 0.20×–33.31× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.187 | 2.203 | 11.77× | 0.05 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 3.805 | 15.511 | 4.08× | 3.05 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 29.263 | 130.816 | 4.47× | 30.52 | 21.79 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.301 | 10.020 | 33.31× | 0.05 | 0.04 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 4.874 | 104.589 | 21.46× | 3.21 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 39.501 | 1084.077 | 27.44× | 31.54 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.305 | 1.936 | 6.35× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 17.567 | 30.822 | 1.75× | 9.16 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 194.983 | 348.267 | 1.79× | 91.56 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.493 | 2.678 | 5.43× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 18.060 | 98.404 | 5.45× | 8.46 | 58.00 | 5.6e-17 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 192.967 | 1068.024 | 5.53× | 83.99 | 579.85 | 3.3e-16 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.143 | 0.483 | 3.37× | 0.05 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 2.713 | 1.150 | 0.42× | 3.05 | 0.29 | 2.8e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 23.480 | 4.741 | 0.20× | 30.52 | 1.91 | 2.5e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.325 | 3.543 | 10.89× | 0.05 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 10.088 | 45.050 | 4.47× | 3.05 | 2.24 | 8.9e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 69.122 | 414.038 | 5.99× | 30.52 | 22.33 | 2.8e-17 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.160 | 0.802 | 5.01× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 10.888 | 14.087 | 1.29× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 156.333 | 181.613 | 1.16× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 12.039 | 22.435 | 1.86× | 0.54 | 1.33 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 89.151 | 98.418 | 1.10× | 1.15 | 4.32 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 817.295 | 983.691 | 1.20× | 7.03 | 34.75 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.043 | 0.049 | 1.15× | 0.05 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 4.047 | 4.659 | 1.15× | 4.58 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 81.990 | 108.041 | 1.32× | 45.78 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.151 | 1.057 | 6.99× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.887 | 1.622 | 0.86× | 2.29 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 23.082 | 10.751 | 0.47× | 22.89 | 15.26 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.200 | 3.326 | 16.64× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 4.867 | 26.617 | 5.47× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 42.958 | 290.018 | 6.75× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.561 | 0.412 | 0.73× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 11.991 | 8.698 | 0.73× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 27.050 | 20.239 | 0.75× | 0.15 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.225 | 0.241 | 1.07× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.292 | 0.241 | 0.83× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.480 | 0.422 | 0.88× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.676 | 0.547 | 0.81× | 0.16 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 30.010 | 31.381 | 1.05× | 15.45 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 349.746 | 350.183 | 1.00× | 154.50 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.725 | 0.622 | 0.86× | 0.04 | 0.02 | 0.0e+00 |
| statistics: Welch t-test | 100,000 | SciPy | 2.103 | 1.240 | 0.59× | 3.06 | 1.53 | 0.0e+00 |
| statistics: Welch t-test | 1,000,000 | SciPy | 14.524 | 7.161 | 0.49× | 30.52 | 15.26 | 0.0e+00 |

## Reading the results

- **Many metrics at once is where EvalSuite is fastest.** `evaluate()` validates the inputs once and builds
  the confusion matrix once, then derives all eight label metrics from it: 21–33× faster than eight
  separate scikit-learn calls. The same applies to sensitivity, specificity, LR+ and LR− (4–11×) and to
  segmentation Dice and IoU per class (5–17×), which come from one pixel confusion matrix.
- **COCO detection evaluation matches pycocotools exactly and is faster** (1.1–1.9×). IoUs are
  computed once per image and class and reused for every area range and IoU threshold, and the greedy
  matching is vectorised over thresholds.
- **Calibration slope and intercept are 5–6× faster than statsmodels' GLM and use far less memory**
  (84 MiB vs 580 MiB at 1M samples).
- **Hausdorff distance (0.7–0.8×)** uses a Euclidean distance transform per class, which also supports
  anisotropic pixel spacing and HD95; the reference computes the exact point-set distance between boundary
  points of one class. Both give identical results.
- **Hypothesis tests call SciPy**, so they match its speed at best; the Welch t-test also computes the
  confidence interval and Cohen's d.
- **Slower rows, shown on purpose:** the diagnostic report (0.2–0.4× at large n) validates labels and reports
  ten measures where the reference computes seven intervals from counts; regression on a million values
  (0.5×) spends most of its ~20 ms checking every value for NaN, infinity, shape and dtype.

Numbers depend on the machine, Python and NumPy versions; run `evalsuite benchmark` on your own hardware.
