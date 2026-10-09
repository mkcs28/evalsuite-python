# Benchmarks

Reproduce with `evalsuite benchmark` (all cases) or one suite: `--suite core` (classification and
regression), `--suite clinical` (clinical, calibration and statistics), `--suite vision` (segmentation and
object detection), `--suite llm` (text generation, retrieval, rater agreement, structured output) or
`--suite metrics` (one row for every registered metric and statistics function; see [Every metric](#every-metric)). Add `--format markdown -o BENCHMARKS-local.md` to save a table.

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
| corpus BLEU and chrF | sacreBLEU |
| ROUGE-1/2/L | rouge-score |
| METEOR (exact and stem matches) | NLTK |
| MRR, MAP@20, NDCG@10 | ranx |
| Krippendorff's alpha | krippendorff |
| JSON Schema compliance | jsonschema |

For the LLM suite, `n / 100` examples are evaluated (sentences, queries, items or documents). For segmentation, `n` is the number of pixels (64 × 64 images with five classes); detection uses
`n / 1000` images with 1–7 objects of five classes each, plus false positives.

```text
EvalSuite 0.4.0 benchmarks | Python 3.12.3 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 10/12 | 5.88× | 0.47×–49.07× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 17/24 | 1.71× | 0.25×–11.13× |
| LLM evaluation | 6 | 18 | 18/18 | 10/18 | 1.48× | 0.83×–7.72× |
| Segmentation and object detection | 3 | 9 | 9/9 | 5/9 | 1.97× | 0.64×–10.39× |
| Overall | 21 | 63 | 63/63 | 42/63 | 2.12× | 0.25×–49.07× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.101 | 1.342 | 13.25× | 0.05 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 2.995 | 15.218 | 5.08× | 3.05 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 23.809 | 135.515 | 5.69× | 30.52 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.049 | 0.045 | 0.92× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.412 | 0.387 | 0.94× | 0.26 | 0.69 | 3.3e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 3.511 | 3.372 | 0.96× | 2.26 | 6.32 | 1.8e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.256 | 9.178 | 35.87× | 0.05 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 2.350 | 115.325 | 49.07× | 3.21 | 3.07 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 28.459 | 1023.820 | 35.98× | 31.54 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.167 | 1.460 | 8.73× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 17.642 | 27.624 | 1.57× | 9.16 | 7.64 | 0.0e+00 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 187.361 | 327.279 | 1.75× | 91.56 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.559 | 2.578 | 4.61× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 15.730 | 115.193 | 7.32× | 8.46 | 58.00 | 5.6e-17 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 205.659 | 1173.684 | 5.71× | 83.99 | 579.85 | 3.9e-16 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.235 | 0.523 | 2.23× | 0.05 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 1.951 | 1.602 | 0.82× | 3.05 | 0.29 | 7.1e-15 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 19.570 | 4.878 | 0.25× | 30.52 | 1.91 | 7.1e-15 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.309 | 3.440 | 11.13× | 0.05 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 6.632 | 43.403 | 6.54× | 3.05 | 2.24 | 5.6e-17 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 75.166 | 411.527 | 5.47× | 30.52 | 22.32 | 4.4e-16 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.268 | 1.437 | 5.36× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 11.350 | 15.891 | 1.40× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 165.010 | 201.852 | 1.22× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 13.097 | 21.589 | 1.65× | 0.54 | 1.33 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 102.270 | 93.783 | 0.92× | 1.15 | 4.29 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 838.714 | 848.551 | 1.01× | 6.92 | 34.02 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.045 | 0.077 | 1.71× | 0.05 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 4.867 | 4.675 | 0.96× | 4.58 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 72.148 | 78.964 | 1.09× | 45.78 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.095 | 0.650 | 6.85× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.912 | 1.763 | 0.92× | 2.29 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 22.854 | 10.633 | 0.47× | 22.89 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.182 | 1.407 | 7.72× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 29.821 | 77.001 | 2.58× | 0.49 | 3.12 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 170.388 | 861.009 | 5.05× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.314 | 2.882 | 9.19× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 3.864 | 40.162 | 10.39× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 34.283 | 256.019 | 7.47× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.515 | 0.397 | 0.77× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 21.556 | 13.741 | 0.64× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 24.165 | 19.832 | 0.82× | 0.15 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.259 | 0.409 | 1.58× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.259 | 0.275 | 1.07× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.231 | 0.232 | 1.00× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.640 | 0.564 | 0.88× | 0.16 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 33.722 | 38.490 | 1.14× | 15.45 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 353.520 | 340.573 | 0.96× | 154.50 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.821 | 0.987 | 1.20× | 0.04 | 0.02 | 0.0e+00 |
| statistics: Welch t-test | 100,000 | SciPy | 2.097 | 1.382 | 0.66× | 3.06 | 1.53 | 0.0e+00 |
| statistics: Welch t-test | 1,000,000 | SciPy | 16.124 | 8.349 | 0.52× | 30.52 | 15.26 | 0.0e+00 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.157 | 0.421 | 2.67× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 14.393 | 33.995 | 2.36× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 144.622 | 330.462 | 2.29× | 0.50 | 0.16 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 2.206 | 1.960 | 0.89× | 0.10 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 255.898 | 250.494 | 0.98× | 0.74 | 21.91 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 2789.790 | 3051.755 | 1.09× | 7.26 | 210.17 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.459 | 0.501 | 1.09× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 52.275 | 58.965 | 1.13× | 0.12 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 488.126 | 626.693 | 1.28× | 1.15 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 1.145 | 0.947 | 0.83× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 120.724 | 111.884 | 0.93× | 0.12 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 1290.112 | 1095.609 | 0.85× | 1.16 | 6.55 | 0.0e+00 |

## Reading the results

- **Many metrics at once is where EvalSuite is fastest.** `evaluate()` validates the inputs once and builds
  the confusion matrix once, then derives all eight label metrics from it: 21–33× faster than eight
  separate scikit-learn calls. The same applies to sensitivity, specificity, LR+ and LR− (4–11×) and to
  segmentation Dice and IoU per class (5–17×), which come from one pixel confusion matrix.
- **LLM metrics match their references exactly** and run at about their speed: BLEU and chrF
  (1.09× sacreBLEU at 10,000 sentences), ROUGE (0.85× rouge-score), METEOR
  (1.28× NLTK), ranking metrics (5.1× ranx, which compiles with numba), JSON Schema
  (2.3× jsonschema) and Krippendorff's alpha (0.96×).
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

## Every metric

One row for each of the 147 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.4.1, Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE.

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 4.10× | 1.26×–13.02× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 21/42 | 1.63× | 0.42×–12.40× |
| LLM evaluation | 69 | 23 | 33 | 13 | 56/56 | 36/46 | 2.31× | 0.09×–31.68× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 14/18 | 4.81× | 0.61×–19.12× |
| **All metrics** | 147 | 79 | 49 | 19 | 128/128 | 123/158 | 2.76× | 0.09×–31.68× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `calibration.calibration_intercept` | statsmodels | 1.294 | 10.199 | 7.88× | 15.405 | 69.147 | 4.49× | 1.1e-16 |
| `calibration.calibration_slope` | statsmodels | 3.138 | 17.346 | 5.53× | 24.893 | 128.4 | 5.16× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.849 | 1.181 | 1.39× | 5.513 | 8.269 | 1.50× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.461 | 2.010 | 4.36× | 3.086 | 6.713 | 2.18× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 2.049 | 5.350 | 2.61× | 21.297 | 27.823 | 1.31× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 1.295 | 5.367 | 4.14× | 8.491 | 23.279 | 2.74× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.406 | 2.544 | 6.26× | 2.051 | 10.699 | 5.22× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.517 | 4.991 | 9.65× | 3.120 | 18.903 | 6.06× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.811 | 1.252 | 1.54× | 5.131 | 8.000 | 1.56× | 2.2e-16 |
| `classification.f1` | scikit-learn | 1.343 | 7.498 | 5.58× | 8.461 | 26.210 | 3.10× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.547 | 4.675 | 8.55× | 3.301 | 21.736 | 6.59× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 1.025 | 6.102 | 5.95× | 12.611 | 47.141 | 3.74× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.395 | 4.312 | 10.92× | 3.103 | 21.165 | 6.82× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 1.088 | 6.704 | 6.16× | 12.068 | 50.632 | 4.20× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.930 | 7.016 | 7.55× | 8.863 | 43.643 | 4.92× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.341 | 4.434 | 13.02× | 3.248 | 21.661 | 6.67× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.957 | 4.978 | 5.20× | 9.220 | 28.700 | 3.11× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.891 | 5.085 | 5.70× | 13.204 | 26.347 | 2.00× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 16.630 | 40.865 | 2.46× | 197.6 | 365.5 | 1.85× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.374 | 4.417 | 11.80× | 5.622 | 25.320 | 4.50× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 1.896 | 5.838 | 3.08× | 15.198 | 36.280 | 2.39× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.569 | 0.124 | 0.22× | 6.020 | 0.848 | 0.14× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.552 | 5.910 | 10.71× | 5.798 | 32.128 | 5.54× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.572 | 6.342 | 11.09× | 3.978 | 26.874 | 6.76× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.556 | 0.048 | 0.09× | 3.655 | 0.295 | 0.08× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.671 | 4.462 | 6.65× | 3.493 | 25.830 | 7.39× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.360 | 4.466 | 12.40× | 4.490 | 27.887 | 6.21× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.318 | 0.076 | 0.24× | 3.614 | 0.654 | 0.18× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 4.981 | 36.096 | 7.25× | 91.231 | 239.4 | 2.62× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.103 | 0.049 | 0.48× | 17.375 | 11.456 | 0.66× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 5.641 | 38.163 | 6.77× | 64.102 | 191.8 | 2.99× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 1.810 | 12.314 | 6.80× | 19.749 | 126.8 | 6.42× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 2.804 | 12.694 | 4.53× | 30.606 | 131.5 | 4.30× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.920 | 0.904 | 0.98× | 8.427 | 9.584 | 1.14× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.838 | 0.686 | 0.82× | 6.987 | 6.608 | 0.95× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.847 | 0.626 | 0.74× | 7.205 | 6.758 | 0.94× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.045 | – | – | 0.439 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.258 | 0.059 | 0.23× | 0.473 | 0.100 | 0.21× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.014 | 0.006 | 0.44× | 0.027 | 0.013 | 0.50× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.495 | – | – | 4.592 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 1.018 | 0.207 | 0.20× | 10.740 | 2.170 | 0.20× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.582 | 0.043 | 0.07× | 5.028 | 0.374 | 0.07× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.116 | 0.030 | 0.26× | 0.936 | 0.346 | 0.37× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.114 | 0.636 | 5.58× | 0.783 | 1.572 | 2.01× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.133 | 0.081 | 0.61× | 1.282 | 0.940 | 0.73× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.054 | 0.414 | 7.60× | 0.348 | 0.811 | 2.33× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.083 | 0.443 | 5.33× | 0.694 | 1.282 | 1.85× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.039 | 0.312 | 8.03× | 0.307 | 0.593 | 1.93× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.048 | 0.013 | 0.28× | 0.276 | 0.145 | 0.53× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.277 | 0.627 | 2.26× | 1.910 | 2.407 | 1.26× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.054 | 0.393 | 7.30× | 0.363 | 0.862 | 2.38× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.118 | 0.798 | 6.76× | 0.967 | 2.058 | 2.13× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.069 | 0.485 | 7.06× | 0.761 | 2.068 | 2.72× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.111 | 0.498 | 4.50× | 1.245 | 1.631 | 1.31× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.132 | 0.030 | 0.22× | 1.228 | 0.312 | 0.25× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.064 | 0.013 | 0.20× | 0.355 | 0.089 | 0.25× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.120 | 0.061 | 0.51× | 1.022 | 0.701 | 0.69× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.113 | 0.030 | 0.26× | 1.314 | 0.355 | 0.27× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.101 | 0.037 | 0.37× | 1.139 | 0.579 | 0.51× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.788 | 12.397 | 15.73× | 6.760 | 104.7 | 15.49× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.852 | 11.951 | 14.03× | 7.468 | 98.526 | 13.19× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.804 | 11.272 | 14.03× | 6.985 | 98.544 | 14.11× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 1.089 | 17.298 | 15.89× | 9.967 | 98.781 | 9.91× | 0.0e+00 |
| `retrieval.precision_at_k` | ranx | 0.784 | 11.431 | 14.58× | 9.390 | 101.7 | 10.83× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 1.374 | 10.869 | 7.91× | 7.055 | 104.6 | 14.82× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 1.471 | 0.900 | 0.61× | 18.541 | 11.489 | 0.62× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.596 | – | – | 6.099 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.293 | 3.796 | 12.96× | 2.974 | 22.166 | 7.45× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 1.871 | 1.360 | 0.73× | 24.299 | 17.352 | 0.71× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.290 | 3.822 | 13.18× | 2.770 | 20.944 | 7.56× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.257 | 3.953 | 15.41× | 3.660 | 21.533 | 5.88× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.315 | 3.879 | 12.32× | 2.896 | 21.262 | 7.34× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.058 | 1.112 | 19.12× | 0.597 | 9.955 | 16.67× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.026 | 0.151 | 5.82× | 0.181 | 0.135 | 0.75× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.653 | 0.800 | 1.23× | 10.822 | 7.880 | 0.73× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 629.6 | – | – | 6,070.5 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.981 | 0.447 | 0.46× | 1.058 | 0.439 | 0.42× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 2.179 | 3.449 | 1.58× | 28.744 | 33.009 | 1.15× | 3.1e-17 |
| `statistics.cohens_d` | NumPy formula | 0.096 | 0.062 | 0.65× | 0.878 | 0.490 | 0.56× | 0.0e+00 |
| `statistics.compare` | timed alone | 184.2 | – | – | 2,478.1 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.465 | 0.456 | 0.98× | 0.451 | 0.428 | 0.95× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 7.277 | – | – | 74.261 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 1.350 | 1.161 | 0.86× | 5.707 | 5.431 | 0.95× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 3.243 | 2.972 | 0.92× | 26.927 | 26.232 | 0.97× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.151 | 0.065 | 0.43× | 1.191 | 0.448 | 0.38× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 4.804 | 4.810 | 1.00× | 59.376 | 51.059 | 0.86× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 3.599 | 3.707 | 1.03× | 35.165 | 40.229 | 1.14× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.163 | 0.131 | 0.80× | 0.422 | 0.642 | 1.52× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 189.9 | – | – | 2,873.6 | – | – | – |
| `statistics.paired_t_test` | SciPy | 1.083 | 0.673 | 0.62× | 4.381 | 2.479 | 0.57× | 0.0e+00 |
| `statistics.proportion_ci` | statsmodels | 0.233 | 0.168 | 0.72× | 0.314 | 0.316 | 1.01× | 1.1e-16 |
| `statistics.roc_auc_ci` | timed alone | 3.660 | – | – | 33.356 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 1.002 | 0.986 | 0.98× | 0.972 | 0.847 | 0.87× | 0.0e+00 |
| `statistics.t_test` | SciPy | 1.691 | 1.341 | 0.79× | 4.075 | 3.363 | 0.83× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 4.320 | 2.622 | 0.61× | 44.235 | 22.984 | 0.52× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.035 | 0.007 | 0.21× | 0.057 | 0.011 | 0.19× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.034 | 0.009 | 0.26× | 0.082 | 0.025 | 0.30× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 3.052 | 2.168 | 0.71× | 47.337 | 23.809 | 0.50× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.023 | 0.011 | 0.50× | 0.389 | 0.118 | 0.30× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.715 | 0.381 | 0.53× | 12.750 | 7.714 | 0.61× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 9.666 | 8.816 | 0.91× | 137.1 | 93.806 | 0.68× | 0.0e+00 |
| `text.bradley_terry` | choix | 3.092 | 66.178 | 21.40× | 3.511 | 111.2 | 31.68× | 3.9e-10 |
| `text.chrf` | sacreBLEU | 19.344 | 19.513 | 1.01× | 246.1 | 225.8 | 0.92× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 26.735 | 45.046 | 1.68× | 317.2 | 498.7 | 1.57× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.302 | – | – | 2.974 | – | – | – |
| `text.citation_recall` | timed alone | 0.224 | – | – | 4.021 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.286 | 0.009 | 0.03× | 1.999 | 0.027 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.069 | 0.101 | 1.47× | 0.628 | 2.541 | 4.04× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.091 | 0.022 | 0.24× | 1.660 | 0.311 | 0.19× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.810 | 0.277 | 0.34× | 10.684 | 5.785 | 0.54× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.672 | 0.202 | 0.30× | 3.830 | 1.212 | 0.32× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.075 | 0.048 | 0.64× | 0.621 | 0.461 | 0.74× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.444 | – | – | 4.723 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.489 | 0.787 | 1.61× | 6.996 | 8.535 | 1.22× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.642 | 0.199 | 0.31× | 5.396 | 1.342 | 0.25× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.052 | 0.035 | 0.67× | 0.358 | 0.271 | 0.76× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.957 | 0.246 | 0.26× | 8.926 | 2.504 | 0.28× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.359 | 0.330 | 0.92× | 3.642 | 1.775 | 0.49× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.097 | 0.035 | 0.36× | 0.487 | 0.131 | 0.27× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.189 | – | – | 0.773 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 1.970 | 4.921 | 2.50× | 17.982 | 45.671 | 2.54× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.241 | 0.208 | 0.86× | 2.428 | 2.018 | 0.83× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.235 | 1.467 | 6.25× | 1.020 | 3.247 | 3.18× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.091 | 0.021 | 0.23× | 1.491 | 0.267 | 0.18× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.260 | 0.211 | 0.81× | 0.922 | 1.015 | 1.10× | 3.1e-15 |
| `text.mauve` | timed alone | 8.807 | – | – | 933.6 | – | – | – |
| `text.meteor` | NLTK | 6.863 | 10.587 | 1.54× | 69.039 | 94.850 | 1.37× | 0.0e+00 |
| `text.model_score` | timed alone | 0.052 | – | – | 0.158 | – | – | – |
| `text.moverscore` | POT | 35.152 | 5.003 | 0.14× | 363.6 | 32.906 | 0.09× | 8.3e-17 |
| `text.perplexity` | NumPy formula | 0.097 | 0.022 | 0.22× | 1.029 | 0.151 | 0.15× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.127 | 0.007 | 0.05× | 0.930 | 0.011 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.490 | – | – | 5.385 | – | – | – |
| `text.rouge_1` | rouge-score | 6.124 | 7.848 | 1.28× | 40.842 | 45.725 | 1.12× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 5.983 | 7.281 | 1.22× | 40.809 | 69.920 | 1.71× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 4.994 | 5.836 | 1.17× | 56.425 | 73.766 | 1.31× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 7.580 | 8.434 | 1.11× | 81.529 | 91.904 | 1.13× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.032 | 0.010 | 0.32× | 0.073 | 0.018 | 0.25× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 12.149 | 18.683 | 1.54× | 301.0 | 571.7 | 1.90× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.843 | – | – | 1.297 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 9.444 | 10.623 | 1.12× | 109.8 | 117.5 | 1.07× | 0.0e+00 |
| `text.ter` | sacreBLEU | 45.166 | 27.133 | 0.60× | 475.2 | 298.1 | 0.63× | 0.0e+00 |
| `text.tool_argument_accuracy` | timed alone | 0.403 | – | – | 5.206 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.448 | – | – | 5.549 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.356 | 0.041 | 0.12× | 4.558 | 0.362 | 0.08× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 1.138 | – | – | 5.750 | – | – | – |
| `text.win_rate` | NumPy formula | 0.071 | 0.011 | 0.15× | 1.045 | 0.056 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.685 | 0.619 | 0.90× | 11.861 | 6.926 | 0.58× | 0.0e+00 |
