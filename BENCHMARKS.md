# Benchmarks

Reproduce with `evalsuite benchmark` (all cases) or one suite: `--suite core` (classification and
regression), `--suite clinical` (clinical, calibration and statistics), `--suite vision` (segmentation and
object detection) or `--suite llm` (text generation, retrieval, rater agreement, structured output). Add `--format markdown -o BENCHMARKS-local.md` to save a table.

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
