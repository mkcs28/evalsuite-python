# Benchmarks

Reproduce with `evalsuite benchmark` (all cases) or one suite: `--suite core` (classification and
regression), `--suite clinical` (clinical, calibration and statistics), `--suite vision` (segmentation and
object detection), `--suite llm` (text generation, retrieval, rater agreement, structured output), `--suite llmsys` (v0.5 LLM systems: radon, codebleu, scikit-learn,
jsonschema, SciPy, NumPy references) or
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
EvalSuite 0.5.1 benchmarks | Python 3.13.16 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 10/12 | 5.84× | 0.71×–55.53× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 14/24 | 1.59× | 0.29×–11.21× |
| LLM evaluation | 6 | 18 | 18/18 | 12/18 | 1.47× | 0.78×–7.49× |
| LLM systems | 6 | 18 | 18/18 | 14/18 | 1.40× | 0.33×–5.77× |
| Segmentation and object detection | 3 | 9 | 9/9 | 6/9 | 1.95× | 0.72×–13.51× |
| Overall | 27 | 81 | 81/81 | 56/81 | 1.89× | 0.29×–55.53× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.135 | 2.049 | 15.15× | 0.05 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 2.830 | 17.233 | 6.09× | 4.58 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 41.480 | 137.611 | 3.32× | 45.78 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.075 | 0.071 | 0.96× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.469 | 0.425 | 0.91× | 0.26 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 4.455 | 4.475 | 1.00× | 2.26 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.356 | 19.784 | 55.53× | 0.05 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 3.002 | 124.020 | 41.31× | 4.58 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 49.432 | 1056.751 | 21.38× | 45.78 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.199 | 1.802 | 9.08× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 18.883 | 34.770 | 1.84× | 9.16 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 200.276 | 338.540 | 1.69× | 91.56 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.862 | 4.383 | 5.08× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 25.106 | 141.277 | 5.63× | 8.46 | 58.06 | 5.6e-17 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 344.319 | 1541.663 | 4.48× | 83.99 | 579.91 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.178 | 0.701 | 3.94× | 0.05 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 2.238 | 1.286 | 0.57× | 4.58 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 33.462 | 9.604 | 0.29× | 45.78 | 2.86 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.432 | 4.844 | 11.21× | 0.05 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 10.032 | 47.526 | 4.74× | 4.58 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 137.871 | 438.134 | 3.18× | 45.78 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.182 | 1.192 | 6.56× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 12.646 | 19.108 | 1.51× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 172.085 | 236.221 | 1.37× | 68.67 | 2.86 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 19.337 | 30.575 | 1.58× | 0.98 | 1.74 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 131.793 | 139.670 | 1.06× | 1.49 | 4.71 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 1147.898 | 1351.027 | 1.18× | 7.00 | 34.44 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.053 | 0.098 | 1.84× | 0.05 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 5.519 | 5.879 | 1.07× | 4.58 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 91.858 | 98.422 | 1.07× | 45.78 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.133 | 0.831 | 6.23× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 2.645 | 2.239 | 0.85× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 32.001 | 22.648 | 0.71× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.244 | 1.830 | 7.49× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 21.350 | 91.433 | 4.28× | 0.49 | 3.13 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 248.612 | 897.025 | 3.61× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.297 | 4.018 | 13.51× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 4.509 | 30.276 | 6.71× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 51.648 | 289.514 | 5.61× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.633 | 0.454 | 0.72× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 16.117 | 12.375 | 0.77× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 31.882 | 23.820 | 0.75× | 0.15 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.314 | 0.289 | 0.92× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.305 | 0.281 | 0.92× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.300 | 0.284 | 0.95× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.738 | 0.712 | 0.96× | 0.16 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 33.733 | 32.472 | 0.96× | 15.45 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 390.304 | 425.774 | 1.09× | 154.50 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 1.013 | 0.858 | 0.85× | 0.04 | 0.02 | 0.0e+00 |
| statistics: Welch t-test | 100,000 | SciPy | 2.705 | 1.593 | 0.59× | 3.06 | 1.53 | 0.0e+00 |
| statistics: Welch t-test | 1,000,000 | SciPy | 37.652 | 16.272 | 0.43× | 30.52 | 15.26 | 0.0e+00 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.210 | 0.412 | 1.97× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 16.044 | 37.886 | 2.36× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 159.359 | 371.723 | 2.33× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 67.412 | 74.305 | 1.10× | 0.37 | 0.25 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 740.336 | 791.899 | 1.07× | 1.73 | 2.06 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 2.932 | 2.743 | 0.94× | 0.07 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.336 | 1.860 | 5.54× | 0.06 | 0.08 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 16.460 | 29.417 | 1.79× | 6.20 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 194.432 | 327.696 | 1.69× | 61.99 | 76.30 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.133 | 0.768 | 5.77× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.739 | 1.671 | 2.26× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 13.292 | 16.853 | 1.27× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.133 | 0.309 | 2.32× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 14.180 | 33.665 | 2.37× | 0.70 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 145.032 | 333.157 | 2.30× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.158 | 0.052 | 0.33× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 3.728 | 1.998 | 0.54× | 0.77 | 0.77 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 41.023 | 19.664 | 0.48× | 7.63 | 7.63 | 0.0e+00 |
| systems: maintainability index (100 programs) | 100,000 | radon | 13.903 | 14.967 | 1.08× | 0.18 | 0.04 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 138.344 | 168.738 | 1.22× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.497 | 0.514 | 1.03× | 0.04 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 2.306 | 2.007 | 0.87× | 0.10 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 257.790 | 258.032 | 1.00× | 0.73 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 2726.175 | 3120.972 | 1.14× | 7.28 | 211.08 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.491 | 0.553 | 1.13× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 50.158 | 57.141 | 1.14× | 0.12 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 507.427 | 590.285 | 1.16× | 1.15 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 1.086 | 0.851 | 0.78× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 110.226 | 97.261 | 0.88× | 0.12 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 1147.257 | 1010.657 | 0.88× | 1.16 | 6.55 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.5.0, Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 3.81× | 1.18×–12.15× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 23/42 | 1.66× | 0.40×–11.65× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 37/46 | 2.35× | 0.10×–45.31× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 13/14 | 2.30× | 0.33×–7.18× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 14/18 | 4.49× | 0.56×–14.79× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 139/172 | 2.67× | 0.10×–45.31× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.011 | 0.003 | 0.28× | 0.013 | 0.003 | 0.26× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.015 | 0.006 | 0.36× | 0.021 | 0.006 | 0.30× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 1.450 | 4.066 | 2.80× | 18.625 | 48.692 | 2.61× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.289 | – | – | 3.006 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.172 | 0.016 | 0.09× | 1.732 | 0.101 | 0.06× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.093 | 0.003 | 0.03× | 0.108 | 0.003 | 0.03× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.005 | 0.004 | 0.79× | 0.006 | 0.005 | 0.87× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.227 | – | – | 3.361 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.031 | 0.004 | 0.12× | 0.042 | 0.005 | 0.12× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 1.127 | 0.863 | 0.77× | 14.451 | 15.052 | 1.04× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.922 | 5.123 | 5.56× | 9.249 | 42.199 | 4.56× | 1.1e-16 |
| `calibration.calibration_slope` | statsmodels | 1.600 | 7.997 | 5.00× | 15.682 | 78.986 | 5.04× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.510 | 0.669 | 1.31× | 6.794 | 6.770 | 1.00× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.223 | 0.823 | 3.70× | 2.143 | 4.486 | 2.09× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 1.205 | 2.682 | 2.23× | 15.475 | 23.181 | 1.50× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.635 | 2.423 | 3.82× | 7.518 | 17.147 | 2.28× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.168 | 1.198 | 7.12× | 1.540 | 8.375 | 5.44× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.249 | 2.025 | 8.12× | 2.211 | 14.100 | 6.38× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.492 | 0.730 | 1.49× | 4.680 | 9.730 | 2.08× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.730 | 3.466 | 4.75× | 10.486 | 19.363 | 1.85× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.260 | 2.941 | 11.32× | 2.995 | 15.334 | 5.12× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.750 | 4.293 | 5.73× | 11.222 | 36.722 | 3.27× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.247 | 2.782 | 11.25× | 2.412 | 14.915 | 6.19× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.760 | 4.552 | 5.99× | 10.324 | 33.064 | 3.20× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.743 | 4.788 | 6.44× | 7.113 | 30.193 | 4.24× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.265 | 2.911 | 10.98× | 2.213 | 15.117 | 6.83× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.726 | 3.700 | 5.10× | 6.608 | 19.719 | 2.98× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.715 | 3.592 | 5.03× | 6.830 | 18.685 | 2.74× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 13.179 | 29.314 | 2.22× | 168.5 | 258.1 | 1.53× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.253 | 3.073 | 12.15× | 2.361 | 14.922 | 6.32× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 1.017 | 3.073 | 3.02× | 10.937 | 24.458 | 2.24× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.242 | 0.058 | 0.24× | 2.115 | 0.430 | 0.20× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.236 | 2.715 | 11.51× | 2.093 | 16.641 | 7.95× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.247 | 2.746 | 11.13× | 2.308 | 16.659 | 7.22× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.261 | 0.030 | 0.11× | 2.574 | 0.219 | 0.09× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.254 | 2.831 | 11.16× | 2.502 | 14.584 | 5.83× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.256 | 2.981 | 11.65× | 2.134 | 15.109 | 7.08× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.231 | 0.057 | 0.25× | 2.226 | 0.429 | 0.19× | 0.0e+00 |
| `code.code_complexity` | radon | 1.140 | 1.193 | 1.05× | 10.844 | 11.803 | 1.09× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 7.688 | 9.051 | 1.18× | 71.991 | 84.568 | 1.17× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.027 | 0.002 | 0.06× | 0.033 | 0.002 | 0.07× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.030 | 0.023 | 0.78× | 0.195 | 0.145 | 0.74× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.005 | 0.004 | 0.80× | 0.005 | 0.004 | 0.82× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.135 | 0.021 | 0.15× | 1.032 | 0.146 | 0.14× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.050 | 0.004 | 0.07× | 0.063 | 0.006 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.147 | 0.011 | 0.08× | 1.205 | 0.066 | 0.05× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.023 | 0.002 | 0.07× | 0.028 | 0.002 | 0.08× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 2.826 | 1.603 | 0.57× | 27.172 | 15.106 | 0.56× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.584 | 0.521 | 0.89× | 6.061 | 5.154 | 0.85× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 3.291 | 25.896 | 7.87× | 40.825 | 135.8 | 3.33× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.059 | 0.032 | 0.53× | 9.874 | 6.244 | 0.63× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 4.041 | 26.242 | 6.49× | 44.168 | 137.7 | 3.12× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.020 | 0.004 | 0.20× | 0.086 | 0.004 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.012 | 0.004 | 0.36× | 0.015 | 0.006 | 0.40× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.027 | 0.006 | 0.21× | 0.031 | 0.011 | 0.35× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.139 | 0.043 | 0.31× | 0.161 | 0.048 | 0.29× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.010 | 0.006 | 0.60× | 0.012 | 0.004 | 0.33× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.130 | 0.002 | 0.01× | 0.137 | 0.002 | 0.02× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.034 | 0.004 | 0.13× | 0.042 | 0.005 | 0.12× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.113 | 0.005 | 0.05× | 0.151 | 0.008 | 0.05× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.094 | 0.004 | 0.04× | 0.112 | 0.005 | 0.04× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.167 | 0.004 | 0.02× | 0.208 | 0.005 | 0.02× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.615 | 0.161 | 0.26× | 6.054 | 1.646 | 0.27× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.619 | 0.509 | 0.82× | 5.928 | 5.632 | 0.95× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.134 | 0.007 | 0.05× | 1.543 | 0.060 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.786 | 0.022 | 0.03× | 7.793 | 0.173 | 0.02× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.071 | 0.027 | 0.39× | 0.082 | 0.024 | 0.29× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.541 | 0.004 | 0.01× | 0.779 | 0.005 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.089 | 0.039 | 0.44× | 0.139 | 0.053 | 0.38× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.074 | 0.005 | 0.06× | 0.099 | 0.005 | 0.06× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 1.196 | 0.478 | 0.40× | 11.914 | 4.978 | 0.42× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.065 | 0.034 | 0.53× | 0.122 | 0.081 | 0.66× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.018 | 0.008 | 0.41× | 0.021 | 0.005 | 0.26× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.695 | 0.027 | 0.04× | 7.209 | 0.231 | 0.03× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.015 | 0.005 | 0.36× | 0.019 | 0.007 | 0.37× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.155 | 1.114 | 7.18× | 0.666 | 2.232 | 3.35× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.093 | 0.005 | 0.05× | 0.895 | 0.015 | 0.02× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.133 | 0.387 | 2.91× | 2.027 | 0.665 | 0.33× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.052 | 0.025 | 0.48× | 0.106 | 0.046 | 0.44× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 1.464 | 9.160 | 6.26× | 15.320 | 96.051 | 6.27× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 2.207 | 9.895 | 4.48× | 23.697 | 103.7 | 4.38× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.744 | 0.747 | 1.00× | 7.297 | 8.838 | 1.21× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.539 | 0.478 | 0.89× | 5.841 | 5.220 | 0.89× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.535 | 0.474 | 0.89× | 5.545 | 5.028 | 0.91× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.033 | – | – | 0.251 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.198 | 0.044 | 0.22× | 0.216 | 0.047 | 0.22× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.009 | 0.004 | 0.45× | 0.010 | 0.005 | 0.46× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.459 | – | – | 3.835 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.876 | 0.186 | 0.21× | 8.772 | 1.866 | 0.21× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.459 | 0.032 | 0.07× | 4.136 | 0.280 | 0.07× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.090 | 0.024 | 0.27× | 0.691 | 0.239 | 0.35× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.082 | 0.369 | 4.49× | 0.524 | 0.960 | 1.83× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.090 | 0.041 | 0.46× | 0.887 | 0.661 | 0.74× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.041 | 0.267 | 6.45× | 0.248 | 0.474 | 1.91× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.059 | 0.305 | 5.15× | 0.480 | 0.844 | 1.76× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.033 | 0.177 | 5.34× | 0.231 | 0.404 | 1.75× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.035 | 0.009 | 0.27× | 0.253 | 0.127 | 0.50× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.160 | 0.423 | 2.64× | 1.331 | 1.572 | 1.18× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.037 | 0.258 | 7.03× | 0.248 | 0.488 | 1.97× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.091 | 0.550 | 6.06× | 0.684 | 1.379 | 2.02× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.055 | 0.338 | 6.11× | 0.373 | 0.997 | 2.67× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.087 | 0.345 | 3.98× | 0.704 | 0.852 | 1.21× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.100 | 0.023 | 0.23× | 0.830 | 0.233 | 0.28× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.045 | 0.009 | 0.21× | 0.256 | 0.069 | 0.27× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.095 | 0.049 | 0.51× | 0.666 | 0.511 | 0.77× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.111 | 0.023 | 0.21× | 0.882 | 0.241 | 0.27× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.077 | 0.032 | 0.41× | 0.700 | 0.399 | 0.57× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.549 | 9.288 | 16.92× | 5.767 | 79.902 | 13.85× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.666 | 9.289 | 13.95× | 6.117 | 81.349 | 13.30× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.576 | 8.718 | 15.14× | 5.597 | 81.900 | 14.63× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.962 | 9.698 | 10.08× | 8.972 | 87.369 | 9.74× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.589 | 9.532 | 16.18× | 5.639 | 84.238 | 14.94× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.567 | 8.727 | 15.38× | 5.957 | 82.513 | 13.85× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.018 | 0.004 | 0.23× | 0.021 | 0.005 | 0.23× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.034 | 0.023 | 0.68× | 0.278 | 0.295 | 1.06× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.136 | 0.801 | 5.90× | 0.199 | 1.090 | 5.49× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.027 | 0.020 | 0.75× | 0.280 | 0.136 | 0.49× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.058 | 0.012 | 0.21× | 0.404 | 0.072 | 0.18× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.015 | 0.004 | 0.26× | 0.031 | 0.007 | 0.24× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.018 | 0.004 | 0.23× | 0.042 | 0.011 | 0.27× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.894 | 0.023 | 0.03× | 7.531 | 0.169 | 0.02× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.077 | 0.012 | 0.16× | 0.087 | 0.008 | 0.10× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.014 | 0.004 | 0.31× | 0.017 | 0.006 | 0.36× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 1.185 | 0.579 | 0.49× | 11.807 | 6.001 | 0.51× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.135 | 0.098 | 0.73× | 0.192 | 0.143 | 0.74× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.005 | 0.004 | 0.81× | 0.005 | 0.004 | 0.86× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 1.238 | 0.022 | 0.02× | 13.144 | 0.140 | 0.01× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.012 | 0.005 | 0.42× | 0.013 | 0.008 | 0.61× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.010 | 0.006 | 0.57× | 0.013 | 0.009 | 0.72× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.273 | 0.051 | 0.19× | 2.060 | 0.451 | 0.22× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.076 | 0.018 | 0.24× | 0.668 | 0.064 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.090 | 0.014 | 0.15× | 0.777 | 0.103 | 0.13× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.018 | 0.004 | 0.23× | 0.023 | 0.006 | 0.26× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.020 | 0.006 | 0.33× | 0.024 | 0.009 | 0.36× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 1.134 | 0.009 | 0.01× | 11.272 | 0.056 | 0.00× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 4.161 | 0.104 | 0.02× | 3.914 | 0.092 | 0.02× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.968 | 0.544 | 0.56× | 11.700 | 7.109 | 0.61× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.381 | – | – | 3.927 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.225 | 2.597 | 11.56× | 2.192 | 14.682 | 6.70× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 1.180 | 0.877 | 0.74× | 15.029 | 10.647 | 0.71× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.218 | 2.418 | 11.11× | 2.199 | 13.878 | 6.31× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.205 | 2.608 | 12.73× | 2.070 | 14.321 | 6.92× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.239 | 2.427 | 10.13× | 2.122 | 14.151 | 6.67× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.048 | 0.713 | 14.79× | 0.392 | 4.715 | 12.02× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.024 | 0.093 | 3.80× | 0.107 | 0.095 | 0.89× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.427 | 0.492 | 1.15× | 5.504 | 5.670 | 1.03× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 433.2 | – | – | 3,798.1 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.670 | 0.267 | 0.40× | 0.653 | 0.269 | 0.41× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 2.086 | 2.895 | 1.39× | 26.292 | 29.585 | 1.13× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.085 | 0.049 | 0.57× | 0.603 | 0.369 | 0.61× | 0.0e+00 |
| `statistics.compare` | timed alone | 122.8 | – | – | 1,227.8 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.340 | 0.312 | 0.92× | 0.330 | 0.275 | 0.83× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 5.243 | – | – | 59.105 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.822 | 0.712 | 0.87× | 2.999 | 2.828 | 0.94× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 2.056 | 1.931 | 0.94× | 19.966 | 21.875 | 1.10× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.095 | 0.049 | 0.51× | 1.229 | 0.516 | 0.42× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 4.211 | 5.396 | 1.28× | 45.381 | 45.096 | 0.99× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 4.020 | 4.148 | 1.03× | 28.524 | 28.405 | 1.00× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.131 | 0.138 | 1.05× | 0.331 | 0.441 | 1.33× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 128.7 | – | – | 1,160.6 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.693 | 0.494 | 0.71× | 1.849 | 0.981 | 0.53× | 0.0e+00 |
| `statistics.proportion_ci` | statsmodels | 0.151 | 0.142 | 0.94× | 0.152 | 0.144 | 0.95× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.776 | – | – | 28.016 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.666 | 0.944 | 1.42× | 0.688 | 0.622 | 0.90× | 0.0e+00 |
| `statistics.t_test` | SciPy | 1.856 | 1.314 | 0.71× | 2.650 | 1.580 | 0.60× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 4.523 | 2.651 | 0.59× | 30.230 | 15.004 | 0.50× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.036 | 0.009 | 0.24× | 0.024 | 0.006 | 0.24× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.048 | 0.013 | 0.27× | 0.035 | 0.011 | 0.32× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 3.534 | 2.813 | 0.80× | 18.451 | 14.699 | 0.80× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.037 | 0.017 | 0.45× | 0.118 | 0.036 | 0.30× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.956 | 0.507 | 0.53× | 4.971 | 2.997 | 0.60× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 7.951 | 6.819 | 0.86× | 80.362 | 80.880 | 1.01× | 0.0e+00 |
| `text.bradley_terry` | choix | 2.151 | 43.353 | 20.16× | 4.206 | 190.6 | 45.31× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 16.909 | 16.164 | 0.96× | 160.2 | 167.8 | 1.05× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 22.674 | 37.548 | 1.66× | 255.2 | 406.6 | 1.59× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.416 | – | – | 2.680 | – | – | – |
| `text.citation_recall` | timed alone | 0.172 | – | – | 2.181 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.208 | 0.006 | 0.03× | 1.547 | 0.015 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.056 | 0.091 | 1.63× | 0.521 | 1.004 | 1.93× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.066 | 0.016 | 0.24× | 0.576 | 0.109 | 0.19× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.688 | 0.222 | 0.32× | 7.702 | 3.796 | 0.49× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.521 | 0.144 | 0.28× | 1.354 | 0.361 | 0.27× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.057 | 0.037 | 0.64× | 0.283 | 0.225 | 0.79× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.382 | – | – | 3.284 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.349 | 0.611 | 1.75× | 3.382 | 6.643 | 1.96× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.445 | 0.153 | 0.34× | 4.734 | 1.278 | 0.27× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.036 | 0.029 | 0.81× | 0.267 | 0.207 | 0.78× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.653 | 0.226 | 0.35× | 7.413 | 2.223 | 0.30× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.306 | 0.131 | 0.43× | 3.236 | 1.435 | 0.44× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.042 | 0.014 | 0.35× | 0.397 | 0.098 | 0.25× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.067 | – | – | 0.610 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 1.577 | 4.042 | 2.56× | 15.534 | 71.308 | 4.59× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.215 | 0.181 | 0.84× | 2.097 | 1.971 | 0.94× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.144 | 0.933 | 6.50× | 0.444 | 1.384 | 3.12× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.070 | 0.016 | 0.22× | 0.632 | 0.217 | 0.34× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.154 | 0.187 | 1.21× | 0.485 | 0.431 | 0.89× | 2.9e-15 |
| `text.mauve` | timed alone | 5.475 | – | – | 905.5 | – | – | – |
| `text.meteor` | NLTK | 5.509 | 6.311 | 1.15× | 56.508 | 65.745 | 1.16× | 0.0e+00 |
| `text.model_score` | timed alone | 0.021 | – | – | 0.233 | – | – | – |
| `text.moverscore` | POT | 20.025 | 2.165 | 0.11× | 222.0 | 21.926 | 0.10× | 1.1e-16 |
| `text.perplexity` | NumPy formula | 0.068 | 0.014 | 0.21× | 0.605 | 0.103 | 0.17× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.090 | 0.005 | 0.05× | 0.762 | 0.008 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.494 | – | – | 4.396 | – | – | – |
| `text.rouge_1` | rouge-score | 3.137 | 3.576 | 1.14× | 31.713 | 35.796 | 1.13× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 3.074 | 3.541 | 1.15× | 31.596 | 36.168 | 1.14× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 4.061 | 4.607 | 1.13× | 45.923 | 57.631 | 1.25× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 6.374 | 7.053 | 1.11× | 103.3 | 122.3 | 1.18× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.023 | 0.007 | 0.33× | 0.078 | 0.019 | 0.24× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 10.073 | 15.009 | 1.49× | 264.4 | 410.4 | 1.55× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.595 | – | – | 0.821 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 7.813 | 8.895 | 1.14× | 85.330 | 93.998 | 1.10× | 0.0e+00 |
| `text.spice` | Python stdlib | 1.405 | 0.122 | 0.09× | 14.407 | 0.773 | 0.05× | 0.0e+00 |
| `text.ter` | sacreBLEU | 39.890 | 23.642 | 0.59× | 368.5 | 239.2 | 0.65× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.349 | – | – | 5.150 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.404 | – | – | 5.551 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.305 | 0.032 | 0.10× | 4.094 | 0.266 | 0.06× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.787 | – | – | 2.625 | – | – | – |
| `text.win_rate` | NumPy formula | 0.059 | 0.008 | 0.14× | 0.412 | 0.016 | 0.04× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.556 | 0.535 | 0.96× | 5.821 | 5.533 | 0.95× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.188 | 0.141 | 0.75× | 0.929 | 0.801 | 0.86× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.077 | 0.039 | 0.50× | 0.984 | 0.821 | 0.83× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.361 | 1.790 | 4.96× | 1.514 | 4.232 | 2.80× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.081 | 0.037 | 0.46× | 0.956 | 0.774 | 0.81× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.073 | 0.026 | 0.35× | 0.912 | 0.682 | 0.75× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.050 | 0.025 | 0.49× | 0.772 | 0.692 | 0.90× | 0.0e+00 |
