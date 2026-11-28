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
EvalSuite 1.1.2 benchmarks | Python 3.13.18 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64, 2 cores | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 12/12 | 6.13× | 1.18×–28.36× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 23/24 | 3.72× | 0.98×–16.87× |
| LLM evaluation | 6 | 18 | 18/18 | 18/18 | 4.23× | 1.15×–11.54× |
| LLM systems | 6 | 18 | 15/15 | 15/15 | 2.00× | 1.11×–6.12× |
| Segmentation and object detection | 3 | 9 | 9/9 | 9/9 | 5.69× | 2.07×–20.71× |
| Overall | 27 | 81 | 78/78 | 77/78 | 3.86× | 0.98×–28.36× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.118 | 1.824 | 15.45× | 0.02 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 0.809 | 4.154 | 5.13× | 0.77 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 8.571 | 35.614 | 4.16× | 7.63 | 21.80 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.039 | 0.045 | 1.15× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.157 | 0.369 | 2.35× | 0.29 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 1.590 | 3.190 | 2.01× | 2.53 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.300 | 8.502 | 28.36× | 0.03 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 1.335 | 22.534 | 16.88× | 1.69 | 3.44 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 9.198 | 171.991 | 18.70× | 16.28 | 34.34 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.113 | 2.068 | 18.23× | 0.02 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 5.220 | 16.654 | 3.19× | 1.53 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 84.029 | 211.633 | 2.52× | 15.26 | 76.30 | 1.1e-16 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.458 | 2.572 | 5.61× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 18.539 | 105.561 | 5.69× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 218.230 | 1216.973 | 5.58× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.120 | 0.527 | 4.38× | 0.02 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 0.689 | 1.522 | 2.21× | 0.76 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 5.533 | 6.486 | 1.17× | 7.63 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.223 | 3.761 | 16.87× | 0.02 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 2.777 | 14.151 | 5.10× | 0.76 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 19.540 | 76.356 | 3.91× | 7.63 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.129 | 0.820 | 6.36× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 11.806 | 16.349 | 1.38× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 169.534 | 196.838 | 1.16× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 1.209 | 19.525 | 16.15× | 0.95 | 1.33 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 4.664 | 90.798 | 19.47× | 1.13 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 40.157 | 831.811 | 20.71× | 2.83 | 34.43 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.044 | 0.053 | 1.20× | 0.04 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 4.970 | 4.848 | 0.98× | 3.82 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 77.409 | 95.951 | 1.24× | 38.15 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.298 | 0.985 | 3.30× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.039 | 1.741 | 1.68× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 9.671 | 11.401 | 1.18× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.227 | 1.654 | 7.29× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 10.476 | 78.902 | 7.53× | 0.31 | 3.11 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 106.735 | 777.157 | 7.28× | 2.99 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.199 | 2.548 | 12.83× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 3.224 | 8.123 | 2.52× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 34.928 | 72.239 | 2.07× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.157 | 0.385 | 2.46× | 0.01 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 3.512 | 8.452 | 2.41× | 0.01 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 7.621 | 18.543 | 2.43× | 0.01 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.018 | 0.254 | 14.35× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.017 | 0.238 | 13.63× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.016 | 0.254 | 15.93× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.092 | 0.568 | 6.15× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 7.427 | 28.622 | 3.85× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 127.457 | 385.463 | 3.02× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.091 | 0.575 | 6.30× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.592 | 1.409 | 2.38× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 5.378 | 8.093 | 1.50× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.138 | 0.330 | 2.38× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 13.256 | 30.912 | 2.33× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 133.204 | 295.472 | 2.22× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | – | 50.968 | – | – | 0.35 | – | – |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | – | 435.386 | – | – | 1.98 | – | – |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | – | 1.837 | – | – | 0.07 | – | – |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.280 | 1.506 | 5.37× | 0.06 | 0.08 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 12.981 | 28.028 | 2.16× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 181.774 | 242.589 | 1.33× | 54.36 | 76.30 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.098 | 0.599 | 6.12× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.598 | 1.446 | 2.42× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 6.279 | 7.546 | 1.20× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.111 | 0.281 | 2.53× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 13.248 | 28.419 | 2.15× | 0.70 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 127.595 | 284.958 | 2.23× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.034 | 0.041 | 1.21× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 0.605 | 2.008 | 3.32× | 0.76 | 0.77 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 8.978 | 15.334 | 1.71× | 7.63 | 7.63 | 0.0e+00 |
| systems: maintainability index (100 programs) | 100,000 | radon | 19.506 | 21.954 | 1.13× | 0.20 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 118.315 | 131.866 | 1.11× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.429 | 0.478 | 1.11× | 0.04 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.156 | 0.854 | 5.49× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 13.020 | 51.458 | 3.95× | 0.58 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 92.439 | 493.795 | 5.34× | 5.82 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.193 | 0.746 | 3.86× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 15.901 | 86.208 | 5.42× | 0.63 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 219.540 | 888.978 | 4.05× | 6.95 | 6.55 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 0.268 | 1.872 | 6.98× | 0.01 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 19.938 | 230.161 | 11.54× | 1.13 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 266.679 | 2399.786 | 9.00× | 11.44 | 211.51 | 0.0e+00 |

CodeBLEU's n-gram terms are timed alone in this run (the `codebleu` reference library needs tree-sitter grammars that were not installed).

## Every metric

One row for each of the 238 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 2.0.0, Python 3.13.18, NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone. The two `multilabel.*` rows time the whole `multilabel_report` (about 30 metrics and per-label curves in one call) against a single scikit-learn function.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 5.93× | 1.19×–28.08× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 41/42 | 5.77× | 0.99×–30.94× |
| Forecasting, multi-label, comparisons and data quality | 13 | 8 | 2 | 3 | 10/10 | 6/16 | 1.32× | 0.06×–29.56× |
| LLM evaluation | 70 | 20 | 34 | 16 | 54/54 | 40/40 | 10.09× | 1.07×–47.98× |
| LLM systems | 77 | 6 | 68 | 3 | 74/74 | 12/12 | 3.30× | 1.08×–7.85× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 9.96× | 2.20×–43.65× |
| **All metrics** | 238 | 90 | 120 | 28 | 210/210 | 169/180 | 5.88× | 0.06×–47.98× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.009 | 0.003 | 0.30× | 0.006 | 0.002 | 0.31× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.016 | 0.004 | 0.27× | 0.009 | 0.003 | 0.39× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 1.355 | 2.691 | 1.99× | 10.923 | 31.234 | 2.86× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.183 | – | – | 1.731 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.093 | 0.010 | 0.10× | 0.915 | 0.064 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.017 | 0.002 | 0.09× | 0.026 | 0.002 | 0.07× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.75× | 0.003 | 0.003 | 0.80× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.134 | – | – | 1.731 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.015 | 0.002 | 0.13× | 0.023 | 0.003 | 0.13× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.444 | 0.448 | 1.01× | 5.415 | 5.562 | 1.03× | 0.0e+00 |
| `bayes.correlated_ttest` | baycomp | 0.439 | 0.076 | 0.17× | 0.453 | 0.075 | 0.17× | 1.1e-16 |
| `calibration.calibration_intercept` | statsmodels | 0.463 | 3.351 | 7.23× | 4.319 | 28.447 | 6.59× | 1.1e-16 |
| `calibration.calibration_slope` | statsmodels | 0.789 | 5.869 | 7.44× | 8.401 | 51.149 | 6.09× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.117 | 0.461 | 3.94× | 0.699 | 4.709 | 6.74× | 5.6e-16 |
| `classification.accuracy` | scikit-learn | 0.056 | 0.526 | 9.45× | 0.415 | 3.385 | 8.15× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.352 | 1.845 | 5.24× | 4.742 | 17.400 | 3.67× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.345 | 1.722 | 5.00× | 3.443 | 12.843 | 3.73× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.058 | 0.825 | 14.13× | 0.592 | 5.884 | 9.93× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.068 | 1.385 | 20.22× | 0.422 | 9.975 | 23.66× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.103 | 0.481 | 4.66× | 0.744 | 5.027 | 6.75× | 6.4e-16 |
| `classification.f1` | scikit-learn | 0.358 | 2.267 | 6.33× | 3.392 | 14.543 | 4.29× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.075 | 1.869 | 25.08× | 0.459 | 11.087 | 24.16× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.467 | 3.083 | 6.60× | 6.144 | 27.627 | 4.50× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.070 | 1.741 | 24.82× | 0.427 | 11.101 | 25.97× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.426 | 2.597 | 6.10× | 5.589 | 23.017 | 4.12× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.360 | 3.177 | 8.82× | 3.368 | 23.185 | 6.88× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.076 | 1.892 | 24.99× | 0.444 | 11.151 | 25.14× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.360 | 2.240 | 6.22× | 3.385 | 14.360 | 4.24× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.349 | 2.273 | 6.52× | 3.361 | 14.163 | 4.21× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 2.533 | 20.038 | 7.91× | 58.962 | 201.0 | 3.41× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.067 | 1.885 | 28.08× | 0.500 | 11.549 | 23.12× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.592 | 2.300 | 3.89× | 7.769 | 20.266 | 2.61× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.060 | 0.035 | 0.58× | 0.482 | 0.371 | 0.77× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.058 | 1.804 | 30.94× | 0.482 | 13.283 | 27.56× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.059 | 1.797 | 30.71× | 0.476 | 13.233 | 27.80× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.143 | 0.017 | 0.12× | 1.595 | 0.165 | 0.10× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.068 | 1.882 | 27.72× | 0.508 | 11.546 | 22.74× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.068 | 1.868 | 27.64× | 0.519 | 11.669 | 22.49× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.059 | 0.035 | 0.60× | 0.467 | 0.364 | 0.78× | 0.0e+00 |
| `code.code_complexity` | radon | 0.724 | 0.799 | 1.10× | 7.065 | 7.605 | 1.08× | 0.0e+00 |
| `code.codebleu` | timed alone | 3.369 | – | – | 31.578 | – | – | – |
| `code.coverage_rate` | NumPy formula | 0.013 | 0.001 | 0.07× | 0.016 | 0.001 | 0.07× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.019 | 0.015 | 0.78× | 0.135 | 0.108 | 0.80× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.003 | 0.002 | 0.77× | 0.003 | 0.003 | 0.83× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.069 | 0.013 | 0.18× | 0.697 | 0.112 | 0.16× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.024 | 0.002 | 0.08× | 0.034 | 0.003 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.080 | 0.007 | 0.09× | 0.686 | 0.046 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.010 | 0.001 | 0.10× | 0.014 | 0.001 | 0.09× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 1.743 | 1.064 | 0.61× | 17.922 | 11.024 | 0.62× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.337 | 0.268 | 0.79× | 3.436 | 2.725 | 0.79× | 0.0e+00 |
| `comparisons.nemenyi` | scikit-posthocs | 0.565 | 0.551 | 0.97× | 0.582 | 0.540 | 0.93× | 0.0e+00 |
| `conformal.classification_sets` | timed alone | 0.301 | – | – | 3.366 | – | – | – |
| `data_quality.label_issues` | cleanlab | 0.526 | 1.735 | 3.30× | 0.981 | 3.108 | 3.17× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 0.311 | 13.575 | 43.65× | 4.406 | 71.536 | 16.24× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.043 | 0.021 | 0.49× | 8.151 | 4.700 | 0.58× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 0.317 | 13.646 | 43.07× | 2.643 | 69.149 | 26.16× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.013 | 0.002 | 0.17× | 0.058 | 0.003 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.33× | 0.007 | 0.004 | 0.50× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.28× | 0.014 | 0.004 | 0.32× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.016 | 0.024 | 1.47× | 0.023 | 0.029 | 1.26× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.004 | 0.002 | 0.42× | 0.007 | 0.002 | 0.28× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.067 | 0.001 | 0.02× | 0.086 | 0.001 | 0.02× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.016 | 0.003 | 0.16× | 0.021 | 0.003 | 0.15× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.030 | 0.003 | 0.10× | 0.047 | 0.005 | 0.10× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.021 | 0.002 | 0.10× | 0.030 | 0.003 | 0.09× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.026 | 0.002 | 0.08× | 0.051 | 0.003 | 0.05× | 0.0e+00 |
| `forecasting.crps` | properscoring | 1.692 | 1.161 | 0.69× | 21.584 | 13.458 | 0.62× | 5.6e-17 |
| `forecasting.diebold_mariano` | timed alone | 0.358 | – | – | 1.282 | – | – | – |
| `forecasting.mase` | NumPy formula | 0.030 | 0.017 | 0.56× | 0.488 | 0.305 | 0.63× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.050 | 0.094 | 1.90× | 0.428 | 0.964 | 2.25× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.439 | 0.406 | 0.92× | 4.452 | 4.407 | 0.99× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.086 | 0.004 | 0.05× | 1.217 | 0.076 | 0.06× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.131 | 0.015 | 0.11× | 2.359 | 0.121 | 0.05× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.017 | 0.008 | 0.49× | 0.043 | 0.015 | 0.35× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.241 | 0.003 | 0.01× | 0.382 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.042 | 0.021 | 0.50× | 0.076 | 0.032 | 0.42× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.030 | 0.002 | 0.07× | 0.051 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.406 | 0.264 | 0.65× | 4.163 | 2.628 | 0.63× | 0.0e+00 |
| `method_agreement.bland_altman` | NumPy formula | 0.506 | 0.008 | 0.02× | 1.389 | 0.080 | 0.06× | 0.0e+00 |
| `method_agreement.icc` | pingouin | 0.365 | 10.148 | 27.82× | 0.788 | 23.288 | 29.56× | 1.1e-16 |
| `multilabel.coverage_error` | scikit-learn | 6.459 | 0.547 | 0.08× | 61.149 | 3.619 | 0.06× | 0.0e+00 |
| `multilabel.label_ranking_average_precision` | scikit-learn | 6.613 | 101.9 | 15.41× | 62.556 | 1,047.6 | 16.75× | 8.4e-15 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.031 | 0.015 | 0.49× | 0.066 | 0.045 | 0.69× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.009 | 0.002 | 0.28× | 0.011 | 0.003 | 0.26× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.108 | 0.017 | 0.15× | 1.237 | 0.136 | 0.11× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.007 | 0.003 | 0.42× | 0.008 | 0.004 | 0.51× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.089 | 0.700 | 7.85× | 0.366 | 1.037 | 2.83× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.057 | 0.003 | 0.05× | 0.407 | 0.005 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.034 | 0.260 | 7.75× | 0.103 | 0.276 | 2.68× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.030 | 0.012 | 0.40× | 0.030 | 0.015 | 0.50× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.277 | 6.295 | 22.75× | 3.323 | 60.861 | 18.32× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 0.393 | 8.871 | 22.60× | 4.431 | 66.959 | 15.11× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.623 | 0.647 | 1.04× | 3.863 | 4.528 | 1.17× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.475 | 0.303 | 0.64× | 2.893 | 2.637 | 0.91× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.292 | 0.431 | 1.48× | 2.951 | 2.623 | 0.89× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.035 | – | – | 0.162 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.226 | 0.029 | 0.13× | 0.127 | 0.025 | 0.19× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.005 | 0.002 | 0.39× | 0.006 | 0.003 | 0.44× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.195 | – | – | 1.756 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.257 | 0.113 | 0.44× | 3.181 | 1.122 | 0.35× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.242 | 0.021 | 0.09× | 2.337 | 0.182 | 0.08× | 0.0e+00 |
| `recalibration.isotonic` | scikit-learn | 1.957 | 1.818 | 0.93× | 24.354 | 17.919 | 0.74× | 2.2e-16 |
| `recalibration.platt` | timed alone | 2.941 | – | – | 32.463 | – | – | – |
| `regression.adjusted_r2` | NumPy formula | 0.046 | 0.015 | 0.33× | 0.597 | 0.177 | 0.30× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.046 | 0.235 | 5.13× | 0.531 | 1.110 | 2.09× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.057 | 0.031 | 0.54× | 0.866 | 0.608 | 0.70× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.021 | 0.157 | 7.45× | 0.365 | 0.648 | 1.78× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.036 | 0.178 | 5.01× | 0.578 | 0.821 | 1.42× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.019 | 0.112 | 5.80× | 0.332 | 0.557 | 1.68× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.019 | 0.006 | 0.30× | 0.286 | 0.119 | 0.42× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.122 | 0.252 | 2.07× | 1.188 | 1.419 | 1.19× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.025 | 0.158 | 6.21× | 0.390 | 0.608 | 1.56× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.054 | 0.289 | 5.35× | 0.573 | 1.295 | 2.26× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.030 | 0.210 | 6.97× | 0.545 | 1.020 | 1.87× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.043 | 0.198 | 4.57× | 0.655 | 0.863 | 1.32× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.058 | 0.015 | 0.26× | 0.872 | 0.236 | 0.27× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.029 | 0.006 | 0.19× | 0.384 | 0.038 | 0.10× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.055 | 0.036 | 0.65× | 0.582 | 0.368 | 0.63× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.052 | 0.015 | 0.29× | 0.790 | 0.187 | 0.24× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.047 | 0.020 | 0.43× | 0.823 | 0.373 | 0.45× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.218 | 5.818 | 26.64× | 2.334 | 56.034 | 24.01× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.249 | 5.913 | 23.73× | 2.468 | 53.925 | 21.85× | 0.0e+00 |
| `retrieval.mrr` | ranx | 0.229 | 5.489 | 23.99× | 2.439 | 55.609 | 22.80× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.239 | 5.539 | 23.16× | 2.586 | 55.756 | 21.56× | 0.0e+00 |
| `retrieval.precision_at_k` | ranx | 0.219 | 5.489 | 25.06× | 2.391 | 55.695 | 23.29× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.220 | 7.165 | 32.49× | 2.304 | 56.499 | 24.52× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.013 | 0.002 | 0.16× | 0.012 | 0.003 | 0.21× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.022 | 0.017 | 0.79× | 0.169 | 0.107 | 0.63× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.111 | 0.770 | 6.92× | 0.077 | 0.460 | 5.96× | 0.0e+00 |
| `robustness.error_rate` | NumPy formula | 0.022 | 0.014 | 0.61× | 0.113 | 0.092 | 0.82× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.041 | 0.008 | 0.19× | 0.200 | 0.049 | 0.25× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.009 | 0.002 | 0.24× | 0.011 | 0.003 | 0.24× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.010 | 0.002 | 0.23× | 0.016 | 0.004 | 0.26× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.177 | 0.021 | 0.12× | 1.424 | 0.120 | 0.08× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.057 | 0.005 | 0.09× | 0.051 | 0.005 | 0.09× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.007 | 0.002 | 0.31× | 0.009 | 0.004 | 0.39× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.236 | 0.502 | 2.12× | 1.917 | 3.410 | 1.78× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.102 | 0.072 | 0.70× | 0.111 | 0.072 | 0.65× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.004 | 0.003 | 0.81× | 0.003 | 0.003 | 0.81× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.315 | 0.022 | 0.07× | 3.014 | 0.104 | 0.03× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.005 | 0.003 | 0.74× | 0.007 | 0.004 | 0.63× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.006 | 0.004 | 0.62× | 0.007 | 0.004 | 0.66× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.182 | 0.058 | 0.32× | 1.202 | 0.340 | 0.28× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.064 | 0.009 | 0.15× | 0.425 | 0.043 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.063 | 0.013 | 0.20× | 0.535 | 0.074 | 0.14× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.015 | 0.004 | 0.25× | 0.014 | 0.004 | 0.27× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.016 | 0.005 | 0.32× | 0.014 | 0.005 | 0.38× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.056 | 0.007 | 0.13× | 0.413 | 0.038 | 0.09× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 3.123 | 0.055 | 0.02× | 2.071 | 0.049 | 0.02× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.144 | 0.322 | 2.24× | 1.831 | 4.025 | 2.20× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.219 | – | – | 2.370 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.139 | 1.656 | 11.90× | 1.541 | 11.716 | 7.60× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.151 | 0.502 | 3.33× | 2.036 | 6.293 | 3.09× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.136 | 1.553 | 11.44× | 1.499 | 11.582 | 7.73× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.123 | 1.646 | 13.36× | 1.438 | 11.604 | 8.07× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.144 | 1.577 | 10.98× | 1.451 | 11.556 | 7.96× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.022 | 0.470 | 20.94× | 0.236 | 3.424 | 14.50× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.010 | 0.054 | 5.49× | 0.043 | 0.056 | 1.28× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.319 | 0.316 | 0.99× | 3.811 | 4.234 | 1.11× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 64.831 | – | – | 279.3 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.022 | 0.182 | 8.10× | 0.023 | 0.172 | 7.36× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.586 | 2.053 | 1.29× | 21.518 | 24.224 | 1.13× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.041 | 0.029 | 0.70× | 0.630 | 0.317 | 0.50× | 0.0e+00 |
| `statistics.compare` | timed alone | 34.358 | – | – | 278.4 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.016 | 0.188 | 11.89× | 0.016 | 0.191 | 12.18× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 1.548 | – | – | 17.322 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.092 | 0.482 | 5.26× | 0.242 | 1.981 | 8.19× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.479 | 1.396 | 2.92× | 7.156 | 14.098 | 1.97× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.058 | 0.029 | 0.50× | 1.005 | 0.326 | 0.32× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.816 | 2.837 | 3.47× | 14.329 | 38.069 | 2.66× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.512 | 1.979 | 3.87× | 8.601 | 24.384 | 2.83× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.014 | 0.055 | 4.04× | 0.149 | 0.427 | 2.86× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 32.759 | – | – | 237.1 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.075 | 0.268 | 3.58× | 0.286 | 0.675 | 2.36× | 8.9e-16 |
| `statistics.proportion_ci` | statsmodels | 0.013 | 0.086 | 6.84× | 0.015 | 0.090 | 5.89× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 0.762 | – | – | 8.017 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.046 | 0.432 | 9.33× | 0.047 | 0.441 | 9.39× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.092 | 0.463 | 5.03× | 0.389 | 1.049 | 2.70× | 8.8e-47 |
| `statistics.wilcoxon_test` | SciPy | 0.308 | 1.214 | 3.94× | 3.217 | 11.137 | 3.46× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.011 | 0.002 | 0.22× | 0.015 | 0.003 | 0.21× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.014 | 0.004 | 0.27× | 0.018 | 0.006 | 0.35× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 0.986 | 0.751 | 0.76× | 10.156 | 7.744 | 0.76× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.011 | 0.005 | 0.49× | 0.072 | 0.026 | 0.36× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.249 | 0.139 | 0.56× | 2.398 | 1.495 | 0.62× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 0.474 | 3.946 | 8.33× | 4.974 | 44.646 | 8.98× | 0.0e+00 |
| `text.bradley_terry` | timed alone | 1.061 | – | – | 3.565 | – | – | – |
| `text.chrf` | sacreBLEU | 1.302 | 9.635 | 7.40× | 9.078 | 111.8 | 12.32× | 1.4e-14 |
| `text.cider` | timed alone | 2.443 | – | – | 23.242 | – | – | – |
| `text.citation_precision` | timed alone | 0.121 | – | – | 1.546 | – | – | – |
| `text.citation_recall` | timed alone | 0.118 | – | – | 1.221 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.135 | 0.003 | 0.02× | 0.941 | 0.010 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.034 | 0.052 | 1.53× | 0.318 | 0.548 | 1.72× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.041 | 0.011 | 0.27× | 0.348 | 0.065 | 0.19× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.153 | 0.155 | 1.01× | 1.707 | 3.214 | 1.88× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.089 | 0.080 | 0.89× | 0.221 | 0.199 | 0.90× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.035 | 0.023 | 0.66× | 0.194 | 0.150 | 0.78× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.231 | – | – | 2.054 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.190 | 0.338 | 1.78× | 1.832 | 3.466 | 1.89× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.033 | 0.083 | 2.53× | 0.086 | 0.647 | 7.54× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.019 | 0.016 | 0.83× | 0.148 | 0.116 | 0.79× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.076 | 0.116 | 1.53× | 0.771 | 1.161 | 1.51× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.177 | 0.079 | 0.45× | 1.765 | 0.893 | 0.51× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.026 | 0.009 | 0.36× | 0.247 | 0.073 | 0.30× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.044 | – | – | 0.386 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 0.954 | 2.293 | 2.40× | 9.460 | 22.367 | 2.36× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.121 | 0.098 | 0.81× | 1.150 | 0.935 | 0.81× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.086 | 0.535 | 6.22× | 0.221 | 0.823 | 3.72× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.042 | 0.011 | 0.27× | 0.405 | 0.156 | 0.39× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.073 | 0.078 | 1.07× | 0.158 | 0.327 | 2.06× | 3.2e-15 |
| `text.mauve` | timed alone | 2.883 | – | – | 463.3 | – | – | – |
| `text.meteor` | NLTK | 0.566 | 3.870 | 6.84× | 6.355 | 41.523 | 6.53× | 0.0e+00 |
| `text.model_score` | timed alone | 0.012 | – | – | 0.069 | – | – | – |
| `text.moverscore` | timed alone | 3.728 | – | – | 50.479 | – | – | – |
| `text.perplexity` | NumPy formula | 0.043 | 0.011 | 0.25× | 0.351 | 0.064 | 0.18× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.052 | 0.003 | 0.06× | 0.419 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.262 | – | – | 2.628 | – | – | – |
| `text.rouge_1` | rouge-score | 0.311 | 2.304 | 7.40× | 3.731 | 22.762 | 6.10× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.320 | 2.334 | 7.28× | 3.605 | 22.977 | 6.37× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 0.351 | 2.783 | 7.93× | 5.577 | 29.480 | 5.29× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 0.577 | 4.133 | 7.16× | 6.125 | 44.194 | 7.22× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.014 | 0.006 | 0.39× | 0.030 | 0.008 | 0.25× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 0.344 | 9.247 | 26.89× | 5.268 | 252.8 | 47.98× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.384 | – | – | 0.428 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 0.572 | 5.427 | 9.50× | 6.358 | 57.355 | 9.02× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.984 | 0.046 | 0.05× | 8.635 | 0.508 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 0.807 | 13.616 | 16.88× | 6.256 | 146.1 | 23.36× | 0.0e+00 |
| `text.tool_argument_accuracy` | timed alone | 0.209 | – | – | 2.758 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.233 | – | – | 3.134 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.176 | 0.019 | 0.11× | 2.340 | 0.152 | 0.06× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.434 | – | – | 1.361 | – | – | – |
| `text.win_rate` | NumPy formula | 0.030 | 0.005 | 0.16× | 0.234 | 0.011 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.369 | 0.353 | 0.95× | 3.881 | 3.681 | 0.95× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.107 | 0.089 | 0.83× | 0.676 | 0.662 | 0.98× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.049 | 0.032 | 0.65× | 0.969 | 0.618 | 0.64× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.219 | 1.084 | 4.96× | 1.109 | 2.961 | 2.67× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.058 | 0.033 | 0.57× | 0.687 | 0.612 | 0.89× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.059 | 0.023 | 0.39× | 0.710 | 0.603 | 0.85× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.036 | 0.021 | 0.60× | 0.602 | 0.580 | 0.96× | 0.0e+00 |

## Rust crate (0.8.5)

`cargo bench -p evalsuite` (Criterion, medians) against the Python package with its compiled core on the
same workloads and machine (2 vCPUs, Linux x86_64, Python 3.13, NumPy 2.5). The values agree within the parity
tolerance; only the time differs.

| Workload | Rust | Python | Speed-up |
| --- | ---: | ---: | ---: |
| Accuracy, 100,000 / 1,000,000 labels | 0.033 / 0.73 ms | 0.69 / 7.9 ms | 21× / 10.8× |
| Macro F1, 5 classes, 100,000 / 1,000,000 | 1.1 / 13.5 ms | 1.5 / 9.9 ms | 1.3× / 0.7× |
| ROC AUC, 100,000 / 1,000,000 | 6.9 / 121 ms | 9.4 / 125 ms | 1.4× / 1.0× |
| RMSE, 1,000,000 | 4.3 ms | 6.0 ms | 1.4× |
| R², 1,000,000 | 8.3 ms | 9.5 ms | 1.2× |
| Median absolute error, 1,000,000 | 8.5 ms | 17.5 ms | 2.1× |
| BCa bootstrap of MAE, 10,000 obs × 2,000 resamples | 100 ms | 362 ms | 3.6× |
| Mann–Whitney U, 10,000 | 0.68 ms | 0.83 ms | 1.2× |
| Welch t-test, 10,000 | 0.064 ms | 0.105 ms | 1.6× |
| BLEU, 1,000 sentences | 11.6 ms | 7.7 ms | 0.7× |
| chrF, 1,000 sentences | 16.0 ms | 11.4 ms | 0.7× |
| TER, 1,000 sentences | 205 ms | 128 ms | 0.6× |
| ROUGE-L, 1,000 sentences | 5.2 ms | 5.0 ms | 1.0× |
| NDCG@10, 10,000 queries | 4.7 ms | 71 ms | 15.3× |
| MRR, 10,000 queries | 4.1 ms | 42 ms | 10.4× |
| Dice, 512 × 512 | 2.2 ms | 4.5 ms | 2.0× |
| HD95, 512 × 512 | 3.5 ms | 2.1 ms | 0.6× |

The Python retrieval metrics now run through the same Rust code with bit-identical results: NDCG@10 on 10,000
queries 112 ms → 32 ms, MRR 84 ms → 25 ms, Precision@k 74 ms → 23 ms.
