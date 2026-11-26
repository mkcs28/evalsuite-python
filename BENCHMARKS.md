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

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.7.1, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 6.03× | 1.26×–29.59× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 41/42 | 6.04× | 0.81×–44.21× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 46/46 | 8.44× | 1.27×–48.62× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 14/14 | 3.08× | 1.04×–7.90× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 10.87× | 2.72×–43.97× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 171/172 | 6.64× | 0.81×–48.62× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.008 | 0.002 | 0.22× | 0.008 | 0.002 | 0.28× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.010 | 0.003 | 0.33× | 0.011 | 0.004 | 0.39× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 1.030 | 3.285 | 3.19× | 13.226 | 38.500 | 2.91× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.210 | – | – | 2.109 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.108 | 0.011 | 0.11× | 1.127 | 0.076 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.021 | 0.002 | 0.09× | 0.031 | 0.002 | 0.07× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.71× | 0.005 | 0.003 | 0.70× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.153 | – | – | 2.100 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.019 | 0.002 | 0.12× | 0.024 | 0.003 | 0.14× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.520 | 0.537 | 1.03× | 6.379 | 6.693 | 1.05× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.571 | 4.496 | 7.87× | 6.119 | 42.484 | 6.94× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 1.087 | 7.962 | 7.33× | 12.061 | 73.868 | 6.12× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.131 | 0.561 | 4.29× | 0.840 | 4.727 | 5.62× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.061 | 0.654 | 10.68× | 0.413 | 3.302 | 7.99× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.436 | 2.461 | 5.64× | 4.696 | 21.095 | 4.49× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.441 | 2.216 | 5.03× | 4.071 | 15.220 | 3.74× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.073 | 1.035 | 14.26× | 0.714 | 7.161 | 10.02× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.084 | 1.705 | 20.31× | 0.496 | 11.782 | 23.73× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.126 | 0.580 | 4.60× | 0.901 | 6.308 | 7.00× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.452 | 2.871 | 6.35× | 4.071 | 17.391 | 4.27× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.084 | 2.484 | 29.59× | 0.501 | 13.562 | 27.08× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.635 | 3.745 | 5.89× | 7.051 | 32.663 | 4.63× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.081 | 2.161 | 26.82× | 0.523 | 12.167 | 23.27× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.616 | 3.452 | 5.60× | 5.814 | 24.813 | 4.27× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.445 | 3.923 | 8.82× | 4.012 | 25.469 | 6.35× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.084 | 2.344 | 27.77× | 0.463 | 11.858 | 25.62× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.478 | 2.870 | 6.00× | 3.589 | 15.442 | 4.30× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.479 | 2.827 | 5.90× | 4.029 | 17.494 | 4.34× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 3.117 | 25.032 | 8.03× | 38.069 | 231.9 | 6.09× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.205 | 2.618 | 12.77× | 0.661 | 19.339 | 29.25× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.818 | 3.027 | 3.70× | 8.036 | 24.567 | 3.06× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.071 | 0.042 | 0.59× | 0.478 | 0.387 | 0.81× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.071 | 2.274 | 32.08× | 0.477 | 15.806 | 33.13× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.070 | 2.235 | 31.99× | 0.477 | 21.100 | 44.21× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.165 | 0.020 | 0.12× | 2.202 | 0.260 | 0.12× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.083 | 2.472 | 29.95× | 0.648 | 17.909 | 27.63× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.077 | 2.483 | 32.17× | 0.504 | 14.164 | 28.13× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.073 | 0.043 | 0.59× | 0.624 | 0.565 | 0.91× | 0.0e+00 |
| `code.code_complexity` | radon | 0.927 | 0.968 | 1.04× | 8.441 | 9.315 | 1.10× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 4.607 | 7.410 | 1.61× | 39.678 | 61.848 | 1.56× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.014 | 0.001 | 0.07× | 0.020 | 0.001 | 0.07× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.022 | 0.017 | 0.74× | 0.152 | 0.123 | 0.81× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.003 | 0.004 | 1.09× | 0.004 | 0.003 | 0.81× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.079 | 0.021 | 0.26× | 0.773 | 0.127 | 0.16× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.039 | 0.002 | 0.06× | 0.040 | 0.003 | 0.08× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.091 | 0.008 | 0.09× | 0.803 | 0.054 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.018 | 0.001 | 0.06× | 0.016 | 0.002 | 0.09× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 2.200 | 1.353 | 0.61× | 21.658 | 13.513 | 0.62× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.448 | 0.314 | 0.70× | 3.813 | 4.779 | 1.25× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 0.709 | 17.901 | 25.24× | 2.935 | 84.215 | 28.70× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.050 | 0.025 | 0.50× | 9.508 | 4.805 | 0.51× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 0.391 | 17.183 | 43.97× | 3.029 | 84.050 | 27.75× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.015 | 0.003 | 0.17× | 0.061 | 0.003 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.003 | 0.40× | 0.008 | 0.004 | 0.51× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.013 | 0.003 | 0.26× | 0.015 | 0.005 | 0.33× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.019 | 0.029 | 1.51× | 0.028 | 0.033 | 1.17× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.005 | 0.003 | 0.62× | 0.007 | 0.002 | 0.28× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.079 | 0.001 | 0.02× | 0.079 | 0.001 | 0.02× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.019 | 0.003 | 0.17× | 0.021 | 0.003 | 0.15× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.037 | 0.004 | 0.09× | 0.043 | 0.005 | 0.11× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.024 | 0.003 | 0.11× | 0.030 | 0.003 | 0.09× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.031 | 0.003 | 0.09× | 0.054 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.057 | 0.115 | 2.03× | 0.468 | 0.899 | 1.92× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.555 | 0.506 | 0.91× | 4.359 | 4.268 | 0.98× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.107 | 0.005 | 0.05× | 0.905 | 0.038 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.162 | 0.019 | 0.11× | 1.305 | 0.115 | 0.09× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.023 | 0.010 | 0.43× | 0.040 | 0.014 | 0.34× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.295 | 0.002 | 0.01× | 0.357 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.050 | 0.024 | 0.48× | 0.069 | 0.032 | 0.47× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.037 | 0.003 | 0.07× | 0.043 | 0.002 | 0.06× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.482 | 0.307 | 0.64× | 3.973 | 2.454 | 0.62× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.035 | 0.020 | 0.58× | 0.075 | 0.054 | 0.72× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.010 | 0.002 | 0.24× | 0.011 | 0.003 | 0.26× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.130 | 0.020 | 0.15× | 1.266 | 0.160 | 0.13× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.007 | 0.003 | 0.50× | 0.010 | 0.005 | 0.45× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.105 | 0.831 | 7.90× | 0.417 | 1.213 | 2.91× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.069 | 0.003 | 0.05× | 0.469 | 0.007 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.041 | 0.306 | 7.48× | 0.114 | 0.327 | 2.86× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.038 | 0.014 | 0.38× | 0.035 | 0.016 | 0.46× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.346 | 7.700 | 22.25× | 3.577 | 68.862 | 19.25× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 0.488 | 8.680 | 17.78× | 5.059 | 72.193 | 14.27× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.438 | 0.790 | 1.80× | 3.621 | 6.744 | 1.86× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.352 | 0.341 | 0.97× | 5.207 | 2.940 | 0.56× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.360 | 0.511 | 1.42× | 3.560 | 3.504 | 0.98× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.037 | – | – | 0.194 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.127 | 0.025 | 0.19× | 0.140 | 0.027 | 0.19× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.006 | 0.003 | 0.46× | 0.007 | 0.003 | 0.44× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.236 | – | – | 2.022 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.527 | 0.129 | 0.25× | 3.736 | 1.280 | 0.34× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.473 | 0.025 | 0.05× | 2.839 | 0.211 | 0.07× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.060 | 0.024 | 0.40× | 0.628 | 0.212 | 0.34× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.057 | 0.289 | 5.11× | 0.698 | 1.123 | 1.61× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.127 | 0.050 | 0.39× | 0.824 | 0.593 | 0.72× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.045 | 0.406 | 8.96× | 0.290 | 0.607 | 2.10× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.057 | 0.254 | 4.43× | 0.525 | 1.210 | 2.30× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.027 | 0.151 | 5.59× | 0.388 | 0.639 | 1.65× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.027 | 0.011 | 0.39× | 0.339 | 0.140 | 0.41× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.195 | 0.428 | 2.20× | 1.722 | 2.173 | 1.26× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.038 | 0.188 | 4.94× | 0.490 | 0.816 | 1.66× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.059 | 0.328 | 5.59× | 0.856 | 1.314 | 1.53× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.033 | 0.229 | 6.92× | 0.481 | 1.034 | 2.15× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.049 | 0.213 | 4.38× | 0.575 | 0.853 | 1.48× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.064 | 0.017 | 0.26× | 0.888 | 0.225 | 0.25× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.032 | 0.006 | 0.18× | 0.359 | 0.072 | 0.20× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.058 | 0.037 | 0.64× | 0.597 | 0.405 | 0.68× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.059 | 0.017 | 0.28× | 0.840 | 0.205 | 0.24× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.052 | 0.021 | 0.41× | 0.774 | 0.339 | 0.44× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.337 | 8.628 | 25.62× | 3.847 | 65.499 | 17.03× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.576 | 6.054 | 10.51× | 3.484 | 53.806 | 15.44× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.316 | 6.016 | 19.05× | 3.548 | 67.256 | 18.96× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.603 | 6.367 | 10.56× | 4.413 | 66.681 | 15.11× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.346 | 6.313 | 18.26× | 3.711 | 71.864 | 19.36× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.354 | 6.914 | 19.51× | 6.155 | 59.739 | 9.71× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.011 | 0.002 | 0.20× | 0.015 | 0.003 | 0.22× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.022 | 0.015 | 0.71× | 0.206 | 0.124 | 0.61× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.083 | 0.526 | 6.32× | 0.090 | 0.530 | 5.87× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.018 | 0.015 | 0.84× | 0.133 | 0.118 | 0.89× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.030 | 0.009 | 0.30× | 0.250 | 0.061 | 0.24× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.010 | 0.002 | 0.24× | 0.014 | 0.003 | 0.23× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.011 | 0.002 | 0.22× | 0.018 | 0.005 | 0.26× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.144 | 0.019 | 0.13× | 1.663 | 0.145 | 0.09× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.051 | 0.005 | 0.09× | 0.057 | 0.006 | 0.10× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.008 | 0.003 | 0.32× | 0.012 | 0.005 | 0.38× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.180 | 0.380 | 2.11× | 2.176 | 3.751 | 1.72× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.074 | 0.048 | 0.65× | 0.117 | 0.075 | 0.64× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.71× | 0.004 | 0.003 | 0.78× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.255 | 0.020 | 0.08× | 2.923 | 0.112 | 0.04× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.005 | 0.002 | 0.50× | 0.007 | 0.005 | 0.66× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.005 | 0.003 | 0.57× | 0.008 | 0.005 | 0.64× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.145 | 0.044 | 0.30× | 1.400 | 0.804 | 0.57× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.053 | 0.008 | 0.15× | 0.521 | 0.066 | 0.13× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.069 | 0.015 | 0.22× | 0.953 | 0.093 | 0.10× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.020 | 0.003 | 0.12× | 0.016 | 0.004 | 0.28× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.012 | 0.004 | 0.31× | 0.016 | 0.006 | 0.39× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.049 | 0.007 | 0.13× | 0.495 | 0.042 | 0.08× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.360 | 0.060 | 0.03× | 2.324 | 0.092 | 0.04× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.133 | 0.363 | 2.72× | 2.499 | 7.233 | 2.89× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.252 | – | – | 4.083 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.155 | 1.869 | 12.09× | 1.979 | 11.026 | 5.57× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.141 | 0.571 | 4.04× | 1.729 | 6.214 | 3.59× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.149 | 1.762 | 11.79× | 1.397 | 11.213 | 8.03× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.132 | 1.816 | 13.72× | 1.433 | 13.412 | 9.36× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.154 | 3.137 | 20.37× | 1.690 | 13.335 | 7.89× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.039 | 0.906 | 23.12× | 0.272 | 3.984 | 14.63× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.016 | 0.100 | 6.31× | 0.056 | 0.062 | 1.12× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.568 | 0.461 | 0.81× | 4.517 | 4.799 | 1.06× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 85.796 | – | – | 285.5 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.038 | 0.320 | 8.45× | 0.046 | 0.248 | 5.41× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.867 | 2.899 | 1.55× | 26.150 | 27.060 | 1.03× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.077 | 0.056 | 0.73× | 0.500 | 0.323 | 0.65× | 0.0e+00 |
| `statistics.compare` | timed alone | 53.394 | – | – | 263.9 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.051 | 0.367 | 7.14× | 0.018 | 0.233 | 12.76× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 1.725 | – | – | 15.387 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.138 | 0.750 | 5.43× | 0.287 | 2.126 | 7.40× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.727 | 2.516 | 3.46× | 7.274 | 15.234 | 2.09× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.150 | 0.062 | 0.41× | 0.884 | 0.321 | 0.36× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 1.291 | 3.634 | 2.81× | 14.638 | 38.009 | 2.60× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.681 | 3.176 | 4.66× | 7.625 | 22.902 | 3.00× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.021 | 0.116 | 5.54× | 0.158 | 0.522 | 3.31× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 41.789 | – | – | 256.0 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.081 | 0.297 | 3.68× | 0.397 | 0.889 | 2.24× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.013 | 0.088 | 6.95× | 0.008 | 0.075 | 9.95× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 0.810 | – | – | 8.720 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.041 | 0.469 | 11.50× | 0.041 | 0.432 | 10.52× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.098 | 0.531 | 5.41× | 0.535 | 1.322 | 2.47× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.323 | 1.482 | 4.59× | 4.868 | 11.840 | 2.43× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.018 | 0.003 | 0.16× | 0.015 | 0.004 | 0.24× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.022 | 0.004 | 0.18× | 0.020 | 0.010 | 0.49× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 1.905 | 1.342 | 0.70× | 12.178 | 9.063 | 0.74× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.013 | 0.008 | 0.57× | 0.086 | 0.032 | 0.37× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.298 | 0.170 | 0.57× | 2.960 | 1.740 | 0.59× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 0.713 | 4.926 | 6.91× | 6.950 | 57.100 | 8.22× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.646 | 37.006 | 22.48× | 2.407 | 117.1 | 48.62× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 1.223 | 11.450 | 9.36× | 11.220 | 128.5 | 11.45× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 2.929 | 23.222 | 7.93× | 29.310 | 234.8 | 8.01× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.139 | – | – | 1.699 | – | – | – |
| `text.citation_recall` | timed alone | 0.114 | – | – | 1.174 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.142 | 0.004 | 0.03× | 1.135 | 0.012 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.041 | 0.061 | 1.47× | 0.382 | 0.673 | 1.76× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.050 | 0.013 | 0.27× | 0.416 | 0.076 | 0.18× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.177 | 0.189 | 1.07× | 2.009 | 3.225 | 1.60× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.112 | 0.108 | 0.96× | 0.342 | 0.382 | 1.12× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.041 | 0.029 | 0.70× | 0.256 | 0.197 | 0.77× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.279 | – | – | 3.036 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.238 | 0.407 | 1.71× | 3.140 | 4.075 | 1.30× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.038 | 0.100 | 2.63× | 0.089 | 0.687 | 7.68× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.022 | 0.018 | 0.83× | 0.195 | 0.147 | 0.75× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.095 | 0.138 | 1.44× | 0.902 | 1.356 | 1.50× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.223 | 0.093 | 0.42× | 2.087 | 1.075 | 0.52× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.031 | 0.010 | 0.32× | 0.300 | 0.089 | 0.30× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.050 | – | – | 0.418 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 1.220 | 3.676 | 3.01× | 11.579 | 25.430 | 2.20× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.205 | 0.165 | 0.81× | 1.265 | 1.082 | 0.86× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.152 | 1.029 | 6.77× | 0.247 | 0.906 | 3.67× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.051 | 0.012 | 0.22× | 0.486 | 0.252 | 0.52× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.074 | 0.095 | 1.27× | 0.252 | 0.509 | 2.02× | 2.9e-15 |
| `text.mauve` | timed alone | 3.527 | – | – | 537.5 | – | – | – |
| `text.meteor` | NLTK | 0.955 | 6.227 | 6.52× | 8.094 | 49.337 | 6.10× | 0.0e+00 |
| `text.model_score` | timed alone | 0.019 | – | – | 0.072 | – | – | – |
| `text.moverscore` | POT | 1.279 | 1.845 | 1.44× | 8.608 | 12.329 | 1.43× | 0.0e+00 |
| `text.perplexity` | NumPy formula | 0.073 | 0.016 | 0.22× | 0.406 | 0.074 | 0.18× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.085 | 0.004 | 0.05× | 0.509 | 0.006 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.453 | – | – | 3.070 | – | – | – |
| `text.rouge_1` | rouge-score | 0.419 | 2.459 | 5.87× | 4.733 | 26.749 | 5.65× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.374 | 2.529 | 6.76× | 4.674 | 21.366 | 4.57× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 0.359 | 3.001 | 8.36× | 4.113 | 28.156 | 6.85× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 0.623 | 4.084 | 6.55× | 7.490 | 53.131 | 7.09× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.014 | 0.004 | 0.28× | 0.032 | 0.007 | 0.21× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 0.513 | 9.026 | 17.60× | 8.020 | 291.1 | 36.30× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.344 | – | – | 0.414 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 0.660 | 5.354 | 8.11× | 8.198 | 56.082 | 6.84× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.871 | 0.043 | 0.05× | 8.239 | 0.598 | 0.07× | 0.0e+00 |
| `text.ter` | sacreBLEU | 0.657 | 14.354 | 21.84× | 7.027 | 166.3 | 23.66× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.232 | – | – | 5.286 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.250 | – | – | 5.451 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.194 | 0.021 | 0.11× | 4.261 | 0.349 | 0.08× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.508 | – | – | 2.801 | – | – | – |
| `text.win_rate` | NumPy formula | 0.033 | 0.005 | 0.16× | 0.437 | 0.020 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.416 | 0.391 | 0.94× | 4.610 | 4.547 | 0.99× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.115 | 0.096 | 0.83× | 0.813 | 0.787 | 0.97× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.057 | 0.035 | 0.62× | 0.837 | 0.764 | 0.91× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.241 | 1.255 | 5.22× | 1.320 | 3.518 | 2.66× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.057 | 0.035 | 0.61× | 0.831 | 0.753 | 0.91× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.049 | 0.025 | 0.51× | 0.842 | 0.678 | 0.81× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.035 | 0.023 | 0.67× | 0.727 | 0.690 | 0.95× | 0.0e+00 |

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
