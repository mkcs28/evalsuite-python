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

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 1.2.0, Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 4.06× | 1.10×–18.62× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 42/42 | 5.12× | 1.01×–17.71× |
| LLM evaluation | 70 | 20 | 34 | 16 | 54/54 | 40/40 | 9.26× | 1.39×–34.59× |
| LLM systems | 77 | 6 | 68 | 3 | 74/74 | 12/12 | 3.34× | 1.05×–8.98× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 6.82× | 2.23×–47.99× |
| **All metrics** | 225 | 82 | 118 | 25 | 200/200 | 164/164 | 5.50× | 1.01×–47.99× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.013 | 0.004 | 0.28× | 0.011 | 0.004 | 0.38× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.013 | 0.007 | 0.53× | 0.014 | 0.007 | 0.49× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 2.040 | 5.266 | 2.58× | 24.247 | 44.130 | 1.82× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.485 | – | – | 2.397 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.237 | 0.021 | 0.09× | 1.292 | 0.078 | 0.06× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.029 | 0.004 | 0.14× | 0.023 | 0.002 | 0.10× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.006 | 0.005 | 0.77× | 0.004 | 0.003 | 0.78× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.335 | – | – | 2.176 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.020 | 0.004 | 0.20× | 0.018 | 0.004 | 0.20× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.656 | 0.618 | 0.94× | 6.801 | 7.674 | 1.13× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.589 | 5.667 | 9.63× | 5.340 | 71.120 | 13.32× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 1.269 | 10.777 | 8.49× | 8.876 | 82.096 | 9.25× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.201 | 0.179 | 0.89× | 1.300 | 2.414 | 1.86× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.097 | 0.398 | 4.10× | 0.560 | 1.130 | 2.02× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.509 | 1.778 | 3.49× | 7.162 | 17.901 | 2.50× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.456 | 1.323 | 2.90× | 5.600 | 7.227 | 1.29× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.072 | 0.755 | 10.48× | 0.561 | 3.480 | 6.21× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.087 | 0.819 | 9.43× | 0.451 | 2.041 | 4.53× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.129 | 0.205 | 1.59× | 0.794 | 1.742 | 2.19× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.473 | 1.945 | 4.12× | 4.232 | 7.693 | 1.82× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.089 | 1.547 | 17.33× | 0.445 | 3.461 | 7.77× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.556 | 2.429 | 4.37× | 6.472 | 18.274 | 2.82× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.092 | 1.411 | 15.37× | 0.508 | 3.291 | 6.49× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.495 | 3.886 | 7.86× | 5.527 | 18.143 | 3.28× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.713 | 3.232 | 4.53× | 5.585 | 10.002 | 1.79× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.117 | 2.182 | 18.62× | 0.534 | 4.303 | 8.07× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.711 | 2.903 | 4.09× | 4.347 | 11.246 | 2.59× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.681 | 2.880 | 4.23× | 4.441 | 10.546 | 2.37× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 6.820 | 26.202 | 3.84× | 72.321 | 174.2 | 2.41× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.125 | 2.230 | 17.80× | 0.448 | 3.499 | 7.81× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.884 | 3.522 | 3.98× | 5.978 | 22.238 | 3.72× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.102 | 0.072 | 0.70× | 0.433 | 0.367 | 0.85× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.104 | 1.573 | 15.06× | 0.432 | 2.665 | 6.17× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.098 | 1.584 | 16.19× | 0.433 | 2.582 | 5.97× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.230 | 0.035 | 0.15× | 1.471 | 0.176 | 0.12× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.121 | 2.143 | 17.71× | 0.496 | 3.351 | 6.76× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.123 | 1.483 | 12.10× | 0.440 | 3.329 | 7.56× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.074 | 0.048 | 0.65× | 0.436 | 0.374 | 0.86× | 0.0e+00 |
| `code.code_complexity` | radon | 1.043 | 1.093 | 1.05× | 9.821 | 10.900 | 1.11× | 0.0e+00 |
| `code.codebleu` | timed alone | 4.888 | – | – | 47.315 | – | – | – |
| `code.coverage_rate` | NumPy formula | 0.012 | 0.001 | 0.08× | 0.016 | 0.001 | 0.09× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.023 | 0.018 | 0.80× | 0.177 | 0.135 | 0.77× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.003 | 0.003 | 0.76× | 0.004 | 0.003 | 0.82× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.085 | 0.016 | 0.19× | 0.992 | 0.140 | 0.14× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.029 | 0.002 | 0.08× | 0.041 | 0.004 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.099 | 0.008 | 0.08× | 0.997 | 0.055 | 0.06× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.010 | 0.001 | 0.10× | 0.013 | 0.002 | 0.12× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 2.382 | 1.482 | 0.62× | 31.188 | 17.153 | 0.55× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.428 | 0.313 | 0.73× | 4.112 | 3.233 | 0.79× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 0.352 | 16.895 | 47.99× | 4.518 | 117.9 | 26.10× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.045 | 0.023 | 0.51× | 8.421 | 5.501 | 0.65× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 0.380 | 15.914 | 41.91× | 3.405 | 105.1 | 30.88× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.015 | 0.003 | 0.16× | 0.075 | 0.003 | 0.04× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.006 | 0.003 | 0.47× | 0.013 | 0.007 | 0.58× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.32× | 0.017 | 0.007 | 0.44× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.018 | 0.028 | 1.61× | 0.035 | 0.040 | 1.14× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.005 | 0.002 | 0.39× | 0.009 | 0.004 | 0.40× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.073 | 0.001 | 0.02× | 0.099 | 0.003 | 0.03× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.016 | 0.003 | 0.17× | 0.037 | 0.004 | 0.11× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.029 | 0.003 | 0.11× | 0.054 | 0.006 | 0.11× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.023 | 0.002 | 0.10× | 0.038 | 0.004 | 0.10× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.029 | 0.002 | 0.08× | 0.058 | 0.005 | 0.09× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.066 | 0.112 | 1.69× | 0.681 | 1.247 | 1.83× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.537 | 0.525 | 0.98× | 7.037 | 6.490 | 0.92× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.112 | 0.006 | 0.05× | 1.301 | 0.054 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.178 | 0.020 | 0.11× | 1.674 | 0.154 | 0.09× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.018 | 0.009 | 0.49× | 0.047 | 0.010 | 0.22× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.261 | 0.003 | 0.01× | 0.366 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.048 | 0.022 | 0.45× | 0.075 | 0.027 | 0.36× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.034 | 0.002 | 0.07× | 0.041 | 0.003 | 0.08× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.474 | 0.306 | 0.64× | 5.134 | 3.252 | 0.63× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.038 | 0.016 | 0.43× | 0.091 | 0.059 | 0.65× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.010 | 0.003 | 0.26× | 0.013 | 0.003 | 0.26× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.150 | 0.022 | 0.15× | 1.426 | 0.181 | 0.13× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.006 | 0.004 | 0.56× | 0.008 | 0.005 | 0.62× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.096 | 0.864 | 8.98× | 0.397 | 1.245 | 3.14× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.069 | 0.003 | 0.05× | 0.501 | 0.006 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.039 | 0.333 | 8.49× | 0.117 | 0.390 | 3.34× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.027 | 0.017 | 0.62× | 0.031 | 0.018 | 0.59× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.355 | 8.074 | 22.77× | 3.901 | 77.508 | 19.87× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 0.504 | 8.822 | 17.49× | 5.515 | 90.028 | 16.32× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.444 | 0.517 | 1.17× | 8.108 | 9.480 | 1.17× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.352 | 0.636 | 1.81× | 6.351 | 5.740 | 0.90× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.715 | 0.663 | 0.93× | 6.307 | 3.458 | 0.55× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.058 | – | – | 0.228 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.262 | 0.056 | 0.21× | 0.141 | 0.029 | 0.20× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.012 | 0.006 | 0.47× | 0.007 | 0.003 | 0.44× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.557 | – | – | 2.423 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.636 | 0.287 | 0.45× | 4.304 | 1.513 | 0.35× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.579 | 0.053 | 0.09× | 2.705 | 0.234 | 0.09× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.102 | 0.033 | 0.32× | 0.540 | 0.232 | 0.43× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.109 | 0.555 | 5.08× | 0.427 | 1.020 | 2.39× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.080 | 0.031 | 0.39× | 0.525 | 0.321 | 0.61× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.053 | 0.354 | 6.63× | 0.248 | 0.487 | 1.97× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.081 | 0.427 | 5.29× | 0.428 | 0.653 | 1.53× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.040 | 0.243 | 6.04× | 0.203 | 0.381 | 1.88× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.053 | 0.017 | 0.31× | 0.210 | 0.097 | 0.46× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.237 | 0.551 | 2.32× | 1.510 | 1.657 | 1.10× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.065 | 0.376 | 5.81× | 0.266 | 0.494 | 1.86× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.111 | 0.649 | 5.86× | 0.558 | 1.698 | 3.04× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.067 | 0.456 | 6.82× | 0.441 | 0.859 | 1.95× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.100 | 0.464 | 4.63× | 0.599 | 1.036 | 1.73× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.127 | 0.033 | 0.26× | 0.929 | 0.270 | 0.29× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.076 | 0.011 | 0.15× | 0.375 | 0.076 | 0.20× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.114 | 0.062 | 0.55× | 0.765 | 0.460 | 0.60× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.119 | 0.032 | 0.27× | 0.648 | 0.161 | 0.25× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.098 | 0.043 | 0.43× | 0.594 | 0.298 | 0.50× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.536 | 8.087 | 15.09× | 3.104 | 70.463 | 22.70× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.353 | 8.448 | 23.94× | 4.842 | 85.837 | 17.73× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.347 | 7.791 | 22.44× | 4.320 | 92.596 | 21.44× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.319 | 7.869 | 24.68× | 3.732 | 119.7 | 32.08× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.292 | 7.577 | 25.96× | 3.227 | 75.039 | 23.25× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.517 | 8.429 | 16.29× | 4.646 | 105.7 | 22.75× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.012 | 0.002 | 0.20× | 0.014 | 0.003 | 0.22× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.026 | 0.018 | 0.70× | 0.224 | 0.153 | 0.68× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.089 | 0.579 | 6.47× | 0.151 | 1.084 | 7.16× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.021 | 0.017 | 0.84× | 0.155 | 0.133 | 0.86× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.033 | 0.009 | 0.28× | 0.274 | 0.100 | 0.36× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.010 | 0.002 | 0.24× | 0.020 | 0.006 | 0.28× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.010 | 0.002 | 0.22× | 0.019 | 0.003 | 0.14× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.171 | 0.019 | 0.11× | 1.722 | 0.154 | 0.09× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.046 | 0.005 | 0.10× | 0.055 | 0.006 | 0.10× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.009 | 0.002 | 0.25× | 0.009 | 0.003 | 0.28× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.220 | 0.404 | 1.84× | 2.578 | 8.113 | 3.15× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.068 | 0.047 | 0.69× | 0.175 | 0.088 | 0.50× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.005 | 0.004 | 0.83× | 0.008 | 0.005 | 0.71× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.313 | 0.018 | 0.06× | 4.732 | 0.130 | 0.03× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.005 | 0.002 | 0.46× | 0.006 | 0.003 | 0.52× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.005 | 0.003 | 0.52× | 0.006 | 0.004 | 0.58× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.171 | 0.052 | 0.30× | 1.730 | 0.475 | 0.27× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.104 | 0.013 | 0.12× | 0.640 | 0.056 | 0.09× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.116 | 0.017 | 0.14× | 0.647 | 0.098 | 0.15× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.019 | 0.004 | 0.20× | 0.014 | 0.003 | 0.19× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.019 | 0.007 | 0.37× | 0.015 | 0.007 | 0.46× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.085 | 0.007 | 0.08× | 0.650 | 0.049 | 0.08× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.403 | 0.052 | 0.02× | 2.479 | 0.052 | 0.02× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.202 | 0.450 | 2.23× | 2.727 | 6.314 | 2.32× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.331 | – | – | 3.340 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.233 | 2.308 | 9.92× | 1.710 | 4.416 | 2.58× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.372 | 1.238 | 3.33× | 3.115 | 9.286 | 2.98× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.237 | 2.178 | 9.20× | 1.743 | 4.545 | 2.61× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.168 | 1.437 | 8.58× | 1.536 | 4.154 | 2.70× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.166 | 1.316 | 7.91× | 1.572 | 4.490 | 2.86× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.029 | 0.365 | 12.61× | 0.285 | 0.868 | 3.05× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.013 | 0.067 | 5.21× | 0.046 | 0.066 | 1.43× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.367 | 0.416 | 1.13× | 4.212 | 4.912 | 1.17× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 85.925 | – | – | 450.3 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.026 | 0.224 | 8.78× | 0.043 | 0.326 | 7.58× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 2.187 | 2.636 | 1.21× | 35.091 | 35.407 | 1.01× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.053 | 0.039 | 0.73× | 0.680 | 0.432 | 0.64× | 0.0e+00 |
| `statistics.compare` | timed alone | 48.408 | – | – | 398.3 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.018 | 0.235 | 13.33× | 0.026 | 0.234 | 9.06× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 1.418 | – | – | 16.135 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.116 | 0.625 | 5.39× | 0.494 | 2.749 | 5.57× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.616 | 1.691 | 2.75× | 7.624 | 14.147 | 1.86× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.076 | 0.038 | 0.50× | 0.929 | 0.415 | 0.45× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.936 | 3.617 | 3.87× | 15.672 | 42.801 | 2.73× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.585 | 2.643 | 4.51× | 7.153 | 27.579 | 3.86× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.018 | 0.073 | 4.00× | 0.124 | 0.345 | 2.78× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 43.175 | – | – | 367.7 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.138 | 0.516 | 3.74× | 0.403 | 1.026 | 2.54× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.022 | 0.153 | 7.07× | 0.014 | 0.150 | 10.40× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 1.127 | – | – | 9.778 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.069 | 0.734 | 10.71× | 0.051 | 0.606 | 11.80× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.151 | 0.868 | 5.75× | 0.515 | 1.448 | 2.81× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.459 | 1.988 | 4.33× | 4.610 | 15.683 | 3.40× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.018 | 0.005 | 0.25× | 0.027 | 0.007 | 0.26× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.022 | 0.007 | 0.31× | 0.035 | 0.014 | 0.41× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 1.573 | 1.815 | 1.15× | 11.007 | 9.054 | 0.82× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.022 | 0.009 | 0.43× | 0.097 | 0.030 | 0.31× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.416 | 0.226 | 0.54× | 2.708 | 2.857 | 1.06× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 0.977 | 5.417 | 5.54× | 9.696 | 73.581 | 7.59× | 0.0e+00 |
| `text.bradley_terry` | timed alone | 1.005 | – | – | 1.890 | – | – | – |
| `text.chrf` | sacreBLEU | 1.704 | 13.637 | 8.00× | 25.449 | 155.2 | 6.10× | 1.4e-14 |
| `text.cider` | timed alone | 3.414 | – | – | 41.325 | – | – | – |
| `text.citation_precision` | timed alone | 0.170 | – | – | 1.843 | – | – | – |
| `text.citation_recall` | timed alone | 0.120 | – | – | 1.350 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.154 | 0.004 | 0.03× | 1.266 | 0.014 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.050 | 0.063 | 1.26× | 0.415 | 0.708 | 1.70× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.044 | 0.009 | 0.21× | 0.372 | 0.060 | 0.16× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.207 | 0.222 | 1.07× | 2.285 | 2.982 | 1.31× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.120 | 0.122 | 1.02× | 0.309 | 0.284 | 0.92× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.042 | 0.042 | 0.99× | 0.227 | 0.200 | 0.88× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.324 | – | – | 2.731 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.274 | 0.422 | 1.54× | 2.459 | 4.287 | 1.74× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.039 | 0.098 | 2.48× | 0.096 | 0.733 | 7.62× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.026 | 0.020 | 0.80× | 0.192 | 0.148 | 0.77× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.105 | 0.144 | 1.38× | 0.893 | 1.415 | 1.58× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.265 | 0.111 | 0.42× | 2.298 | 1.117 | 0.49× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.035 | 0.012 | 0.34× | 0.336 | 0.095 | 0.28× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.057 | – | – | 0.498 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 1.368 | 3.284 | 2.40× | 18.581 | 34.601 | 1.86× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.166 | 0.135 | 0.81× | 1.635 | 1.362 | 0.83× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.115 | 0.752 | 6.54× | 0.303 | 1.026 | 3.38× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.070 | 0.014 | 0.20× | 0.571 | 0.191 | 0.33× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.068 | 0.094 | 1.39× | 0.147 | 0.367 | 2.50× | 2.9e-15 |
| `text.mauve` | timed alone | 3.262 | – | – | 490.8 | – | – | – |
| `text.meteor` | NLTK | 0.782 | 5.431 | 6.95× | 7.914 | 54.297 | 6.86× | 0.0e+00 |
| `text.model_score` | timed alone | 0.015 | – | – | 0.088 | – | – | – |
| `text.moverscore` | timed alone | 4.629 | – | – | 74.707 | – | – | – |
| `text.perplexity` | NumPy formula | 0.045 | 0.008 | 0.18× | 0.375 | 0.062 | 0.17× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.068 | 0.003 | 0.05× | 0.638 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.345 | – | – | 3.429 | – | – | – |
| `text.rouge_1` | rouge-score | 0.444 | 3.127 | 7.04× | 4.787 | 31.003 | 6.48× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.711 | 3.040 | 4.27× | 4.464 | 33.621 | 7.53× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 0.453 | 3.701 | 8.17× | 7.936 | 39.080 | 4.92× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 0.888 | 6.010 | 6.77× | 14.368 | 57.815 | 4.02× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.016 | 0.005 | 0.29× | 0.040 | 0.009 | 0.23× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 0.504 | 13.239 | 26.27× | 12.224 | 422.8 | 34.59× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 1.287 | – | – | 0.959 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 1.151 | 12.363 | 10.74× | 10.477 | 139.0 | 13.27× | 0.0e+00 |
| `text.spice` | Python stdlib | 1.971 | 0.094 | 0.05× | 22.114 | 1.220 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 1.738 | 27.839 | 16.02× | 10.820 | 192.1 | 17.76× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.496 | – | – | 3.233 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.619 | – | – | 3.491 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.440 | 0.040 | 0.09× | 2.597 | 0.198 | 0.08× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.933 | – | – | 1.757 | – | – | – |
| `text.win_rate` | NumPy formula | 0.071 | 0.009 | 0.13× | 0.347 | 0.014 | 0.04× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.837 | 0.784 | 0.94× | 5.230 | 5.047 | 0.97× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.283 | 0.249 | 0.88× | 0.972 | 0.880 | 0.90× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.114 | 0.066 | 0.58× | 0.901 | 0.802 | 0.89× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.645 | 2.418 | 3.75× | 1.255 | 3.145 | 2.51× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.092 | 0.041 | 0.44× | 0.918 | 0.826 | 0.90× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.070 | 0.032 | 0.45× | 0.901 | 0.788 | 0.87× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.042 | 0.031 | 0.76× | 0.829 | 0.780 | 0.94× | 0.0e+00 |

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
