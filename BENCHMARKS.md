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
EvalSuite 0.5.4 benchmarks | Python 3.13.18 | NumPy 2.5.4 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 12/12 | 10.34× | 1.74×–98.98× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 24/24 | 4.19× | 1.08×–21.26× |
| LLM evaluation | 6 | 18 | 18/18 | 17/18 | 1.96× | 1.00×–8.59× |
| LLM systems | 6 | 18 | 18/18 | 15/18 | 1.63× | 0.69×–5.75× |
| Segmentation and object detection | 3 | 9 | 9/9 | 7/9 | 2.28× | 0.96×–16.01× |
| Overall | 27 | 81 | 81/81 | 75/81 | 3.07× | 0.69×–98.98× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.063 | 0.976 | 15.53× | 0.02 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 0.765 | 11.836 | 15.46× | 0.77 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 7.031 | 115.858 | 16.48× | 7.63 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.042 | 0.042 | 1.00× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.270 | 0.434 | 1.61× | 0.26 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 1.549 | 3.064 | 1.98× | 2.26 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.166 | 6.704 | 40.41× | 0.03 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 0.909 | 90.023 | 98.98× | 1.69 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 8.833 | 865.770 | 98.01× | 16.28 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.118 | 1.079 | 9.18× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 11.062 | 24.255 | 2.19× | 8.39 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 163.624 | 284.486 | 1.74× | 83.93 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.639 | 3.472 | 5.43× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 15.576 | 162.840 | 10.45× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 182.088 | 1206.253 | 6.62× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.138 | 0.617 | 4.46× | 0.02 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 0.500 | 0.997 | 1.99× | 0.76 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 4.973 | 5.381 | 1.08× | 7.63 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.251 | 4.614 | 18.37× | 0.02 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 1.820 | 38.675 | 21.26× | 0.76 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 21.284 | 330.349 | 15.52× | 7.63 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.162 | 1.075 | 6.64× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 9.914 | 14.663 | 1.48× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 145.591 | 197.408 | 1.36× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 10.872 | 16.712 | 1.54× | 0.54 | 1.36 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 71.203 | 69.519 | 0.98× | 1.15 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 650.695 | 626.020 | 0.96× | 7.00 | 34.44 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.052 | 0.065 | 1.25× | 0.04 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 3.638 | 4.397 | 1.21× | 3.82 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 63.667 | 72.436 | 1.14× | 38.15 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.114 | 0.667 | 5.87× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 0.699 | 1.865 | 2.67× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 8.385 | 14.707 | 1.75× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.144 | 1.240 | 8.59× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 11.176 | 55.208 | 4.94× | 0.49 | 3.12 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 122.959 | 578.367 | 4.70× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.230 | 3.682 | 16.01× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 2.998 | 22.857 | 7.62× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 30.522 | 221.641 | 7.26× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.501 | 0.547 | 1.09× | 0.03 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 5.989 | 6.411 | 1.07× | 0.03 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 12.297 | 13.692 | 1.11× | 0.03 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.026 | 0.269 | 10.56× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.015 | 0.187 | 12.24× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.015 | 0.177 | 11.90× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.112 | 0.665 | 5.94× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 6.783 | 22.833 | 3.37× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 136.856 | 286.173 | 2.09× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.110 | 0.729 | 6.62× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.416 | 1.094 | 2.63× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 5.945 | 9.031 | 1.52× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.108 | 0.248 | 2.30× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 9.245 | 23.274 | 2.52× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 96.893 | 227.957 | 2.35× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 40.923 | 48.881 | 1.19× | 0.36 | 0.25 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 422.971 | 509.601 | 1.20× | 1.76 | 2.06 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 1.837 | 1.809 | 0.98× | 0.07 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.234 | 1.287 | 5.51× | 0.06 | 0.08 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 11.624 | 23.114 | 1.99× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 164.076 | 254.096 | 1.55× | 54.36 | 76.30 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.079 | 0.453 | 5.75× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.530 | 1.288 | 2.43× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 9.232 | 10.742 | 1.16× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.076 | 0.208 | 2.72× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 8.348 | 21.379 | 2.56× | 0.70 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 93.952 | 211.690 | 2.25× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.024 | 0.034 | 1.37× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 1.630 | 1.593 | 0.98× | 0.77 | 0.77 | 3.1e-15 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 19.156 | 13.170 | 0.69× | 7.63 | 7.63 | 1.3e-15 |
| systems: maintainability index (100 programs) | 100,000 | radon | 8.200 | 9.104 | 1.11× | 0.18 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 89.222 | 95.385 | 1.07× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.349 | 0.354 | 1.02× | 0.04 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 1.243 | 1.385 | 1.11× | 0.11 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 135.804 | 164.316 | 1.21× | 0.26 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 1454.193 | 1932.585 | 1.33× | 2.52 | 211.52 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.311 | 0.368 | 1.18× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 30.157 | 35.808 | 1.19× | 0.12 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 323.896 | 384.123 | 1.19× | 1.15 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.299 | 0.585 | 1.95× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 32.250 | 60.337 | 1.87× | 0.12 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 340.118 | 630.704 | 1.85× | 1.16 | 6.55 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.5.4, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 5.59× | 1.16×–29.58× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 40/42 | 5.31× | 0.90×–32.16× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 43/46 | 3.72× | 0.74×–46.83× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 14/14 | 2.97× | 1.06×–8.00× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 16/18 | 5.24× | 0.75×–21.61× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 165/172 | 4.67× | 0.74×–46.83× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.007 | 0.002 | 0.27× | 0.007 | 0.002 | 0.26× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.009 | 0.003 | 0.31× | 0.009 | 0.004 | 0.42× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 0.888 | 2.683 | 3.02× | 10.592 | 31.931 | 3.01× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.186 | – | – | 1.729 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.092 | 0.010 | 0.10× | 0.927 | 0.065 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.017 | 0.002 | 0.10× | 0.025 | 0.002 | 0.07× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.74× | 0.003 | 0.003 | 0.80× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.134 | – | – | 1.691 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.017 | 0.002 | 0.12× | 0.024 | 0.003 | 0.12× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.621 | 0.473 | 0.76× | 7.310 | 5.610 | 0.77× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.519 | 3.413 | 6.58× | 4.966 | 31.907 | 6.42× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 0.947 | 6.605 | 6.97× | 9.639 | 58.595 | 6.08× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.328 | 0.460 | 1.40× | 3.071 | 4.847 | 1.58× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.050 | 0.550 | 11.09× | 0.405 | 3.318 | 8.19× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.901 | 1.962 | 2.18× | 10.630 | 16.740 | 1.57× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.388 | 1.884 | 4.85× | 3.401 | 12.738 | 3.75× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.058 | 0.933 | 16.13× | 0.532 | 5.935 | 11.16× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.068 | 1.478 | 21.59× | 0.405 | 9.813 | 24.24× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.288 | 0.478 | 1.66× | 2.859 | 5.017 | 1.75× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.382 | 2.439 | 6.38× | 3.422 | 14.357 | 4.20× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.074 | 1.932 | 26.28× | 0.437 | 11.051 | 25.31× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.569 | 3.240 | 5.69× | 5.972 | 27.499 | 4.60× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.101 | 2.072 | 20.50× | 0.436 | 11.115 | 25.51× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.486 | 3.027 | 6.23× | 5.561 | 24.216 | 4.35× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.366 | 3.280 | 8.97× | 3.429 | 23.483 | 6.85× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.067 | 1.975 | 29.58× | 0.410 | 11.230 | 27.40× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.364 | 2.353 | 6.46× | 3.465 | 14.369 | 4.15× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.587 | 2.576 | 4.39× | 3.368 | 14.369 | 4.27× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 11.697 | 22.443 | 1.92× | 121.5 | 191.7 | 1.58× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.068 | 1.908 | 27.98× | 0.460 | 11.123 | 24.18× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.669 | 2.472 | 3.69× | 7.017 | 20.126 | 2.87× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.060 | 0.035 | 0.59× | 0.418 | 0.311 | 0.74× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.058 | 1.776 | 30.84× | 0.412 | 12.890 | 31.28× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.058 | 1.856 | 32.16× | 0.432 | 12.778 | 29.56× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.143 | 0.017 | 0.12× | 1.442 | 0.150 | 0.10× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.068 | 2.007 | 29.40× | 0.427 | 11.078 | 25.96× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.070 | 1.991 | 28.37× | 0.429 | 11.060 | 25.79× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.059 | 0.035 | 0.59× | 0.410 | 0.310 | 0.76× | 0.0e+00 |
| `code.code_complexity` | radon | 0.755 | 0.804 | 1.06× | 6.840 | 7.526 | 1.10× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 4.678 | 6.076 | 1.30× | 41.839 | 53.765 | 1.29× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.011 | 0.001 | 0.08× | 0.016 | 0.001 | 0.08× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.018 | 0.014 | 0.76× | 0.129 | 0.105 | 0.81× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.003 | 0.002 | 0.81× | 0.003 | 0.003 | 0.82× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.067 | 0.014 | 0.21× | 0.635 | 0.109 | 0.17× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.026 | 0.002 | 0.08× | 0.032 | 0.003 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.076 | 0.007 | 0.09× | 0.673 | 0.044 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.010 | 0.001 | 0.09× | 0.013 | 0.001 | 0.10× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 1.786 | 1.087 | 0.61× | 17.053 | 10.595 | 0.62× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.344 | 0.261 | 0.76× | 3.471 | 2.681 | 0.77× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 1.813 | 13.960 | 7.70× | 21.620 | 70.263 | 3.25× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.041 | 0.021 | 0.50× | 6.931 | 4.137 | 0.60× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 2.199 | 14.049 | 6.39× | 24.107 | 72.489 | 3.01× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.012 | 0.002 | 0.18× | 0.052 | 0.003 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.36× | 0.007 | 0.004 | 0.49× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.27× | 0.015 | 0.004 | 0.27× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.022 | 0.023 | 1.03× | 0.022 | 0.027 | 1.23× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.004 | 0.002 | 0.48× | 0.007 | 0.002 | 0.29× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.067 | 0.001 | 0.02× | 0.078 | 0.001 | 0.02× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.016 | 0.003 | 0.17× | 0.022 | 0.003 | 0.16× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.030 | 0.003 | 0.10× | 0.041 | 0.005 | 0.11× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.019 | 0.002 | 0.11× | 0.028 | 0.003 | 0.09× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.026 | 0.002 | 0.07× | 0.052 | 0.003 | 0.06× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.332 | 0.096 | 0.29× | 3.351 | 0.959 | 0.29× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.458 | 0.415 | 0.91× | 4.351 | 4.259 | 0.98× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.091 | 0.004 | 0.05× | 0.990 | 0.041 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.473 | 0.016 | 0.03× | 4.503 | 0.122 | 0.03× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.017 | 0.008 | 0.48× | 0.042 | 0.015 | 0.36× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.241 | 0.002 | 0.01× | 0.375 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.040 | 0.021 | 0.52× | 0.073 | 0.033 | 0.44× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.030 | 0.002 | 0.07× | 0.048 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.403 | 0.262 | 0.65× | 4.150 | 2.620 | 0.63× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.030 | 0.016 | 0.53× | 0.065 | 0.046 | 0.70× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.008 | 0.002 | 0.26× | 0.010 | 0.003 | 0.27× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.406 | 0.017 | 0.04× | 4.108 | 0.146 | 0.04× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.007 | 0.003 | 0.42× | 0.011 | 0.004 | 0.37× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.088 | 0.697 | 7.94× | 0.373 | 1.015 | 2.72× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.056 | 0.003 | 0.05× | 0.409 | 0.006 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.032 | 0.258 | 8.00× | 0.098 | 0.276 | 2.81× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.025 | 0.012 | 0.49× | 0.031 | 0.015 | 0.48× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.885 | 6.162 | 6.97× | 9.337 | 60.606 | 6.49× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 1.475 | 6.774 | 4.59× | 15.298 | 67.094 | 4.39× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.365 | 0.425 | 1.17× | 3.882 | 4.586 | 1.18× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.290 | 0.266 | 0.92× | 2.908 | 2.702 | 0.93× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.293 | 0.263 | 0.90× | 3.014 | 2.764 | 0.92× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.023 | – | – | 0.171 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.112 | 0.023 | 0.21× | 0.121 | 0.024 | 0.20× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.005 | 0.002 | 0.41× | 0.006 | 0.003 | 0.44× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.187 | – | – | 1.770 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.492 | 0.110 | 0.22× | 4.862 | 1.718 | 0.35× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.245 | 0.021 | 0.09× | 3.787 | 0.343 | 0.09× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.045 | 0.016 | 0.35× | 0.669 | 0.253 | 0.38× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.048 | 0.251 | 5.26× | 0.574 | 1.206 | 2.10× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.057 | 0.033 | 0.57× | 0.948 | 0.538 | 0.57× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.024 | 0.178 | 7.55× | 0.258 | 0.506 | 1.96× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.037 | 0.199 | 5.31× | 0.445 | 0.712 | 1.60× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.022 | 0.121 | 5.58× | 0.237 | 0.423 | 1.78× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.022 | 0.008 | 0.36× | 0.193 | 0.082 | 0.42× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.121 | 0.265 | 2.18× | 1.193 | 1.389 | 1.16× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.026 | 0.168 | 6.38× | 0.283 | 0.454 | 1.60× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.056 | 0.291 | 5.20× | 0.492 | 1.076 | 2.19× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.032 | 0.252 | 7.84× | 0.404 | 0.783 | 1.94× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.045 | 0.216 | 4.80× | 0.506 | 0.636 | 1.26× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.057 | 0.015 | 0.27× | 0.718 | 0.190 | 0.26× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.029 | 0.006 | 0.19× | 0.291 | 0.037 | 0.13× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.055 | 0.037 | 0.66× | 0.487 | 0.348 | 0.72× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.054 | 0.015 | 0.29× | 0.644 | 0.149 | 0.23× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.050 | 0.021 | 0.42× | 0.688 | 0.316 | 0.46× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.319 | 5.926 | 18.59× | 3.251 | 55.581 | 17.10× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.335 | 5.911 | 17.63× | 3.533 | 56.414 | 15.97× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.322 | 5.955 | 18.51× | 3.337 | 56.640 | 16.97× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.466 | 5.952 | 12.77× | 4.764 | 57.144 | 12.00× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.342 | 5.921 | 17.32× | 3.313 | 57.754 | 17.43× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.320 | 5.976 | 18.68× | 3.303 | 55.376 | 16.76× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.010 | 0.002 | 0.21× | 0.012 | 0.003 | 0.22× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.019 | 0.014 | 0.73× | 0.169 | 0.106 | 0.63× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.070 | 0.453 | 6.44× | 0.073 | 0.454 | 6.20× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.016 | 0.013 | 0.80× | 0.112 | 0.097 | 0.86× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.026 | 0.007 | 0.29× | 0.204 | 0.050 | 0.24× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.009 | 0.002 | 0.23× | 0.010 | 0.003 | 0.26× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.009 | 0.002 | 0.23× | 0.016 | 0.004 | 0.25× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.440 | 0.014 | 0.03× | 4.414 | 0.118 | 0.03× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.042 | 0.004 | 0.11× | 0.048 | 0.005 | 0.10× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.007 | 0.002 | 0.31× | 0.009 | 0.004 | 0.38× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.674 | 0.336 | 0.50× | 6.888 | 3.413 | 0.50× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.062 | 0.043 | 0.70× | 0.102 | 0.068 | 0.66× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.77× | 0.003 | 0.003 | 0.83× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.691 | 0.017 | 0.02× | 7.467 | 0.104 | 0.01× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.004 | 0.002 | 0.50× | 0.006 | 0.004 | 0.67× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.004 | 0.003 | 0.59× | 0.007 | 0.005 | 0.69× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.128 | 0.039 | 0.30× | 1.214 | 0.355 | 0.29× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.046 | 0.007 | 0.15× | 0.432 | 0.044 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.058 | 0.010 | 0.17× | 0.510 | 0.075 | 0.15× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.010 | 0.002 | 0.22× | 0.014 | 0.004 | 0.28× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.011 | 0.004 | 0.32× | 0.014 | 0.005 | 0.38× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.233 | 0.006 | 0.02× | 2.264 | 0.038 | 0.02× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.034 | 0.050 | 0.02× | 2.037 | 0.053 | 0.03× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.422 | 0.314 | 0.75× | 5.386 | 4.078 | 0.76× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.215 | – | – | 2.344 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.138 | 1.682 | 12.19× | 1.443 | 11.505 | 7.97× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.435 | 0.502 | 1.15× | 5.922 | 6.212 | 1.05× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.138 | 1.604 | 11.66× | 1.446 | 11.254 | 7.78× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.123 | 1.654 | 13.41× | 1.376 | 11.276 | 8.20× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.142 | 1.625 | 11.43× | 1.423 | 11.150 | 7.84× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.022 | 0.478 | 21.61× | 0.225 | 3.260 | 14.47× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.010 | 0.053 | 5.33× | 0.040 | 0.055 | 1.37× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.278 | 0.312 | 1.12× | 3.469 | 4.008 | 1.16× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 135.5 | – | – | 1,333.8 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.026 | 0.185 | 7.26× | 0.023 | 0.220 | 9.49× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.569 | 2.119 | 1.35× | 21.709 | 22.862 | 1.05× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.041 | 0.030 | 0.72× | 0.487 | 0.266 | 0.55× | 0.0e+00 |
| `statistics.compare` | timed alone | 40.933 | – | – | 561.0 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.016 | 0.192 | 12.05× | 0.016 | 0.179 | 11.49× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 3.981 | – | – | 45.103 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.095 | 0.516 | 5.40× | 0.242 | 1.723 | 7.11× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.523 | 1.470 | 2.81× | 6.464 | 13.084 | 2.02× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.059 | 0.030 | 0.52× | 0.802 | 0.264 | 0.33× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.847 | 3.006 | 3.55× | 12.389 | 34.450 | 2.78× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.566 | 2.026 | 3.58× | 6.337 | 21.503 | 3.39× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.013 | 0.051 | 4.04× | 0.192 | 0.536 | 2.79× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 38.763 | – | – | 545.2 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.076 | 0.276 | 3.63× | 0.417 | 0.920 | 2.21× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.012 | 0.088 | 7.19× | 0.008 | 0.079 | 9.79× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.038 | – | – | 21.474 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.457 | 0.435 | 0.95× | 0.539 | 0.488 | 0.90× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.090 | 0.475 | 5.30× | 0.540 | 1.265 | 2.34× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.306 | 1.312 | 4.29× | 3.620 | 11.136 | 3.08× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.011 | 0.003 | 0.25× | 0.014 | 0.003 | 0.23× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.014 | 0.004 | 0.27× | 0.019 | 0.007 | 0.36× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 0.957 | 0.758 | 0.79× | 9.918 | 7.771 | 0.78× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.010 | 0.005 | 0.50× | 0.072 | 0.026 | 0.36× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.245 | 0.138 | 0.56× | 2.390 | 1.474 | 0.62× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 2.761 | 3.954 | 1.43× | 30.543 | 44.820 | 1.47× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.143 | 24.369 | 21.31× | 2.109 | 98.771 | 46.83× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 9.355 | 9.612 | 1.03× | 102.4 | 112.3 | 1.10× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 13.508 | 18.646 | 1.38× | 149.8 | 231.9 | 1.55× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.118 | – | – | 1.703 | – | – | – |
| `text.citation_recall` | timed alone | 0.093 | – | – | 1.168 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.120 | 0.003 | 0.03× | 0.921 | 0.010 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.035 | 0.051 | 1.47× | 0.309 | 0.558 | 1.80× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.039 | 0.011 | 0.29× | 0.327 | 0.065 | 0.20× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.438 | 0.150 | 0.34× | 4.633 | 2.669 | 0.58× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.317 | 0.078 | 0.25× | 0.901 | 0.197 | 0.22× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.034 | 0.023 | 0.69× | 0.198 | 0.158 | 0.80× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.231 | – | – | 1.959 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.195 | 0.339 | 1.74× | 1.822 | 4.817 | 2.64× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.033 | 0.083 | 2.50× | 0.106 | 0.648 | 6.08× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.020 | 0.016 | 0.82× | 0.157 | 0.125 | 0.80× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.350 | 0.115 | 0.33× | 3.691 | 1.149 | 0.31× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.184 | 0.082 | 0.44× | 1.752 | 0.960 | 0.55× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.025 | 0.009 | 0.37× | 0.403 | 0.118 | 0.29× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.044 | – | – | 0.617 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 0.975 | 2.325 | 2.38× | 13.841 | 22.643 | 1.64× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.211 | 0.165 | 0.78× | 1.181 | 0.986 | 0.84× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.126 | 0.846 | 6.70× | 0.230 | 0.805 | 3.50× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.075 | 0.020 | 0.26× | 0.416 | 0.149 | 0.36× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.093 | 0.121 | 1.30× | 0.181 | 0.314 | 1.73× | 2.9e-15 |
| `text.mauve` | timed alone | 4.322 | – | – | 450.2 | – | – | – |
| `text.meteor` | NLTK | 5.521 | 4.076 | 0.74× | 34.407 | 38.826 | 1.13× | 0.0e+00 |
| `text.model_score` | timed alone | 0.013 | – | – | 0.067 | – | – | – |
| `text.moverscore` | POT | 1.192 | 1.101 | 0.92× | 12.678 | 11.455 | 0.90× | 5.6e-17 |
| `text.perplexity` | NumPy formula | 0.040 | 0.011 | 0.26× | 0.330 | 0.065 | 0.20× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.049 | 0.003 | 0.06× | 0.437 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.256 | – | – | 2.492 | – | – | – |
| `text.rouge_1` | rouge-score | 0.602 | 2.288 | 3.80× | 6.078 | 21.595 | 3.55× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.762 | 2.238 | 2.94× | 7.272 | 22.875 | 3.15× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 1.250 | 2.759 | 2.21× | 13.270 | 29.575 | 2.23× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 2.588 | 4.162 | 1.61× | 27.397 | 44.050 | 1.61× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.013 | 0.004 | 0.30× | 0.029 | 0.008 | 0.25× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 5.345 | 9.109 | 1.70× | 132.3 | 253.6 | 1.92× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.344 | – | – | 0.421 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 3.003 | 5.467 | 1.82× | 30.513 | 58.583 | 1.92× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.941 | 0.044 | 0.05× | 8.705 | 0.518 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 9.380 | 13.527 | 1.44× | 92.727 | 142.2 | 1.53× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.209 | – | – | 2.407 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.326 | – | – | 2.599 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.252 | 0.028 | 0.11× | 1.990 | 0.152 | 0.08× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.553 | – | – | 1.326 | – | – | – |
| `text.win_rate` | NumPy formula | 0.047 | 0.007 | 0.15× | 0.242 | 0.011 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.517 | 0.498 | 0.96× | 3.742 | 3.647 | 0.97× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.158 | 0.122 | 0.77× | 0.674 | 0.659 | 0.98× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.073 | 0.043 | 0.59× | 0.681 | 0.625 | 0.92× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.343 | 1.578 | 4.60× | 1.064 | 2.838 | 2.67× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.070 | 0.044 | 0.62× | 0.673 | 0.603 | 0.90× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.053 | 0.026 | 0.49× | 0.661 | 0.573 | 0.87× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.036 | 0.027 | 0.75× | 0.597 | 0.549 | 0.92× | 0.0e+00 |
