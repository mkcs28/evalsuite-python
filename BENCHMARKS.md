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
EvalSuite 0.6.1 benchmarks | Python 3.13.18 | NumPy 2.5.4 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 12/12 | 10.21× | 1.66×–90.02× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 22/24 | 3.90× | 0.97×–18.22× |
| LLM evaluation | 6 | 18 | 18/18 | 17/18 | 4.15× | 0.99×–11.18× |
| LLM systems | 6 | 18 | 18/18 | 17/18 | 1.77× | 0.89×–5.93× |
| Segmentation and object detection | 3 | 9 | 9/9 | 9/9 | 6.98× | 1.71×–22.55× |
| Overall | 27 | 81 | 81/81 | 77/81 | 4.08× | 0.89×–90.02× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.063 | 0.959 | 15.26× | 0.02 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 0.773 | 11.924 | 15.42× | 0.77 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 6.943 | 102.395 | 14.75× | 7.63 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.041 | 0.040 | 0.99× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.151 | 0.295 | 1.95× | 0.29 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 1.391 | 3.282 | 2.36× | 2.53 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.159 | 6.762 | 42.61× | 0.03 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 1.036 | 91.821 | 88.62× | 1.69 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 9.058 | 815.355 | 90.02× | 16.28 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.115 | 1.083 | 9.43× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 11.448 | 25.334 | 2.21× | 8.39 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 144.438 | 253.033 | 1.75× | 83.93 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.426 | 2.288 | 5.37× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 22.268 | 78.273 | 3.52× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 163.246 | 921.867 | 5.65× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.088 | 0.394 | 4.45× | 0.02 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 0.578 | 0.988 | 1.71× | 0.76 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 4.349 | 4.526 | 1.04× | 7.63 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.183 | 3.030 | 16.57× | 0.02 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 2.108 | 36.798 | 17.45× | 0.76 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 17.125 | 311.991 | 18.22× | 7.63 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.102 | 0.654 | 6.42× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 9.314 | 12.371 | 1.33× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 129.868 | 169.469 | 1.30× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 1.120 | 16.620 | 14.84× | 0.95 | 1.36 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 3.521 | 69.088 | 19.62× | 1.13 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 27.323 | 616.003 | 22.55× | 2.83 | 34.54 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.043 | 0.043 | 0.99× | 0.04 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 3.262 | 3.740 | 1.15× | 3.82 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 85.777 | 82.846 | 0.97× | 38.15 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.069 | 0.473 | 6.86× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 0.857 | 2.226 | 2.60× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 5.608 | 9.326 | 1.66× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.143 | 1.295 | 9.06× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 11.772 | 57.271 | 4.87× | 0.49 | 3.11 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 122.212 | 587.674 | 4.81× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.163 | 2.598 | 15.91× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 2.750 | 23.078 | 8.39× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 28.893 | 204.868 | 7.09× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.161 | 0.275 | 1.71× | 0.27 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 3.305 | 6.390 | 1.93× | 0.29 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 7.031 | 13.474 | 1.92× | 0.28 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.016 | 0.184 | 11.55× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.015 | 0.199 | 12.95× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.015 | 0.185 | 12.06× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.083 | 0.476 | 5.75× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 6.218 | 21.475 | 3.45× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 109.884 | 331.462 | 3.02× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.068 | 0.453 | 6.62× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.404 | 1.076 | 2.66× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 4.852 | 7.208 | 1.49× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.109 | 0.248 | 2.27× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 9.699 | 22.509 | 2.32× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 95.333 | 224.485 | 2.35× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 31.127 | 48.627 | 1.56× | 0.37 | 0.25 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 323.966 | 521.448 | 1.61× | 1.98 | 2.06 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 1.281 | 1.798 | 1.40× | 0.07 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.222 | 1.189 | 5.37× | 0.06 | 0.08 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 12.370 | 23.516 | 1.90× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 168.384 | 278.679 | 1.66× | 54.36 | 76.30 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.079 | 0.466 | 5.93× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.807 | 1.717 | 2.13× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 11.523 | 12.420 | 1.08× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.081 | 0.202 | 2.48× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 8.470 | 21.415 | 2.53× | 0.70 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 108.958 | 209.233 | 1.92× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.029 | 0.040 | 1.37× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 0.894 | 1.606 | 1.80× | 0.76 | 0.77 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 14.923 | 13.283 | 0.89× | 7.63 | 7.63 | 0.0e+00 |
| systems: maintainability index (100 programs) | 100,000 | radon | 8.316 | 9.234 | 1.11× | 0.18 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 89.780 | 97.221 | 1.08× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.328 | 0.354 | 1.08× | 0.04 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 0.229 | 1.333 | 5.81× | 0.01 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 15.883 | 160.607 | 10.11× | 1.13 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 175.774 | 1965.678 | 11.18× | 11.44 | 211.52 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.069 | 0.352 | 5.10× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 6.360 | 35.857 | 5.64× | 0.58 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 61.533 | 369.592 | 6.01× | 5.82 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.117 | 0.549 | 4.68× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 11.526 | 59.524 | 5.16× | 0.63 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 153.936 | 617.206 | 4.01× | 6.95 | 6.55 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.6.1, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 5.68× | 1.19×–28.69× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 42/42 | 5.83× | 1.12×–37.45× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 46/46 | 8.21× | 1.17×–44.59× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 14/14 | 3.13× | 1.05×–7.61× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 9.15× | 1.19×–42.53× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 172/172 | 6.32× | 1.05×–44.59× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.007 | 0.002 | 0.25× | 0.010 | 0.003 | 0.31× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.009 | 0.003 | 0.33× | 0.014 | 0.006 | 0.43× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 0.884 | 2.638 | 2.99× | 10.877 | 32.089 | 2.95× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.183 | – | – | 1.744 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.095 | 0.010 | 0.11× | 0.986 | 0.065 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.016 | 0.002 | 0.11× | 0.025 | 0.002 | 0.07× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.73× | 0.003 | 0.003 | 0.78× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.132 | – | – | 1.823 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.014 | 0.002 | 0.14× | 0.023 | 0.003 | 0.12× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.620 | 0.472 | 0.76× | 7.338 | 5.691 | 0.78× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.460 | 3.016 | 6.55× | 6.363 | 37.090 | 5.83× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 0.832 | 5.509 | 6.62× | 13.282 | 66.627 | 5.02× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.332 | 0.458 | 1.38× | 3.355 | 5.142 | 1.53× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.051 | 0.546 | 10.62× | 0.476 | 3.489 | 7.33× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.853 | 1.823 | 2.14× | 11.570 | 18.163 | 1.57× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.341 | 1.711 | 5.02× | 3.661 | 13.227 | 3.61× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.058 | 0.836 | 14.43× | 0.758 | 6.502 | 8.57× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.066 | 1.389 | 21.05× | 0.495 | 10.248 | 20.69× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.276 | 0.482 | 1.75× | 3.139 | 5.281 | 1.68× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.366 | 2.215 | 6.06× | 3.678 | 15.071 | 4.10× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.073 | 1.881 | 25.65× | 0.539 | 11.510 | 21.37× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.498 | 3.030 | 6.08× | 7.216 | 28.371 | 3.93× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.070 | 1.757 | 25.24× | 0.581 | 11.510 | 19.79× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.418 | 2.564 | 6.14× | 6.493 | 23.950 | 3.69× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.349 | 3.056 | 8.76× | 3.554 | 24.396 | 6.86× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.065 | 1.876 | 28.69× | 0.490 | 11.866 | 24.20× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.363 | 2.213 | 6.10× | 3.610 | 14.978 | 4.15× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.357 | 2.265 | 6.35× | 3.519 | 14.791 | 4.20× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 9.223 | 20.116 | 2.18× | 121.5 | 187.8 | 1.55× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.066 | 1.806 | 27.45× | 0.581 | 11.885 | 20.44× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.597 | 2.350 | 3.94× | 7.968 | 19.484 | 2.45× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.059 | 0.035 | 0.60× | 0.339 | 0.262 | 0.77× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.060 | 1.766 | 29.23× | 0.334 | 12.514 | 37.45× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.058 | 1.792 | 31.16× | 0.338 | 12.296 | 36.33× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.140 | 0.017 | 0.12× | 1.296 | 0.128 | 0.10× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.070 | 2.046 | 29.23× | 0.359 | 10.856 | 30.27× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.071 | 2.059 | 29.01× | 0.353 | 10.784 | 30.53× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.059 | 0.036 | 0.60× | 0.349 | 0.266 | 0.76× | 0.0e+00 |
| `code.code_complexity` | radon | 0.735 | 0.771 | 1.05× | 6.745 | 7.524 | 1.12× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 3.471 | 5.968 | 1.72× | 31.504 | 52.811 | 1.68× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.012 | 0.001 | 0.09× | 0.016 | 0.001 | 0.07× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.019 | 0.014 | 0.75× | 0.126 | 0.100 | 0.80× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.003 | 0.002 | 0.76× | 0.003 | 0.003 | 0.79× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.068 | 0.013 | 0.19× | 0.615 | 0.106 | 0.17× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.026 | 0.002 | 0.08× | 0.033 | 0.003 | 0.08× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.079 | 0.010 | 0.12× | 0.670 | 0.044 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.010 | 0.001 | 0.09× | 0.014 | 0.001 | 0.09× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 1.794 | 1.063 | 0.59× | 17.208 | 10.537 | 0.61× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.348 | 0.277 | 0.80× | 3.489 | 2.814 | 0.81× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 0.322 | 13.701 | 42.53× | 2.262 | 70.456 | 31.15× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.039 | 0.020 | 0.53× | 6.970 | 3.762 | 0.54× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 0.348 | 14.692 | 42.27× | 2.367 | 67.273 | 28.43× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.013 | 0.002 | 0.17× | 0.052 | 0.003 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.37× | 0.008 | 0.003 | 0.41× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.26× | 0.015 | 0.004 | 0.30× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.015 | 0.025 | 1.64× | 0.033 | 0.029 | 0.89× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.004 | 0.002 | 0.46× | 0.007 | 0.002 | 0.29× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.068 | 0.001 | 0.02× | 0.081 | 0.001 | 0.02× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.016 | 0.003 | 0.17× | 0.021 | 0.003 | 0.15× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.029 | 0.003 | 0.10× | 0.045 | 0.004 | 0.10× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.021 | 0.002 | 0.10× | 0.031 | 0.003 | 0.09× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.026 | 0.002 | 0.08× | 0.053 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.334 | 0.098 | 0.29× | 3.422 | 0.981 | 0.29× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.439 | 0.407 | 0.93× | 4.345 | 4.183 | 0.96× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.086 | 0.005 | 0.05× | 0.955 | 0.040 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.451 | 0.015 | 0.03× | 4.518 | 0.120 | 0.03× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.018 | 0.009 | 0.48× | 0.041 | 0.016 | 0.38× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.247 | 0.002 | 0.01× | 0.371 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.039 | 0.021 | 0.54× | 0.072 | 0.032 | 0.44× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.030 | 0.002 | 0.07× | 0.049 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.417 | 0.271 | 0.65× | 4.084 | 2.613 | 0.64× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.030 | 0.016 | 0.53× | 0.064 | 0.045 | 0.70× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.008 | 0.002 | 0.25× | 0.010 | 0.003 | 0.26× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.421 | 0.018 | 0.04× | 3.957 | 0.130 | 0.03× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.007 | 0.003 | 0.44× | 0.008 | 0.004 | 0.50× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.091 | 0.692 | 7.61× | 0.339 | 0.955 | 2.82× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.058 | 0.003 | 0.05× | 0.383 | 0.005 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.034 | 0.262 | 7.61× | 0.093 | 0.263 | 2.84× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.026 | 0.013 | 0.50× | 0.028 | 0.015 | 0.52× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.874 | 6.356 | 7.27× | 9.030 | 78.180 | 8.66× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 1.422 | 6.896 | 4.85× | 19.463 | 81.382 | 4.18× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.727 | 0.596 | 0.82× | 3.842 | 4.507 | 1.17× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.465 | 0.464 | 1.00× | 2.983 | 2.762 | 0.93× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.419 | 0.419 | 1.00× | 3.106 | 2.773 | 0.89× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.031 | – | – | 0.166 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.175 | 0.036 | 0.21× | 0.121 | 0.024 | 0.20× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.009 | 0.004 | 0.50× | 0.006 | 0.003 | 0.44× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.347 | – | – | 1.888 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.759 | 0.194 | 0.26× | 4.937 | 1.109 | 0.22× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.354 | 0.031 | 0.09× | 2.327 | 0.180 | 0.08× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.073 | 0.026 | 0.35× | 0.380 | 0.137 | 0.36× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.066 | 0.355 | 5.41× | 0.331 | 0.689 | 2.08× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.089 | 0.050 | 0.56× | 0.665 | 0.473 | 0.71× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.035 | 0.341 | 9.73× | 0.207 | 0.401 | 1.94× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.055 | 0.433 | 7.88× | 0.384 | 0.620 | 1.61× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.033 | 0.204 | 6.23× | 0.189 | 0.335 | 1.77× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.035 | 0.010 | 0.29× | 0.162 | 0.080 | 0.49× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.170 | 0.601 | 3.54× | 1.088 | 1.291 | 1.19× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.045 | 0.375 | 8.37× | 0.233 | 0.381 | 1.64× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.090 | 0.624 | 6.96× | 0.425 | 0.900 | 2.12× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.047 | 0.562 | 11.99× | 0.323 | 0.645 | 2.00× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.061 | 0.250 | 4.08× | 0.394 | 0.543 | 1.38× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.058 | 0.015 | 0.26× | 0.546 | 0.155 | 0.28× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.030 | 0.006 | 0.20× | 0.236 | 0.038 | 0.16× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.054 | 0.033 | 0.62× | 0.440 | 0.304 | 0.69× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.055 | 0.015 | 0.27× | 0.496 | 0.132 | 0.27× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.047 | 0.022 | 0.47× | 0.524 | 0.262 | 0.50× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.315 | 6.065 | 19.23× | 3.200 | 50.638 | 15.83× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.342 | 5.788 | 16.92× | 3.571 | 53.240 | 14.91× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.323 | 5.802 | 17.95× | 5.013 | 52.727 | 10.52× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.486 | 5.986 | 12.31× | 4.850 | 54.818 | 11.30× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.314 | 5.858 | 18.63× | 3.348 | 56.260 | 16.80× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.324 | 5.745 | 17.71× | 3.361 | 56.567 | 16.83× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.010 | 0.003 | 0.30× | 0.012 | 0.003 | 0.23× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.029 | 0.014 | 0.48× | 0.162 | 0.104 | 0.64× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.076 | 0.528 | 6.92× | 0.077 | 0.463 | 5.99× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.016 | 0.013 | 0.82× | 0.111 | 0.098 | 0.88× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.027 | 0.008 | 0.30× | 0.206 | 0.050 | 0.24× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.009 | 0.002 | 0.25× | 0.011 | 0.003 | 0.24× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.010 | 0.002 | 0.22× | 0.014 | 0.004 | 0.27× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.460 | 0.015 | 0.03× | 4.537 | 0.122 | 0.03× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.045 | 0.005 | 0.11× | 0.050 | 0.005 | 0.10× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.007 | 0.005 | 0.71× | 0.010 | 0.004 | 0.38× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.739 | 0.337 | 0.46× | 6.851 | 3.471 | 0.51× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.066 | 0.045 | 0.68× | 0.111 | 0.070 | 0.63× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.78× | 0.003 | 0.003 | 0.82× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.711 | 0.017 | 0.02× | 7.938 | 0.102 | 0.01× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.004 | 0.002 | 0.51× | 0.007 | 0.004 | 0.63× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.005 | 0.003 | 0.54× | 0.007 | 0.005 | 0.67× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.123 | 0.039 | 0.32× | 1.275 | 0.354 | 0.28× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.049 | 0.007 | 0.14× | 0.453 | 0.044 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.058 | 0.010 | 0.17× | 0.511 | 0.078 | 0.15× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.010 | 0.002 | 0.21× | 0.014 | 0.004 | 0.28× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.011 | 0.004 | 0.33× | 0.014 | 0.005 | 0.37× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.236 | 0.006 | 0.02× | 2.311 | 0.038 | 0.02× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.063 | 0.055 | 0.03× | 2.068 | 0.053 | 0.03× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.261 | 0.333 | 1.28× | 3.544 | 4.227 | 1.19× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.214 | – | – | 2.429 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.142 | 1.676 | 11.80× | 1.500 | 11.539 | 7.69× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.256 | 0.505 | 1.97× | 3.618 | 6.291 | 1.74× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.137 | 1.631 | 11.89× | 1.556 | 11.630 | 7.47× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.123 | 1.698 | 13.82× | 1.456 | 11.626 | 7.99× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.141 | 1.649 | 11.69× | 1.484 | 11.532 | 7.77× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.023 | 0.476 | 20.88× | 0.239 | 3.463 | 14.49× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.010 | 0.059 | 6.03× | 0.048 | 0.057 | 1.18× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.278 | 0.315 | 1.13× | 3.826 | 4.827 | 1.26× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 64.880 | – | – | 262.4 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.025 | 0.187 | 7.55× | 0.023 | 0.178 | 7.64× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.608 | 2.286 | 1.42× | 22.029 | 24.704 | 1.12× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.041 | 0.029 | 0.70× | 0.593 | 0.319 | 0.54× | 0.0e+00 |
| `statistics.compare` | timed alone | 33.447 | – | – | 287.5 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.016 | 0.182 | 11.16× | 0.017 | 0.186 | 10.62× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 4.131 | – | – | 47.442 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.104 | 0.493 | 4.72× | 0.244 | 1.735 | 7.11× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.655 | 1.597 | 2.44× | 7.509 | 14.773 | 1.97× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.057 | 0.029 | 0.52× | 1.078 | 0.348 | 0.32× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.930 | 3.128 | 3.36× | 15.173 | 39.246 | 2.59× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.638 | 2.212 | 3.47× | 8.814 | 25.081 | 2.85× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.014 | 0.058 | 4.17× | 0.259 | 0.510 | 1.97× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 35.538 | – | – | 257.4 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.074 | 0.284 | 3.83× | 0.301 | 0.682 | 2.27× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.012 | 0.085 | 6.97× | 0.008 | 0.084 | 10.34× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.048 | – | – | 20.591 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.039 | 0.442 | 11.22× | 0.046 | 0.443 | 9.73× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.090 | 0.470 | 5.25× | 0.394 | 1.035 | 2.63× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.395 | 1.240 | 3.14× | 3.328 | 10.524 | 3.16× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.011 | 0.003 | 0.23× | 0.014 | 0.003 | 0.22× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.014 | 0.004 | 0.28× | 0.019 | 0.006 | 0.34× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 0.994 | 0.782 | 0.79× | 9.970 | 7.606 | 0.76× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.011 | 0.005 | 0.46× | 0.073 | 0.027 | 0.37× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.250 | 0.138 | 0.55× | 2.361 | 1.472 | 0.62× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 0.508 | 4.074 | 8.03× | 5.027 | 43.540 | 8.66× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.218 | 24.707 | 20.28× | 2.159 | 96.273 | 44.59× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 0.867 | 9.915 | 11.43× | 8.187 | 109.0 | 13.31× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 2.484 | 20.468 | 8.24× | 22.686 | 200.1 | 8.82× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.126 | – | – | 1.369 | – | – | – |
| `text.citation_recall` | timed alone | 0.092 | – | – | 1.057 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.124 | 0.003 | 0.03× | 0.908 | 0.010 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.034 | 0.051 | 1.51× | 0.300 | 0.536 | 1.78× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.039 | 0.011 | 0.28× | 0.330 | 0.066 | 0.20× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.148 | 0.158 | 1.07× | 1.678 | 2.259 | 1.35× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.328 | 0.096 | 0.29× | 0.854 | 0.234 | 0.27× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.037 | 0.023 | 0.62× | 0.172 | 0.146 | 0.85× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.236 | – | – | 2.113 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.199 | 0.352 | 1.77× | 1.785 | 3.596 | 2.02× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.037 | 0.085 | 2.29× | 0.085 | 0.667 | 7.86× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.020 | 0.016 | 0.82× | 0.164 | 0.129 | 0.78× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.353 | 0.119 | 0.34× | 3.695 | 1.204 | 0.33× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.183 | 0.084 | 0.46× | 1.708 | 0.909 | 0.53× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.027 | 0.010 | 0.35× | 0.243 | 0.073 | 0.30× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.044 | – | – | 0.371 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 0.986 | 2.336 | 2.37× | 9.821 | 25.806 | 2.63× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.119 | 0.099 | 0.83× | 1.160 | 0.958 | 0.83× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.087 | 0.576 | 6.65× | 0.229 | 0.807 | 3.52× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.046 | 0.010 | 0.21× | 0.407 | 0.151 | 0.37× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.068 | 0.080 | 1.17× | 0.158 | 0.338 | 2.13× | 2.9e-15 |
| `text.mauve` | timed alone | 3.042 | – | – | 538.0 | – | – | – |
| `text.meteor` | NLTK | 0.613 | 3.960 | 6.46× | 6.737 | 42.165 | 6.26× | 0.0e+00 |
| `text.model_score` | timed alone | 0.013 | – | – | 0.067 | – | – | – |
| `text.moverscore` | POT | 0.764 | 1.102 | 1.44× | 8.295 | 11.535 | 1.39× | 0.0e+00 |
| `text.perplexity` | NumPy formula | 0.045 | 0.011 | 0.23× | 0.326 | 0.064 | 0.20× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.052 | 0.003 | 0.05× | 0.429 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.260 | – | – | 2.518 | – | – | – |
| `text.rouge_1` | rouge-score | 0.419 | 2.298 | 5.49× | 4.456 | 22.709 | 5.10× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.383 | 2.445 | 6.39× | 4.445 | 21.644 | 4.87× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 0.627 | 2.782 | 4.43× | 4.442 | 28.232 | 6.35× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 0.619 | 4.269 | 6.89× | 7.918 | 44.688 | 5.64× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.014 | 0.004 | 0.30× | 0.031 | 0.007 | 0.23× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 0.521 | 9.375 | 17.98× | 6.838 | 250.6 | 36.65× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.348 | – | – | 0.415 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 1.082 | 7.265 | 6.71× | 6.829 | 58.476 | 8.56× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.952 | 0.046 | 0.05× | 9.148 | 0.490 | 0.05× | 0.0e+00 |
| `text.ter` | sacreBLEU | 0.663 | 13.895 | 20.97× | 6.430 | 137.9 | 21.45× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.210 | – | – | 2.460 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.228 | – | – | 2.675 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.171 | 0.019 | 0.11× | 2.003 | 0.146 | 0.07× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.446 | – | – | 1.400 | – | – | – |
| `text.win_rate` | NumPy formula | 0.031 | 0.005 | 0.15× | 0.241 | 0.011 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.386 | 0.371 | 0.96× | 3.845 | 3.689 | 0.96× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.101 | 0.087 | 0.85× | 0.682 | 0.677 | 0.99× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.050 | 0.032 | 0.65× | 0.667 | 0.644 | 0.97× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.213 | 1.214 | 5.71× | 1.091 | 2.865 | 2.63× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.054 | 0.032 | 0.59× | 0.694 | 0.656 | 0.95× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.049 | 0.023 | 0.47× | 0.672 | 0.582 | 0.87× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.032 | 0.022 | 0.69× | 0.610 | 0.582 | 0.95× | 0.0e+00 |
