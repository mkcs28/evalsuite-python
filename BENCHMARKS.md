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
EvalSuite 0.6.0 benchmarks | Python 3.13.18 | NumPy 2.5.4 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 12/12 | 9.75× | 1.62×–87.97× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 24/24 | 3.89× | 1.04×–17.26× |
| LLM evaluation | 6 | 18 | 18/18 | 18/18 | 4.08× | 1.08×–9.12× |
| LLM systems | 6 | 18 | 18/18 | 18/18 | 1.80× | 1.01×–6.24× |
| Segmentation and object detection | 3 | 9 | 9/9 | 9/9 | 3.12× | 1.21×–14.64× |
| Overall | 27 | 81 | 81/81 | 81/81 | 3.70× | 1.01×–87.97× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.066 | 1.012 | 15.40× | 0.02 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 0.602 | 10.696 | 17.76× | 0.77 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 10.194 | 107.441 | 10.54× | 7.63 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.040 | 0.044 | 1.08× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.156 | 0.324 | 2.08× | 0.29 | 0.69 | 3.3e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 1.283 | 2.942 | 2.29× | 2.53 | 6.32 | 1.8e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.170 | 7.165 | 42.05× | 0.03 | 0.04 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 1.067 | 93.859 | 87.97× | 1.69 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 11.194 | 818.474 | 73.12× | 16.28 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.119 | 1.111 | 9.32× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 11.305 | 24.578 | 2.17× | 8.39 | 7.63 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 175.408 | 292.753 | 1.67× | 83.93 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.376 | 2.074 | 5.52× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 18.746 | 105.116 | 5.61× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 234.689 | 938.016 | 4.00× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.084 | 0.381 | 4.53× | 0.02 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 0.633 | 1.271 | 2.01× | 0.76 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 4.266 | 4.443 | 1.04× | 7.63 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.167 | 2.718 | 16.31× | 0.02 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 2.110 | 36.422 | 17.26× | 0.76 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 18.753 | 309.706 | 16.51× | 7.63 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.104 | 0.635 | 6.12× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 9.897 | 16.321 | 1.65× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 149.297 | 230.123 | 1.54× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 7.449 | 15.531 | 2.08× | 0.54 | 1.36 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 35.679 | 72.201 | 2.02× | 1.18 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 268.502 | 628.832 | 2.34× | 7.39 | 34.54 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.033 | 0.037 | 1.14× | 0.04 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 3.626 | 3.987 | 1.10× | 3.82 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 69.376 | 81.231 | 1.17× | 38.15 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.068 | 0.486 | 7.13× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 0.932 | 2.255 | 2.42× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 7.413 | 12.004 | 1.62× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.142 | 1.190 | 8.41× | 0.01 | 0.03 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 11.917 | 57.654 | 4.84× | 0.49 | 3.12 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 121.749 | 546.680 | 4.49× | 5.36 | 31.01 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.158 | 2.312 | 14.64× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 2.910 | 23.249 | 7.99× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 34.234 | 200.766 | 5.86× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.149 | 0.262 | 1.76× | 0.27 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 3.327 | 6.436 | 1.93× | 0.29 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 11.038 | 13.370 | 1.21× | 0.28 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.016 | 0.173 | 10.71× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.016 | 0.204 | 12.68× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.016 | 0.186 | 11.97× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.078 | 0.430 | 5.48× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 8.596 | 24.076 | 2.80× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 152.121 | 323.012 | 2.12× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.069 | 0.425 | 6.14× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.696 | 1.618 | 2.33× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 7.663 | 11.608 | 1.51× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.104 | 0.252 | 2.42× | 0.00 | 0.01 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 9.462 | 22.816 | 2.41× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 88.856 | 224.701 | 2.53× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 41.859 | 50.619 | 1.21× | 0.38 | 0.26 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 385.312 | 505.097 | 1.31× | 1.75 | 2.05 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 1.974 | 2.882 | 1.46× | 0.08 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.396 | 1.781 | 4.50× | 0.06 | 0.08 | 1.1e-16 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 12.279 | 23.578 | 1.92× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 165.971 | 265.258 | 1.60× | 54.36 | 76.30 | 1.1e-16 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.112 | 0.699 | 6.24× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.751 | 1.691 | 2.25× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 10.764 | 12.140 | 1.13× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.085 | 0.298 | 3.52× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 7.980 | 21.519 | 2.70× | 0.71 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 79.149 | 209.216 | 2.64× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.040 | 0.058 | 1.44× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 0.908 | 1.393 | 1.53× | 0.76 | 0.77 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 13.680 | 15.377 | 1.12× | 7.63 | 7.63 | 0.0e+00 |
| systems: maintainability index (100 programs) | 100,000 | radon | 8.916 | 9.618 | 1.08× | 0.22 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 84.490 | 93.367 | 1.11× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.432 | 0.437 | 1.01× | 0.05 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 0.235 | 1.539 | 6.55× | 0.01 | 0.25 | 0.0e+00 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 20.006 | 150.135 | 7.50× | 1.14 | 21.83 | 2.8e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 215.230 | 1962.790 | 9.12× | 11.44 | 211.15 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.067 | 0.365 | 5.41× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 6.103 | 34.618 | 5.67× | 0.58 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 71.025 | 363.420 | 5.12× | 5.82 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.123 | 0.629 | 5.13× | 0.01 | 0.02 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 11.305 | 57.278 | 5.07× | 0.63 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 154.609 | 608.341 | 3.93× | 6.95 | 6.56 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.6.0, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 5.79× | 1.18×–30.42× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 42/42 | 6.06× | 1.01×–35.83× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 46/46 | 7.54× | 1.17×–45.14× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 14/14 | 2.91× | 1.11×–7.62× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 6.20× | 1.19×–20.70× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 172/172 | 5.99× | 1.01×–45.14× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.007 | 0.002 | 0.25× | 0.006 | 0.002 | 0.28× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.009 | 0.003 | 0.33× | 0.009 | 0.005 | 0.57× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 0.896 | 2.648 | 2.95× | 10.955 | 31.395 | 2.87× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.179 | – | – | 1.758 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.095 | 0.010 | 0.10× | 0.945 | 0.066 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.017 | 0.002 | 0.09× | 0.025 | 0.002 | 0.07× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.73× | 0.003 | 0.003 | 0.80× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.134 | – | – | 1.613 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.016 | 0.002 | 0.13× | 0.023 | 0.003 | 0.13× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.623 | 0.472 | 0.76× | 7.263 | 5.497 | 0.76× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.478 | 3.747 | 7.83× | 4.323 | 27.208 | 6.29× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 0.942 | 6.434 | 6.83× | 8.269 | 50.887 | 6.15× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.331 | 0.457 | 1.38× | 2.762 | 4.520 | 1.64× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.048 | 0.527 | 10.94× | 0.301 | 3.260 | 10.84× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.867 | 2.009 | 2.32× | 9.793 | 16.115 | 1.65× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.355 | 1.908 | 5.38× | 3.139 | 11.767 | 3.75× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.055 | 0.912 | 16.53× | 0.398 | 5.319 | 13.36× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.066 | 1.490 | 22.73× | 0.335 | 9.399 | 28.03× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.283 | 0.482 | 1.70× | 2.651 | 4.712 | 1.78× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.378 | 2.513 | 6.65× | 3.222 | 13.620 | 4.23× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.084 | 2.032 | 24.15× | 0.366 | 10.385 | 28.39× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.562 | 3.276 | 5.83× | 5.207 | 26.363 | 5.06× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.069 | 1.860 | 26.84× | 0.374 | 10.553 | 28.21× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.530 | 3.282 | 6.19× | 4.654 | 21.808 | 4.69× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.359 | 3.369 | 9.39× | 3.243 | 21.903 | 6.75× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.071 | 2.024 | 28.50× | 0.358 | 10.475 | 29.24× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.376 | 2.431 | 6.46× | 3.206 | 13.368 | 4.17× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.367 | 2.436 | 6.63× | 3.122 | 13.657 | 4.37× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 10.462 | 22.040 | 2.11× | 110.9 | 177.3 | 1.60× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.064 | 1.821 | 28.64× | 0.344 | 10.460 | 30.42× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.566 | 2.286 | 4.04× | 6.387 | 19.447 | 3.04× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.059 | 0.034 | 0.58× | 0.354 | 0.264 | 0.75× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.058 | 1.735 | 29.97× | 0.340 | 12.192 | 35.83× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.056 | 1.710 | 30.53× | 0.333 | 11.891 | 35.74× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.132 | 0.017 | 0.13× | 1.245 | 0.121 | 0.10× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.063 | 1.815 | 28.68× | 0.343 | 10.186 | 29.69× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.065 | 1.781 | 27.58× | 0.346 | 10.336 | 29.89× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.057 | 0.034 | 0.59× | 0.334 | 0.255 | 0.76× | 0.0e+00 |
| `code.code_complexity` | radon | 0.694 | 0.769 | 1.11× | 6.604 | 7.333 | 1.11× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 4.563 | 5.677 | 1.24× | 40.632 | 51.579 | 1.27× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.017 | 0.001 | 0.06× | 0.014 | 0.001 | 0.09× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.018 | 0.013 | 0.74× | 0.134 | 0.110 | 0.82× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.005 | 0.003 | 0.73× | 0.003 | 0.003 | 0.83× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.063 | 0.012 | 0.20× | 0.642 | 0.109 | 0.17× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.024 | 0.002 | 0.08× | 0.032 | 0.003 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.072 | 0.007 | 0.09× | 0.685 | 0.046 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.009 | 0.001 | 0.16× | 0.013 | 0.001 | 0.10× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 1.705 | 1.021 | 0.60× | 18.031 | 10.944 | 0.61× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.330 | 0.260 | 0.79× | 3.435 | 2.747 | 0.80× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 1.416 | 13.039 | 9.21× | 15.347 | 68.368 | 4.45× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.036 | 0.020 | 0.54× | 7.805 | 4.233 | 0.54× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 1.687 | 12.858 | 7.62× | 14.691 | 69.885 | 4.76× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.012 | 0.002 | 0.17× | 0.055 | 0.003 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.36× | 0.007 | 0.004 | 0.50× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.27× | 0.016 | 0.004 | 0.27× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.018 | 0.024 | 1.37× | 0.023 | 0.027 | 1.17× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.004 | 0.002 | 0.48× | 0.007 | 0.002 | 0.28× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.063 | 0.001 | 0.02× | 0.082 | 0.001 | 0.01× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.015 | 0.003 | 0.17× | 0.023 | 0.003 | 0.14× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.028 | 0.003 | 0.11× | 0.044 | 0.005 | 0.11× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.020 | 0.002 | 0.10× | 0.030 | 0.003 | 0.10× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.027 | 0.002 | 0.07× | 0.052 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.333 | 0.095 | 0.28× | 3.360 | 0.974 | 0.29× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.421 | 0.392 | 0.93× | 4.377 | 4.165 | 0.95× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.084 | 0.004 | 0.05× | 0.938 | 0.041 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.444 | 0.015 | 0.03× | 4.485 | 0.117 | 0.03× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.017 | 0.008 | 0.48× | 0.041 | 0.013 | 0.33× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.238 | 0.002 | 0.01× | 0.376 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.039 | 0.021 | 0.53× | 0.070 | 0.032 | 0.46× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.029 | 0.002 | 0.07× | 0.044 | 0.002 | 0.05× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.412 | 0.268 | 0.65× | 4.029 | 2.514 | 0.62× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.030 | 0.016 | 0.52× | 0.065 | 0.046 | 0.71× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.008 | 0.002 | 0.25× | 0.010 | 0.003 | 0.27× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.393 | 0.017 | 0.04× | 3.840 | 0.135 | 0.04× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.007 | 0.003 | 0.44× | 0.008 | 0.004 | 0.49× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.087 | 0.663 | 7.62× | 0.353 | 1.026 | 2.91× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.055 | 0.003 | 0.05× | 0.409 | 0.006 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.035 | 0.254 | 7.21× | 0.104 | 0.284 | 2.72× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.026 | 0.012 | 0.47× | 0.028 | 0.014 | 0.51× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.876 | 6.060 | 6.92× | 9.219 | 58.256 | 6.32× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 1.427 | 6.496 | 4.55× | 14.844 | 64.291 | 4.33× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.350 | 0.434 | 1.24× | 3.901 | 4.567 | 1.17× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.288 | 0.255 | 0.88× | 2.881 | 2.673 | 0.93× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.522 | 0.405 | 0.77× | 2.922 | 2.689 | 0.92× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.038 | – | – | 0.171 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.159 | 0.032 | 0.20× | 0.120 | 0.023 | 0.19× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.008 | 0.004 | 0.47× | 0.006 | 0.003 | 0.45× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.282 | – | – | 1.752 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.693 | 0.152 | 0.22× | 4.794 | 1.042 | 0.22× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.363 | 0.032 | 0.09× | 2.263 | 0.177 | 0.08× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.061 | 0.023 | 0.38× | 0.384 | 0.140 | 0.36× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.069 | 0.297 | 4.31× | 0.318 | 0.771 | 2.43× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.077 | 0.043 | 0.56× | 0.616 | 0.479 | 0.78× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.031 | 0.203 | 6.60× | 0.203 | 0.396 | 1.95× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.047 | 0.227 | 4.85× | 0.363 | 0.560 | 1.54× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.027 | 0.148 | 5.51× | 0.188 | 0.331 | 1.76× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.029 | 0.010 | 0.35× | 0.154 | 0.076 | 0.50× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.146 | 0.307 | 2.10× | 1.099 | 1.295 | 1.18× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.038 | 0.199 | 5.25× | 0.216 | 0.392 | 1.82× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.075 | 0.368 | 4.92× | 0.422 | 0.868 | 2.06× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.036 | 0.238 | 6.70× | 0.310 | 0.624 | 2.01× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.059 | 0.263 | 4.48× | 0.375 | 0.527 | 1.41× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.078 | 0.023 | 0.30× | 0.543 | 0.152 | 0.28× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.040 | 0.009 | 0.22× | 0.213 | 0.036 | 0.17× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.080 | 0.051 | 0.63× | 0.407 | 0.296 | 0.73× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.076 | 0.025 | 0.33× | 0.492 | 0.128 | 0.26× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.064 | 0.029 | 0.45× | 0.508 | 0.259 | 0.51× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.457 | 5.490 | 12.01× | 3.084 | 51.494 | 16.69× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.330 | 5.414 | 16.41× | 3.423 | 51.143 | 14.94× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.303 | 5.414 | 17.87× | 3.140 | 52.576 | 16.74× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.449 | 5.446 | 12.12× | 4.455 | 52.123 | 11.70× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.301 | 5.437 | 18.05× | 3.107 | 51.547 | 16.59× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.302 | 5.435 | 17.98× | 3.108 | 52.446 | 16.87× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.010 | 0.002 | 0.20× | 0.012 | 0.003 | 0.21× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.019 | 0.013 | 0.71× | 0.167 | 0.105 | 0.63× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.071 | 0.418 | 5.90× | 0.076 | 0.439 | 5.76× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.015 | 0.013 | 0.83× | 0.110 | 0.095 | 0.87× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.025 | 0.008 | 0.30× | 0.208 | 0.050 | 0.24× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.008 | 0.002 | 0.25× | 0.011 | 0.003 | 0.24× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.009 | 0.002 | 0.22× | 0.015 | 0.004 | 0.26× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.425 | 0.014 | 0.03× | 4.346 | 0.113 | 0.03× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.042 | 0.004 | 0.09× | 0.046 | 0.005 | 0.10× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.007 | 0.002 | 0.30× | 0.009 | 0.004 | 0.38× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.667 | 0.328 | 0.49× | 6.524 | 3.336 | 0.51× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.062 | 0.041 | 0.66× | 0.099 | 0.066 | 0.66× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.75× | 0.003 | 0.003 | 0.85× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.628 | 0.015 | 0.02× | 6.870 | 0.149 | 0.02× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.004 | 0.002 | 0.55× | 0.009 | 0.006 | 0.69× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.004 | 0.003 | 0.61× | 0.009 | 0.007 | 0.75× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.120 | 0.038 | 0.32× | 1.844 | 0.599 | 0.33× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.045 | 0.007 | 0.15× | 0.751 | 0.068 | 0.09× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.055 | 0.010 | 0.18× | 0.784 | 0.121 | 0.15× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.009 | 0.002 | 0.23× | 0.020 | 0.007 | 0.34× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.011 | 0.003 | 0.31× | 0.029 | 0.014 | 0.47× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.225 | 0.005 | 0.02× | 3.363 | 0.036 | 0.01× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 1.941 | 0.052 | 0.03× | 1.959 | 0.053 | 0.03× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.238 | 0.306 | 1.29× | 3.320 | 3.963 | 1.19× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.207 | – | – | 2.216 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.133 | 1.562 | 11.79× | 1.341 | 10.438 | 7.78× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.242 | 0.473 | 1.96× | 3.389 | 6.308 | 1.86× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.131 | 1.522 | 11.60× | 1.402 | 10.415 | 7.43× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.116 | 1.566 | 13.52× | 1.337 | 10.825 | 8.10× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.138 | 1.516 | 10.97× | 1.374 | 10.899 | 7.93× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.022 | 0.456 | 20.70× | 0.219 | 3.110 | 14.22× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.010 | 0.051 | 4.84× | 0.031 | 0.056 | 1.78× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.258 | 0.295 | 1.14× | 3.244 | 3.622 | 1.12× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 64.144 | – | – | 236.4 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.025 | 0.179 | 7.24× | 0.022 | 0.203 | 9.30× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.528 | 1.949 | 1.28× | 21.988 | 22.265 | 1.01× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.041 | 0.029 | 0.70× | 0.371 | 0.224 | 0.60× | 0.0e+00 |
| `statistics.compare` | timed alone | 30.852 | – | – | 234.4 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.015 | 0.179 | 11.61× | 0.016 | 0.193 | 11.96× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 3.694 | – | – | 41.485 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.092 | 0.455 | 4.93× | 0.241 | 1.733 | 7.20× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.471 | 1.304 | 2.77× | 5.524 | 11.566 | 2.09× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.056 | 0.027 | 0.49× | 0.606 | 0.259 | 0.43× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.740 | 2.725 | 3.68× | 10.119 | 32.823 | 3.24× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.511 | 1.964 | 3.84× | 5.970 | 20.664 | 3.46× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.013 | 0.054 | 4.07× | 0.103 | 0.278 | 2.71× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 30.686 | – | – | 280.5 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.072 | 0.266 | 3.70× | 0.500 | 1.090 | 2.18× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.012 | 0.082 | 6.95× | 0.008 | 0.081 | 10.03× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 1.915 | – | – | 22.668 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.037 | 0.421 | 11.31× | 0.043 | 0.521 | 12.11× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.084 | 0.422 | 5.04× | 0.728 | 1.640 | 2.25× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.283 | 1.189 | 4.20× | 4.304 | 10.594 | 2.46× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.011 | 0.002 | 0.23× | 0.013 | 0.003 | 0.23× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.013 | 0.004 | 0.29× | 0.019 | 0.006 | 0.33× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 0.964 | 0.757 | 0.79× | 9.249 | 7.497 | 0.81× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.010 | 0.005 | 0.49× | 0.072 | 0.026 | 0.35× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.236 | 0.131 | 0.56× | 2.360 | 1.443 | 0.61× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 0.546 | 3.726 | 6.82× | 6.929 | 45.166 | 6.52× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.145 | 23.985 | 20.94× | 2.106 | 95.065 | 45.14× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 1.205 | 9.340 | 7.75× | 14.409 | 113.2 | 7.86× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 12.169 | 17.951 | 1.48× | 148.7 | 240.2 | 1.62× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.121 | – | – | 1.617 | – | – | – |
| `text.citation_recall` | timed alone | 0.093 | – | – | 1.374 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.116 | 0.003 | 0.03× | 0.928 | 0.010 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.033 | 0.050 | 1.50× | 0.317 | 0.543 | 1.71× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.039 | 0.011 | 0.28× | 0.334 | 0.065 | 0.19× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.417 | 0.158 | 0.38× | 4.510 | 3.054 | 0.68× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.313 | 0.083 | 0.26× | 0.818 | 0.219 | 0.27× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.034 | 0.023 | 0.67× | 0.201 | 0.164 | 0.82× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.222 | – | – | 1.987 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.192 | 0.340 | 1.77× | 1.772 | 3.552 | 2.00× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.034 | 0.081 | 2.37× | 0.091 | 0.622 | 6.83× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.019 | 0.016 | 0.82× | 0.148 | 0.115 | 0.78× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.347 | 0.113 | 0.33× | 3.724 | 1.166 | 0.31× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.178 | 0.077 | 0.43× | 1.745 | 0.950 | 0.54× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.025 | 0.009 | 0.35× | 0.246 | 0.074 | 0.30× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.043 | – | – | 0.377 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 0.953 | 2.256 | 2.37× | 9.721 | 23.320 | 2.40× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.114 | 0.096 | 0.84× | 1.143 | 0.955 | 0.83× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.087 | 0.528 | 6.06× | 0.226 | 0.849 | 3.76× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.046 | 0.010 | 0.22× | 0.416 | 0.156 | 0.37× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.067 | 0.079 | 1.17× | 0.159 | 0.353 | 2.22× | 2.9e-15 |
| `text.mauve` | timed alone | 2.694 | – | – | 469.5 | – | – | – |
| `text.meteor` | NLTK | 0.539 | 3.613 | 6.70× | 6.634 | 41.346 | 6.23× | 0.0e+00 |
| `text.model_score` | timed alone | 0.013 | – | – | 0.067 | – | – | – |
| `text.moverscore` | POT | 0.738 | 1.048 | 1.42× | 7.940 | 10.850 | 1.37× | 0.0e+00 |
| `text.perplexity` | NumPy formula | 0.039 | 0.010 | 0.27× | 0.326 | 0.066 | 0.20× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.050 | 0.003 | 0.05× | 0.440 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.265 | – | – | 2.593 | – | – | – |
| `text.rouge_1` | rouge-score | 0.294 | 2.193 | 7.47× | 3.645 | 21.590 | 5.92× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.288 | 2.253 | 7.82× | 3.453 | 21.665 | 6.27× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 0.301 | 2.756 | 9.15× | 3.485 | 27.371 | 7.85× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 0.555 | 3.802 | 6.85× | 6.332 | 40.481 | 6.39× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.014 | 0.004 | 0.27× | 0.032 | 0.008 | 0.23× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 0.493 | 8.800 | 17.85× | 12.642 | 232.6 | 18.40× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.328 | – | – | 0.424 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 0.633 | 5.252 | 8.30× | 7.373 | 57.861 | 7.85× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.863 | 0.044 | 0.05× | 8.318 | 0.508 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 0.768 | 13.413 | 17.46× | 7.676 | 132.0 | 17.20× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.212 | – | – | 2.351 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.235 | – | – | 2.692 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.174 | 0.018 | 0.11× | 2.057 | 0.148 | 0.07× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.449 | – | – | 1.392 | – | – | – |
| `text.win_rate` | NumPy formula | 0.029 | 0.004 | 0.15× | 0.233 | 0.011 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.377 | 0.354 | 0.94× | 3.850 | 3.754 | 0.98× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.102 | 0.083 | 0.81× | 0.677 | 0.645 | 0.95× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.050 | 0.030 | 0.61× | 0.654 | 0.618 | 0.94× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.227 | 1.088 | 4.80× | 1.043 | 2.842 | 2.72× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.052 | 0.031 | 0.59× | 0.685 | 0.633 | 0.92× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.049 | 0.023 | 0.46× | 0.681 | 0.586 | 0.86× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.031 | 0.022 | 0.70× | 0.601 | 0.589 | 0.98× | 0.0e+00 |
