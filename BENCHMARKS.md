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
EvalSuite 0.5.3 benchmarks | Python 3.13.18 | NumPy 2.5.4 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 11/12 | 5.95× | 0.69×–37.77× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 16/24 | 1.84× | 0.22×–18.69× |
| LLM evaluation | 6 | 18 | 18/18 | 12/18 | 1.54× | 0.77×–10.61× |
| LLM systems | 6 | 18 | 18/18 | 15/18 | 1.69× | 0.88×–5.95× |
| Segmentation and object detection | 3 | 9 | 9/9 | 4/9 | 1.82× | 0.63×–14.85× |
| Overall | 27 | 81 | 81/81 | 58/81 | 2.06× | 0.22×–37.77× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.073 | 1.017 | 13.93× | 0.05 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 2.309 | 12.176 | 5.27× | 3.05 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 35.932 | 122.057 | 3.40× | 30.52 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.044 | 0.042 | 0.96× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.294 | 0.299 | 1.02× | 0.26 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 4.429 | 3.430 | 0.77× | 2.26 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.205 | 7.176 | 35.08× | 0.05 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 2.455 | 92.713 | 37.77× | 3.21 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 23.270 | 866.515 | 37.24× | 31.54 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.130 | 1.120 | 8.64× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 12.090 | 25.921 | 2.14× | 9.16 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 187.916 | 305.657 | 1.63× | 91.56 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.425 | 2.113 | 4.97× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 20.358 | 106.851 | 5.25× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 236.611 | 1085.832 | 4.59× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.195 | 0.419 | 2.14× | 0.05 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 2.069 | 1.014 | 0.49× | 3.05 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 26.241 | 5.708 | 0.22× | 30.52 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.257 | 4.796 | 18.69× | 0.05 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 7.636 | 37.485 | 4.91× | 3.05 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 122.873 | 353.920 | 2.88× | 30.52 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.121 | 0.699 | 5.78× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 10.587 | 17.095 | 1.61× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 167.295 | 238.083 | 1.42× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 18.640 | 16.799 | 0.90× | 0.54 | 1.36 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 71.504 | 73.813 | 1.03× | 1.15 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 684.665 | 660.799 | 0.97× | 7.00 | 34.44 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.038 | 0.044 | 1.15× | 0.05 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 3.724 | 4.317 | 1.16× | 4.58 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 71.309 | 92.360 | 1.30× | 45.78 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.071 | 0.489 | 6.84× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.980 | 2.251 | 1.14× | 2.29 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 27.335 | 18.826 | 0.69× | 22.89 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.141 | 1.500 | 10.61× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 11.197 | 59.094 | 5.28× | 0.49 | 3.11 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 129.496 | 614.724 | 4.75× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.167 | 2.478 | 14.85× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 3.059 | 23.589 | 7.71× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 35.701 | 235.116 | 6.59× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.815 | 0.512 | 0.63× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 8.914 | 6.393 | 0.72× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 18.815 | 13.576 | 0.72× | 0.15 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.195 | 0.182 | 0.93× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.193 | 0.180 | 0.93× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.200 | 0.180 | 0.90× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.475 | 0.446 | 0.94× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 26.267 | 25.270 | 0.96× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 342.336 | 325.801 | 0.95× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.074 | 0.448 | 6.08× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.697 | 1.672 | 2.40× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 8.241 | 11.862 | 1.44× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.107 | 0.260 | 2.43× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 9.313 | 23.184 | 2.49× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 93.933 | 223.482 | 2.38× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 42.628 | 51.913 | 1.22× | 0.38 | 0.26 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 413.797 | 507.778 | 1.23× | 1.75 | 2.05 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 1.950 | 2.974 | 1.53× | 0.08 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.246 | 1.124 | 4.57× | 0.06 | 0.08 | 1.1e-16 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 11.003 | 21.023 | 1.91× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 154.114 | 251.822 | 1.63× | 54.36 | 76.30 | 1.1e-16 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.077 | 0.460 | 5.95× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.694 | 1.762 | 2.54× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 10.888 | 12.854 | 1.18× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.089 | 0.259 | 2.90× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 7.467 | 22.586 | 3.02× | 0.71 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 78.402 | 213.173 | 2.72× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.038 | 0.035 | 0.93× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 1.575 | 1.422 | 0.90× | 0.77 | 0.77 | 2.2e-15 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 18.031 | 15.863 | 0.88× | 7.63 | 7.63 | 2.2e-15 |
| systems: maintainability index (100 programs) | 100,000 | radon | 8.802 | 10.027 | 1.14× | 0.22 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 89.291 | 95.989 | 1.08× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.459 | 0.470 | 1.02× | 0.05 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 1.636 | 1.403 | 0.86× | 0.10 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 167.845 | 175.486 | 1.05× | 0.73 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 1728.817 | 1993.131 | 1.15× | 7.28 | 211.52 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.350 | 0.365 | 1.04× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 32.509 | 35.895 | 1.10× | 0.12 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 319.055 | 358.623 | 1.12× | 1.15 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.661 | 0.565 | 0.85× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 69.925 | 60.174 | 0.86× | 0.12 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 716.028 | 612.508 | 0.86× | 1.16 | 6.56 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.5.3, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 3.98× | 1.17×–13.70× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 23/42 | 1.84× | 0.46×–13.55× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 37/46 | 2.34× | 0.07×–43.71× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 12/14 | 2.31× | 0.37×–7.59× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 14/18 | 4.88× | 0.56×–22.16× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 138/172 | 2.80× | 0.07×–43.71× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.008 | 0.002 | 0.23× | 0.013 | 0.003 | 0.27× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.009 | 0.003 | 0.33× | 0.017 | 0.006 | 0.36× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 0.853 | 2.592 | 3.04× | 10.631 | 32.188 | 3.03× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.180 | – | – | 1.793 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.090 | 0.010 | 0.11× | 0.989 | 0.064 | 0.06× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.025 | 0.002 | 0.06× | 0.036 | 0.002 | 0.05× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.79× | 0.004 | 0.003 | 0.76× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.130 | – | – | 1.828 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.019 | 0.002 | 0.10× | 0.027 | 0.003 | 0.10× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.605 | 0.461 | 0.76× | 7.406 | 5.588 | 0.75× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.516 | 3.645 | 7.07× | 6.724 | 34.149 | 5.08× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 0.997 | 6.445 | 6.47× | 8.890 | 52.460 | 5.90× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.387 | 0.461 | 1.19× | 3.373 | 4.741 | 1.41× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.131 | 0.531 | 4.06× | 1.206 | 3.352 | 2.78× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.924 | 1.998 | 2.16× | 11.414 | 17.481 | 1.53× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.487 | 1.916 | 3.93× | 5.316 | 12.675 | 2.38× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.107 | 0.913 | 8.51× | 0.965 | 5.843 | 6.05× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.147 | 1.407 | 9.55× | 1.352 | 9.983 | 7.38× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.339 | 0.480 | 1.42× | 3.387 | 4.975 | 1.47× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.538 | 2.440 | 4.53× | 4.825 | 14.652 | 3.04× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.162 | 1.993 | 12.28× | 1.492 | 11.107 | 7.45× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.598 | 3.254 | 5.44× | 6.203 | 28.796 | 4.64× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.149 | 1.866 | 12.48× | 1.316 | 10.970 | 8.34× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.663 | 3.294 | 4.97× | 6.721 | 29.073 | 4.33× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.507 | 3.361 | 6.63× | 5.266 | 24.819 | 4.71× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.143 | 1.961 | 13.70× | 1.963 | 11.806 | 6.01× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.507 | 2.405 | 4.74× | 5.750 | 15.522 | 2.70× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.488 | 2.416 | 4.95× | 5.370 | 15.632 | 2.91× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 10.563 | 22.947 | 2.17× | 142.9 | 215.1 | 1.51× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.150 | 2.020 | 13.44× | 1.900 | 11.715 | 6.17× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.710 | 2.548 | 3.59× | 8.855 | 21.593 | 2.44× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.133 | 0.035 | 0.26× | 1.852 | 0.347 | 0.19× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.146 | 1.935 | 13.30× | 1.885 | 13.401 | 7.11× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.148 | 1.826 | 12.37× | 1.892 | 13.739 | 7.26× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.177 | 0.016 | 0.09× | 1.906 | 0.154 | 0.08× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.148 | 2.001 | 13.55× | 1.905 | 11.463 | 6.02× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.147 | 1.999 | 13.55× | 1.905 | 11.775 | 6.18× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.221 | 0.046 | 0.21× | 1.913 | 0.352 | 0.18× | 0.0e+00 |
| `code.code_complexity` | radon | 1.154 | 0.754 | 0.65× | 6.820 | 7.572 | 1.11× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 4.624 | 6.097 | 1.32× | 42.178 | 53.451 | 1.27× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.024 | 0.001 | 0.05× | 0.018 | 0.001 | 0.07× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.025 | 0.017 | 0.67× | 0.130 | 0.103 | 0.80× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.004 | 0.003 | 0.74× | 0.003 | 0.003 | 0.77× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.092 | 0.017 | 0.18× | 0.637 | 0.107 | 0.17× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.035 | 0.003 | 0.08× | 0.033 | 0.003 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.104 | 0.010 | 0.10× | 0.679 | 0.045 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.018 | 0.001 | 0.07× | 0.015 | 0.001 | 0.08× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 1.752 | 1.027 | 0.59× | 17.500 | 10.635 | 0.61× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.343 | 0.262 | 0.76× | 3.372 | 2.713 | 0.80× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 1.740 | 13.588 | 7.81× | 21.794 | 70.750 | 3.25× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.042 | 0.021 | 0.50× | 8.855 | 6.105 | 0.69× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 2.459 | 14.921 | 6.07× | 24.935 | 73.324 | 2.94× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.015 | 0.002 | 0.15× | 0.057 | 0.003 | 0.04× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.35× | 0.009 | 0.003 | 0.39× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.013 | 0.003 | 0.22× | 0.017 | 0.004 | 0.22× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.023 | 0.026 | 1.09× | 0.032 | 0.029 | 0.89× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.006 | 0.002 | 0.30× | 0.008 | 0.002 | 0.26× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.076 | 0.001 | 0.02× | 0.089 | 0.001 | 0.01× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.026 | 0.004 | 0.14× | 0.026 | 0.003 | 0.13× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.036 | 0.004 | 0.12× | 0.049 | 0.004 | 0.09× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.033 | 0.003 | 0.09× | 0.037 | 0.002 | 0.07× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.039 | 0.003 | 0.07× | 0.074 | 0.002 | 0.03× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.377 | 0.098 | 0.26× | 3.317 | 0.950 | 0.29× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.455 | 0.422 | 0.93× | 4.511 | 4.257 | 0.94× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.086 | 0.005 | 0.05× | 0.966 | 0.040 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.436 | 0.015 | 0.03× | 4.705 | 0.123 | 0.03× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.022 | 0.008 | 0.38× | 0.049 | 0.015 | 0.31× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.250 | 0.002 | 0.01× | 0.398 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.044 | 0.020 | 0.46× | 0.080 | 0.033 | 0.40× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.033 | 0.002 | 0.06× | 0.055 | 0.003 | 0.05× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.613 | 0.258 | 0.42× | 6.318 | 2.625 | 0.42× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.029 | 0.016 | 0.55× | 0.065 | 0.047 | 0.72× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.008 | 0.002 | 0.24× | 0.010 | 0.003 | 0.25× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.408 | 0.017 | 0.04× | 4.120 | 0.134 | 0.03× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.009 | 0.003 | 0.34× | 0.010 | 0.004 | 0.38× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.089 | 0.674 | 7.59× | 0.350 | 1.042 | 2.98× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.058 | 0.003 | 0.05× | 0.404 | 0.006 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.092 | 0.249 | 2.72× | 0.787 | 0.288 | 0.37× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.026 | 0.012 | 0.47× | 0.028 | 0.015 | 0.54× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.940 | 6.426 | 6.84× | 9.443 | 63.067 | 6.68× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 1.526 | 7.027 | 4.61× | 15.172 | 67.208 | 4.43× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.358 | 0.423 | 1.18× | 3.823 | 4.610 | 1.21× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.284 | 0.257 | 0.90× | 2.852 | 2.609 | 0.91× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.290 | 0.258 | 0.89× | 2.914 | 2.627 | 0.90× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.024 | – | – | 0.171 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.109 | 0.021 | 0.19× | 0.122 | 0.026 | 0.21× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.005 | 0.002 | 0.43× | 0.006 | 0.003 | 0.45× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.194 | – | – | 1.732 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.479 | 0.113 | 0.23× | 4.847 | 1.085 | 0.22× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.241 | 0.021 | 0.09× | 2.262 | 0.179 | 0.08× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.051 | 0.016 | 0.30× | 0.744 | 0.185 | 0.25× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.049 | 0.234 | 4.76× | 0.546 | 1.194 | 2.19× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.061 | 0.033 | 0.54× | 0.954 | 0.613 | 0.64× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.023 | 0.157 | 6.76× | 0.369 | 0.673 | 1.83× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.037 | 0.175 | 4.72× | 0.613 | 0.921 | 1.50× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.023 | 0.118 | 5.26× | 0.339 | 0.580 | 1.71× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.021 | 0.008 | 0.37× | 0.283 | 0.115 | 0.41× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.120 | 0.255 | 2.12× | 1.362 | 1.603 | 1.18× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.024 | 0.159 | 6.50× | 0.381 | 0.603 | 1.58× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.054 | 0.277 | 5.17× | 0.577 | 1.320 | 2.29× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.031 | 0.197 | 6.38× | 0.569 | 1.131 | 1.99× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.049 | 0.208 | 4.22× | 0.771 | 0.906 | 1.17× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.056 | 0.015 | 0.27× | 1.031 | 0.249 | 0.24× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.027 | 0.006 | 0.21× | 0.402 | 0.038 | 0.09× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.054 | 0.031 | 0.58× | 0.583 | 0.381 | 0.65× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.053 | 0.015 | 0.29× | 0.945 | 0.182 | 0.19× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.050 | 0.022 | 0.45× | 0.972 | 0.407 | 0.42× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.313 | 6.083 | 19.42× | 3.290 | 59.338 | 18.04× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.336 | 6.244 | 18.61× | 3.528 | 58.099 | 16.47× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.323 | 6.120 | 18.93× | 3.327 | 57.537 | 17.29× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.468 | 6.089 | 13.01× | 4.586 | 56.548 | 12.33× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.328 | 6.117 | 18.64× | 5.734 | 59.951 | 10.45× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.323 | 6.189 | 19.17× | 3.395 | 57.101 | 16.82× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.010 | 0.002 | 0.20× | 0.012 | 0.003 | 0.21× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.020 | 0.014 | 0.72× | 0.176 | 0.105 | 0.60× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.070 | 0.437 | 6.23× | 0.075 | 0.438 | 5.82× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.017 | 0.013 | 0.76× | 0.116 | 0.098 | 0.85× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.027 | 0.008 | 0.28× | 0.205 | 0.050 | 0.25× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.008 | 0.002 | 0.25× | 0.010 | 0.003 | 0.24× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.009 | 0.002 | 0.24× | 0.016 | 0.004 | 0.24× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.434 | 0.015 | 0.03× | 4.464 | 0.124 | 0.03× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.041 | 0.004 | 0.10× | 0.052 | 0.005 | 0.09× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.007 | 0.002 | 0.30× | 0.010 | 0.004 | 0.39× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.673 | 0.342 | 0.51× | 6.968 | 3.384 | 0.49× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.067 | 0.044 | 0.65× | 0.106 | 0.071 | 0.67× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.76× | 0.003 | 0.003 | 0.80× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.687 | 0.018 | 0.03× | 8.060 | 0.099 | 0.01× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.004 | 0.002 | 0.52× | 0.006 | 0.004 | 0.64× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.005 | 0.002 | 0.54× | 0.007 | 0.004 | 0.67× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.126 | 0.039 | 0.31× | 1.222 | 0.338 | 0.28× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.046 | 0.007 | 0.15× | 0.423 | 0.044 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.059 | 0.010 | 0.16× | 0.507 | 0.076 | 0.15× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.010 | 0.002 | 0.21× | 0.014 | 0.004 | 0.27× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.011 | 0.004 | 0.32× | 0.013 | 0.005 | 0.41× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.600 | 0.007 | 0.01× | 5.839 | 0.039 | 0.01× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.061 | 0.053 | 0.03× | 1.998 | 0.051 | 0.03× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.579 | 0.334 | 0.58× | 7.314 | 4.111 | 0.56× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.213 | – | – | 2.419 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.139 | 1.799 | 12.97× | 1.499 | 11.924 | 7.96× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.732 | 0.569 | 0.78× | 9.039 | 6.249 | 0.69× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.142 | 1.657 | 11.67× | 1.556 | 11.839 | 7.61× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.121 | 1.730 | 14.24× | 1.416 | 11.914 | 8.42× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.143 | 1.730 | 12.10× | 1.512 | 11.512 | 7.62× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.022 | 0.482 | 22.16× | 0.243 | 3.435 | 14.14× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.014 | 0.079 | 5.49× | 0.084 | 0.055 | 0.65× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.278 | 0.310 | 1.11× | 3.988 | 4.600 | 1.15× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 264.8 | – | – | 2,544.0 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.370 | 0.172 | 0.47× | 0.368 | 0.169 | 0.46× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.565 | 2.255 | 1.44× | 21.340 | 22.424 | 1.05× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.042 | 0.030 | 0.72× | 0.522 | 0.314 | 0.60× | 0.0e+00 |
| `statistics.compare` | timed alone | 81.650 | – | – | 1,209.2 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.197 | 0.174 | 0.88× | 0.188 | 0.175 | 0.93× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 3.823 | – | – | 47.960 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.495 | 0.474 | 0.96× | 1.776 | 1.721 | 0.97× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 1.380 | 1.343 | 0.97× | 14.871 | 14.121 | 0.95× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.059 | 0.030 | 0.51× | 1.114 | 0.347 | 0.31× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 2.878 | 2.811 | 0.98× | 39.311 | 39.129 | 1.00× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 2.029 | 1.970 | 0.97× | 24.943 | 25.065 | 1.00× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.057 | 0.052 | 0.92× | 0.357 | 0.448 | 1.25× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 75.702 | – | – | 863.5 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.421 | 0.274 | 0.65× | 1.124 | 0.634 | 0.56× | 0.0e+00 |
| `statistics.proportion_ci` | statsmodels | 0.084 | 0.085 | 1.02× | 0.081 | 0.087 | 1.08× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.079 | – | – | 20.610 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.464 | 0.442 | 0.95× | 0.467 | 0.443 | 0.95× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.090 | 0.515 | 5.69× | 0.393 | 1.016 | 2.59× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 2.289 | 1.314 | 0.57× | 21.653 | 10.794 | 0.50× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.011 | 0.003 | 0.23× | 0.014 | 0.003 | 0.23× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.019 | 0.004 | 0.20× | 0.019 | 0.006 | 0.33× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 0.915 | 0.739 | 0.81× | 9.846 | 7.589 | 0.77× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.010 | 0.005 | 0.49× | 0.073 | 0.030 | 0.41× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.242 | 0.136 | 0.56× | 2.382 | 1.504 | 0.63× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 4.918 | 4.163 | 0.85× | 51.862 | 51.859 | 1.00× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.151 | 24.290 | 21.11× | 2.166 | 94.686 | 43.71× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 10.492 | 9.575 | 0.91× | 106.2 | 117.8 | 1.11× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 12.882 | 18.792 | 1.46× | 166.5 | 217.8 | 1.31× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.122 | – | – | 1.617 | – | – | – |
| `text.citation_recall` | timed alone | 0.095 | – | – | 1.271 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.118 | 0.003 | 0.03× | 0.936 | 0.010 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.035 | 0.052 | 1.49× | 0.311 | 0.544 | 1.75× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.039 | 0.011 | 0.29× | 0.340 | 0.065 | 0.19× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.474 | 0.152 | 0.32× | 4.893 | 3.109 | 0.64× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.321 | 0.081 | 0.25× | 0.781 | 0.199 | 0.25× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.034 | 0.023 | 0.67× | 0.197 | 0.158 | 0.80× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.239 | – | – | 1.847 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.198 | 0.352 | 1.78× | 1.781 | 3.482 | 1.96× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.282 | 0.086 | 0.30× | 2.768 | 0.631 | 0.23× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.019 | 0.016 | 0.81× | 0.149 | 0.116 | 0.78× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.352 | 0.114 | 0.32× | 3.563 | 1.140 | 0.32× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.182 | 0.080 | 0.44× | 1.708 | 0.962 | 0.56× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.027 | 0.009 | 0.34× | 0.240 | 0.074 | 0.31× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.043 | – | – | 0.373 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 0.976 | 2.305 | 2.36× | 9.355 | 23.004 | 2.46× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.118 | 0.100 | 0.85× | 1.119 | 0.959 | 0.86× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.084 | 0.533 | 6.32× | 0.231 | 0.762 | 3.30× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.044 | 0.010 | 0.23× | 0.421 | 0.151 | 0.36× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.078 | 0.083 | 1.06× | 0.309 | 0.312 | 1.01× | 2.9e-15 |
| `text.mauve` | timed alone | 2.902 | – | – | 560.1 | – | – | – |
| `text.meteor` | NLTK | 3.350 | 4.020 | 1.20× | 33.931 | 39.640 | 1.17× | 0.0e+00 |
| `text.model_score` | timed alone | 0.012 | – | – | 0.100 | – | – | – |
| `text.moverscore` | POT | 13.680 | 1.094 | 0.08× | 191.6 | 12.551 | 0.07× | 1.1e-16 |
| `text.perplexity` | NumPy formula | 0.040 | 0.009 | 0.24× | 0.332 | 0.064 | 0.19× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.050 | 0.003 | 0.05× | 0.429 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.253 | – | – | 2.653 | – | – | – |
| `text.rouge_1` | rouge-score | 1.875 | 2.170 | 1.16× | 20.300 | 23.655 | 1.17× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 1.898 | 2.204 | 1.16× | 20.589 | 23.657 | 1.15× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 2.391 | 2.727 | 1.14× | 26.246 | 30.040 | 1.14× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 3.764 | 4.164 | 1.11× | 41.516 | 45.533 | 1.10× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.013 | 0.004 | 0.30× | 0.034 | 0.008 | 0.23× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 6.548 | 8.981 | 1.37× | 170.3 | 252.4 | 1.48× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.332 | – | – | 0.434 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 4.975 | 5.312 | 1.07× | 53.541 | 58.870 | 1.10× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.890 | 0.045 | 0.05× | 9.052 | 0.539 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 23.411 | 13.528 | 0.58× | 222.0 | 142.6 | 0.64× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.206 | – | – | 2.453 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.229 | – | – | 2.718 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.174 | 0.019 | 0.11× | 2.180 | 0.151 | 0.07× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.449 | – | – | 1.474 | – | – | – |
| `text.win_rate` | NumPy formula | 0.030 | 0.005 | 0.15× | 0.242 | 0.014 | 0.06× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.377 | 0.370 | 0.98× | 3.834 | 3.658 | 0.95× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.107 | 0.086 | 0.80× | 0.687 | 0.685 | 1.00× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.053 | 0.032 | 0.60× | 0.707 | 0.643 | 0.91× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.224 | 1.145 | 5.12× | 1.087 | 2.909 | 2.68× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.057 | 0.033 | 0.57× | 0.696 | 0.631 | 0.91× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.047 | 0.023 | 0.48× | 0.701 | 0.630 | 0.90× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.034 | 0.021 | 0.62× | 0.618 | 0.594 | 0.96× | 0.0e+00 |
