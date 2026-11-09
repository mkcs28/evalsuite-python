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
EvalSuite 0.5.5 benchmarks | Python 3.13.18 | NumPy 2.5.4 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 12/12 | 10.66× | 1.44×–100.84× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 24/24 | 3.84× | 1.04×–18.07× |
| LLM evaluation | 6 | 18 | 18/18 | 18/18 | 2.37× | 1.11×–9.03× |
| LLM systems | 6 | 18 | 18/18 | 18/18 | 1.75× | 1.04×–6.04× |
| Segmentation and object detection | 3 | 9 | 9/9 | 9/9 | 3.13× | 1.22×–15.97× |
| Overall | 27 | 81 | 81/81 | 81/81 | 3.29× | 1.04×–100.84× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.061 | 0.994 | 16.38× | 0.02 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 0.732 | 11.483 | 15.68× | 0.77 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 7.139 | 113.744 | 15.93× | 7.63 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.039 | 0.044 | 1.11× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.147 | 0.301 | 2.04× | 0.29 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 1.360 | 2.783 | 2.05× | 2.53 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.168 | 6.722 | 39.90× | 0.03 | 0.04 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 0.873 | 88.059 | 100.84× | 1.69 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 9.326 | 841.173 | 90.19× | 16.28 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.118 | 1.118 | 9.46× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 10.818 | 24.057 | 2.22× | 8.39 | 7.63 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 171.396 | 293.972 | 1.72× | 83.93 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.564 | 3.073 | 5.44× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 16.191 | 94.164 | 5.82× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 227.674 | 1188.164 | 5.22× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.130 | 0.536 | 4.13× | 0.02 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 0.552 | 1.320 | 2.39× | 0.76 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 4.973 | 5.386 | 1.08× | 7.63 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.252 | 3.975 | 15.75× | 0.02 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 2.643 | 36.457 | 13.79× | 0.76 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 19.484 | 351.991 | 18.07× | 7.63 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.156 | 1.010 | 6.48× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 10.187 | 14.669 | 1.44× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 149.151 | 200.911 | 1.35× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 13.165 | 16.095 | 1.22× | 0.55 | 1.36 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 43.629 | 78.068 | 1.79× | 1.27 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 369.040 | 683.098 | 1.85× | 8.00 | 34.44 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.053 | 0.059 | 1.12× | 0.04 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 3.752 | 3.888 | 1.04× | 3.82 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 78.803 | 84.907 | 1.08× | 38.15 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.071 | 0.468 | 6.62× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 0.726 | 3.060 | 4.21× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 9.373 | 13.468 | 1.44× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.139 | 1.252 | 9.03× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 11.713 | 61.059 | 5.21× | 0.49 | 3.11 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 120.490 | 589.855 | 4.90× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.213 | 3.409 | 15.97× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 2.913 | 22.548 | 7.74× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 30.902 | 224.469 | 7.26× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.237 | 0.463 | 1.96× | 0.27 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 4.903 | 10.650 | 2.17× | 0.29 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 7.054 | 13.314 | 1.89× | 0.28 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.024 | 0.257 | 10.74× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.016 | 0.186 | 11.93× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.015 | 0.170 | 11.15× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.120 | 0.604 | 5.01× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 7.499 | 23.549 | 3.14× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 148.000 | 317.069 | 2.14× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.101 | 0.612 | 6.08× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.600 | 1.432 | 2.39× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 7.050 | 9.713 | 1.38× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.108 | 0.257 | 2.38× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 9.624 | 23.215 | 2.41× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 98.006 | 223.289 | 2.28× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 41.466 | 48.310 | 1.17× | 0.36 | 0.25 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 431.295 | 523.049 | 1.21× | 1.76 | 2.06 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 1.751 | 1.827 | 1.04× | 0.07 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.232 | 1.128 | 4.86× | 0.06 | 0.08 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 11.779 | 22.937 | 1.95× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 160.506 | 272.958 | 1.70× | 54.36 | 76.30 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.077 | 0.463 | 6.04× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.548 | 1.434 | 2.62× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 6.664 | 7.250 | 1.09× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.078 | 0.221 | 2.83× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 7.892 | 22.828 | 2.89× | 0.70 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 105.037 | 225.640 | 2.15× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.031 | 0.034 | 1.10× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 0.807 | 1.612 | 2.00× | 0.76 | 0.77 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 10.752 | 12.188 | 1.13× | 7.63 | 7.63 | 0.0e+00 |
| systems: maintainability index (100 programs) | 100,000 | radon | 8.207 | 9.189 | 1.12× | 0.18 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 90.300 | 99.302 | 1.10× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.332 | 0.352 | 1.06× | 0.04 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 1.195 | 1.483 | 1.24× | 0.11 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 139.586 | 167.781 | 1.20× | 0.26 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 1435.788 | 2056.377 | 1.43× | 2.52 | 211.52 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.108 | 0.356 | 3.31× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 12.057 | 38.551 | 3.20× | 0.12 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 124.738 | 384.377 | 3.08× | 1.15 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.316 | 0.553 | 1.75× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 33.337 | 59.790 | 1.79× | 0.12 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 354.711 | 618.624 | 1.74× | 1.16 | 6.55 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.5.5, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 5.65× | 1.20×–30.57× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 42/42 | 5.93× | 1.08×–36.37× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 46/46 | 4.02× | 1.07×–46.36× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 14/14 | 2.93× | 1.06×–8.04× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 5.93× | 1.22×–22.06× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 172/172 | 4.97× | 1.06×–46.36× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.006 | 0.002 | 0.25× | 0.007 | 0.002 | 0.27× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.009 | 0.003 | 0.35× | 0.008 | 0.004 | 0.42× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 0.860 | 2.563 | 2.98× | 10.324 | 31.455 | 3.05× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.234 | – | – | 1.753 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.093 | 0.010 | 0.10× | 0.907 | 0.065 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.015 | 0.002 | 0.11× | 0.024 | 0.002 | 0.08× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.84× | 0.003 | 0.003 | 0.83× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.131 | – | – | 1.704 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.016 | 0.003 | 0.17× | 0.023 | 0.003 | 0.12× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.723 | 0.454 | 0.63× | 7.386 | 5.633 | 0.76× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.466 | 3.701 | 7.93× | 5.160 | 33.948 | 6.58× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 1.184 | 6.271 | 5.30× | 10.496 | 50.186 | 4.78× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.322 | 0.473 | 1.47× | 2.836 | 4.585 | 1.62× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.052 | 0.601 | 11.56× | 0.298 | 3.258 | 10.92× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.867 | 1.935 | 2.23× | 10.314 | 16.510 | 1.60× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.352 | 1.805 | 5.13× | 3.253 | 11.882 | 3.65× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.057 | 0.869 | 15.38× | 0.404 | 5.378 | 13.30× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.066 | 1.433 | 21.59× | 0.340 | 9.323 | 27.42× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.303 | 0.476 | 1.57× | 2.638 | 4.686 | 1.78× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.369 | 2.363 | 6.41× | 3.221 | 13.806 | 4.29× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.072 | 1.930 | 26.79× | 0.363 | 10.538 | 29.06× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.579 | 3.296 | 5.69× | 6.393 | 28.098 | 4.40× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.070 | 1.809 | 25.97× | 0.442 | 11.198 | 25.34× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.483 | 3.261 | 6.76× | 5.559 | 22.992 | 4.14× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.841 | 3.179 | 3.78× | 3.238 | 22.615 | 6.98× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.067 | 1.898 | 28.29× | 0.360 | 10.690 | 29.67× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.367 | 2.350 | 6.41× | 3.366 | 23.270 | 6.91× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.372 | 2.376 | 6.39× | 4.827 | 14.873 | 3.08× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 9.894 | 21.736 | 2.20× | 130.9 | 202.4 | 1.55× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.067 | 2.056 | 30.57× | 0.414 | 11.314 | 27.34× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.638 | 2.547 | 3.99× | 7.209 | 20.840 | 2.89× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.062 | 0.035 | 0.57× | 0.419 | 0.321 | 0.77× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.063 | 1.905 | 30.16× | 0.414 | 13.047 | 31.48× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.060 | 1.998 | 33.12× | 0.411 | 12.989 | 31.62× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.139 | 0.017 | 0.12× | 1.465 | 0.145 | 0.10× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.069 | 2.099 | 30.43× | 0.437 | 11.159 | 25.56× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.065 | 2.374 | 36.37× | 0.422 | 11.202 | 26.53× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.075 | 0.043 | 0.57× | 0.402 | 0.315 | 0.78× | 0.0e+00 |
| `code.code_complexity` | radon | 0.762 | 0.810 | 1.06× | 6.845 | 7.681 | 1.12× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 4.942 | 6.004 | 1.21× | 42.036 | 53.504 | 1.27× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.018 | 0.001 | 0.07× | 0.016 | 0.001 | 0.08× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.027 | 0.019 | 0.69× | 0.129 | 0.106 | 0.82× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.004 | 0.003 | 0.82× | 0.003 | 0.003 | 0.79× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.106 | 0.019 | 0.18× | 0.636 | 0.106 | 0.17× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.036 | 0.003 | 0.09× | 0.030 | 0.003 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.120 | 0.012 | 0.10× | 0.683 | 0.045 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.016 | 0.001 | 0.09× | 0.013 | 0.001 | 0.10× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 2.530 | 1.041 | 0.41× | 17.563 | 11.065 | 0.63× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.347 | 0.262 | 0.76× | 3.372 | 2.678 | 0.79× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 1.708 | 13.987 | 8.19× | 15.669 | 68.450 | 4.37× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.073 | 0.034 | 0.47× | 9.618 | 4.073 | 0.42× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 3.212 | 13.640 | 4.25× | 17.764 | 74.921 | 4.22× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.012 | 0.002 | 0.18× | 0.083 | 0.003 | 0.03× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.35× | 0.009 | 0.004 | 0.41× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.27× | 0.014 | 0.004 | 0.32× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.016 | 0.023 | 1.41× | 0.028 | 0.029 | 1.01× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.004 | 0.002 | 0.49× | 0.008 | 0.002 | 0.29× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.068 | 0.001 | 0.02× | 0.079 | 0.001 | 0.02× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.016 | 0.003 | 0.17× | 0.022 | 0.003 | 0.14× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.034 | 0.003 | 0.09× | 0.044 | 0.005 | 0.10× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.020 | 0.002 | 0.10× | 0.032 | 0.003 | 0.08× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.026 | 0.002 | 0.08× | 0.048 | 0.003 | 0.06× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.325 | 0.095 | 0.29× | 3.309 | 0.965 | 0.29× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.408 | 0.382 | 0.94× | 4.638 | 4.337 | 0.94× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.091 | 0.004 | 0.05× | 1.922 | 0.066 | 0.03× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.444 | 0.016 | 0.03× | 6.960 | 0.187 | 0.03× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.018 | 0.008 | 0.45× | 0.068 | 0.022 | 0.33× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.240 | 0.002 | 0.01× | 0.627 | 0.004 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.039 | 0.020 | 0.52× | 0.123 | 0.049 | 0.40× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.030 | 0.002 | 0.07× | 0.076 | 0.004 | 0.05× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.408 | 0.258 | 0.63× | 4.155 | 2.651 | 0.64× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.030 | 0.016 | 0.55× | 0.065 | 0.045 | 0.70× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.008 | 0.002 | 0.26× | 0.010 | 0.004 | 0.39× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.393 | 0.017 | 0.04× | 4.040 | 0.139 | 0.03× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.007 | 0.003 | 0.43× | 0.008 | 0.004 | 0.49× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.087 | 0.701 | 8.04× | 0.347 | 1.018 | 2.93× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.056 | 0.003 | 0.05× | 0.398 | 0.006 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.042 | 0.251 | 5.93× | 0.096 | 0.286 | 2.97× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.026 | 0.012 | 0.47× | 0.029 | 0.015 | 0.51× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 0.885 | 6.091 | 6.88× | 9.136 | 60.445 | 6.62× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 1.437 | 6.753 | 4.70× | 15.228 | 67.110 | 4.41× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.366 | 0.416 | 1.14× | 3.898 | 4.612 | 1.18× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.283 | 0.270 | 0.96× | 2.888 | 2.613 | 0.90× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.288 | 0.264 | 0.92× | 2.978 | 2.674 | 0.90× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.023 | – | – | 0.173 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.105 | 0.021 | 0.20× | 0.118 | 0.024 | 0.20× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.005 | 0.002 | 0.43× | 0.006 | 0.003 | 0.46× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.188 | – | – | 1.786 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.481 | 0.108 | 0.22× | 5.025 | 1.086 | 0.22× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.246 | 0.020 | 0.08× | 2.262 | 0.184 | 0.08× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.047 | 0.016 | 0.35× | 0.591 | 0.173 | 0.29× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.049 | 0.227 | 4.64× | 0.407 | 0.937 | 2.30× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.061 | 0.035 | 0.57× | 0.734 | 0.550 | 0.75× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.023 | 0.168 | 7.44× | 0.271 | 0.514 | 1.90× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.039 | 0.186 | 4.76× | 0.468 | 0.764 | 1.63× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.022 | 0.118 | 5.45× | 0.251 | 0.441 | 1.76× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.021 | 0.008 | 0.36× | 0.207 | 0.085 | 0.41× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.122 | 0.260 | 2.12× | 1.187 | 1.420 | 1.20× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.031 | 0.165 | 5.40× | 0.305 | 0.474 | 1.55× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.053 | 0.297 | 5.61× | 0.515 | 1.115 | 2.16× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.032 | 0.198 | 6.11× | 0.418 | 0.825 | 1.97× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.045 | 0.207 | 4.63× | 0.546 | 0.665 | 1.22× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.057 | 0.016 | 0.28× | 0.765 | 0.197 | 0.26× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.030 | 0.006 | 0.19× | 0.305 | 0.038 | 0.12× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.052 | 0.034 | 0.66× | 0.526 | 0.329 | 0.63× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.054 | 0.016 | 0.29× | 0.693 | 0.151 | 0.22× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.047 | 0.023 | 0.49× | 0.750 | 0.323 | 0.43× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.320 | 6.004 | 18.75× | 3.203 | 55.755 | 17.41× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.349 | 5.888 | 16.88× | 3.611 | 55.211 | 15.29× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.316 | 6.065 | 19.19× | 3.161 | 53.534 | 16.94× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.461 | 6.156 | 13.35× | 4.602 | 53.590 | 11.64× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.453 | 5.832 | 12.87× | 3.401 | 54.984 | 16.17× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.323 | 6.562 | 20.29× | 3.228 | 56.152 | 17.40× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.010 | 0.002 | 0.21× | 0.012 | 0.003 | 0.21× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.019 | 0.014 | 0.73× | 0.161 | 0.106 | 0.66× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.074 | 0.439 | 5.97× | 0.074 | 0.434 | 5.84× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.016 | 0.013 | 0.80× | 0.111 | 0.094 | 0.85× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.028 | 0.008 | 0.27× | 0.200 | 0.050 | 0.25× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.008 | 0.002 | 0.25× | 0.010 | 0.003 | 0.24× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.010 | 0.002 | 0.23× | 0.016 | 0.004 | 0.26× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.448 | 0.016 | 0.04× | 4.320 | 0.117 | 0.03× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.044 | 0.004 | 0.09× | 0.058 | 0.005 | 0.08× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.008 | 0.002 | 0.29× | 0.010 | 0.004 | 0.39× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.703 | 0.339 | 0.48× | 6.724 | 3.406 | 0.51× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.063 | 0.045 | 0.71× | 0.099 | 0.067 | 0.68× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.74× | 0.003 | 0.003 | 0.79× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.683 | 0.017 | 0.02× | 10.402 | 0.114 | 0.01× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.004 | 0.002 | 0.54× | 0.008 | 0.004 | 0.53× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.004 | 0.003 | 0.58× | 0.007 | 0.005 | 0.70× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.124 | 0.039 | 0.32× | 1.809 | 0.344 | 0.19× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.053 | 0.007 | 0.13× | 0.428 | 0.044 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.060 | 0.010 | 0.17× | 0.517 | 0.074 | 0.14× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.010 | 0.002 | 0.22× | 0.014 | 0.004 | 0.28× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.010 | 0.004 | 0.36× | 0.015 | 0.007 | 0.49× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.230 | 0.006 | 0.02× | 2.459 | 0.039 | 0.02× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.037 | 0.055 | 0.03× | 2.089 | 0.050 | 0.02× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.251 | 0.318 | 1.27× | 3.440 | 4.200 | 1.22× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.217 | – | – | 2.370 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.137 | 1.666 | 12.17× | 1.425 | 11.225 | 7.88× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.257 | 0.503 | 1.96× | 3.541 | 6.237 | 1.76× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.143 | 1.596 | 11.14× | 1.455 | 11.079 | 7.62× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.123 | 1.643 | 13.35× | 1.412 | 11.222 | 7.95× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.144 | 1.625 | 11.26× | 1.433 | 11.095 | 7.74× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.023 | 0.499 | 22.06× | 0.230 | 3.343 | 14.55× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.010 | 0.059 | 5.93× | 0.037 | 0.062 | 1.66× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.278 | 0.302 | 1.08× | 3.419 | 3.927 | 1.15× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 149.9 | – | – | 1,502.5 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.024 | 0.182 | 7.50× | 0.024 | 0.185 | 7.84× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.578 | 2.004 | 1.27× | 21.883 | 23.744 | 1.09× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.041 | 0.030 | 0.73× | 0.506 | 0.272 | 0.54× | 0.0e+00 |
| `statistics.compare` | timed alone | 42.407 | – | – | 575.1 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.016 | 0.183 | 11.46× | 0.016 | 0.193 | 12.40× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 4.009 | – | – | 46.601 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.095 | 0.474 | 5.00× | 0.238 | 1.753 | 7.36× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.519 | 1.460 | 2.81× | 7.712 | 13.482 | 1.75× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.058 | 0.030 | 0.52× | 0.812 | 0.272 | 0.34× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.880 | 3.079 | 3.50× | 16.670 | 38.007 | 2.28× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.543 | 2.164 | 3.99× | 7.851 | 23.986 | 3.06× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.014 | 0.059 | 4.33× | 0.198 | 0.536 | 2.70× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 44.397 | – | – | 560.6 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.078 | 0.286 | 3.69× | 0.399 | 0.904 | 2.27× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.014 | 0.087 | 6.15× | 0.008 | 0.081 | 9.87× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.077 | – | – | 21.683 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.044 | 0.502 | 11.39× | 0.041 | 0.464 | 11.41× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.091 | 0.505 | 5.58× | 0.534 | 1.271 | 2.38× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.352 | 1.349 | 3.83× | 3.733 | 11.172 | 2.99× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.011 | 0.003 | 0.25× | 0.013 | 0.003 | 0.23× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.017 | 0.005 | 0.32× | 0.018 | 0.006 | 0.34× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 0.972 | 0.780 | 0.80× | 9.722 | 7.631 | 0.78× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.014 | 0.007 | 0.47× | 0.073 | 0.026 | 0.36× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.247 | 0.138 | 0.56× | 2.425 | 1.546 | 0.64× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 2.749 | 4.193 | 1.53× | 29.632 | 54.962 | 1.85× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.164 | 25.278 | 21.72× | 2.094 | 97.073 | 46.36× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 9.324 | 9.954 | 1.07× | 98.954 | 112.9 | 1.14× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 14.323 | 20.742 | 1.45× | 155.3 | 224.9 | 1.45× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.127 | – | – | 1.455 | – | – | – |
| `text.citation_recall` | timed alone | 0.095 | – | – | 1.137 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.119 | 0.003 | 0.03× | 0.946 | 0.011 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.035 | 0.051 | 1.48× | 0.304 | 0.530 | 1.74× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.039 | 0.011 | 0.28× | 0.331 | 0.065 | 0.20× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.436 | 0.159 | 0.37× | 4.599 | 2.659 | 0.58× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.325 | 0.082 | 0.25× | 0.805 | 0.201 | 0.25× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.033 | 0.023 | 0.70× | 0.186 | 0.153 | 0.82× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.242 | – | – | 1.943 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.200 | 0.345 | 1.72× | 1.759 | 3.543 | 2.01× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.036 | 0.084 | 2.32× | 0.085 | 0.620 | 7.32× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.020 | 0.016 | 0.82× | 0.150 | 0.116 | 0.77× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.353 | 0.115 | 0.33× | 3.605 | 1.139 | 0.32× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.184 | 0.081 | 0.44× | 1.713 | 0.857 | 0.50× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.026 | 0.009 | 0.35× | 0.243 | 0.078 | 0.32× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.044 | – | – | 0.359 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 1.007 | 2.520 | 2.50× | 9.370 | 22.907 | 2.44× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.121 | 0.101 | 0.83× | 1.178 | 0.969 | 0.82× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.089 | 0.567 | 6.39× | 0.222 | 0.812 | 3.66× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.047 | 0.010 | 0.22× | 0.397 | 0.153 | 0.39× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.069 | 0.081 | 1.18× | 0.158 | 0.291 | 1.84× | 2.9e-15 |
| `text.mauve` | timed alone | 3.334 | – | – | 448.1 | – | – | – |
| `text.meteor` | NLTK | 1.199 | 4.440 | 3.70× | 12.012 | 42.359 | 3.53× | 0.0e+00 |
| `text.model_score` | timed alone | 0.012 | – | – | 0.067 | – | – | – |
| `text.moverscore` | POT | 0.793 | 1.119 | 1.41× | 8.287 | 11.612 | 1.40× | 0.0e+00 |
| `text.perplexity` | NumPy formula | 0.040 | 0.010 | 0.26× | 0.329 | 0.065 | 0.20× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.051 | 0.003 | 0.05× | 0.432 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.260 | – | – | 2.614 | – | – | – |
| `text.rouge_1` | rouge-score | 0.621 | 2.571 | 4.14× | 7.782 | 22.602 | 2.90× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.803 | 2.501 | 3.12× | 10.402 | 22.287 | 2.14× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 1.333 | 3.019 | 2.27× | 13.862 | 28.340 | 2.04× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 2.836 | 4.234 | 1.49× | 28.062 | 42.502 | 1.51× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.013 | 0.004 | 0.30× | 0.032 | 0.010 | 0.30× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 5.219 | 9.130 | 1.75× | 135.9 | 241.2 | 1.78× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.351 | – | – | 0.417 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 3.179 | 5.696 | 1.79× | 30.310 | 57.558 | 1.90× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.912 | 0.045 | 0.05× | 8.679 | 0.491 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 9.376 | 13.458 | 1.44× | 93.804 | 141.3 | 1.51× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.209 | – | – | 2.718 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.229 | – | – | 3.020 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.175 | 0.019 | 0.11× | 2.261 | 0.145 | 0.06× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.459 | – | – | 1.361 | – | – | – |
| `text.win_rate` | NumPy formula | 0.030 | 0.005 | 0.15× | 0.250 | 0.012 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.390 | 0.372 | 0.95× | 3.779 | 3.645 | 0.96× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.102 | 0.086 | 0.85× | 0.674 | 0.661 | 0.98× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.051 | 0.032 | 0.62× | 0.670 | 0.630 | 0.94× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.220 | 1.112 | 5.06× | 1.104 | 3.031 | 2.75× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.053 | 0.032 | 0.60× | 0.706 | 0.634 | 0.90× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.047 | 0.023 | 0.49× | 0.672 | 0.604 | 0.90× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.031 | 0.021 | 0.67× | 0.612 | 0.584 | 0.95× | 0.0e+00 |
