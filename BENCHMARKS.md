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
EvalSuite 0.7.0 benchmarks | Python 3.13.18 | NumPy 2.5.4 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 12/12 | 11.94× | 1.59×–87.58× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 23/24 | 4.08× | 0.81×–23.71× |
| LLM evaluation | 6 | 18 | 18/18 | 18/18 | 4.06× | 1.11×–12.30× |
| LLM systems | 6 | 18 | 18/18 | 17/18 | 1.86× | 0.97×–9.95× |
| Segmentation and object detection | 3 | 9 | 9/9 | 9/9 | 6.42× | 1.62×–20.37× |
| Overall | 27 | 81 | 81/81 | 79/81 | 4.22× | 0.81×–87.58× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.063 | 0.975 | 15.56× | 0.02 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 0.703 | 11.536 | 16.41× | 0.77 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 7.074 | 109.545 | 15.49× | 7.63 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.039 | 0.043 | 1.11× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.315 | 0.535 | 1.70× | 0.29 | 0.69 | 3.8e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 1.258 | 2.709 | 2.15× | 2.53 | 6.32 | 1.7e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.164 | 6.841 | 41.80× | 0.03 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 1.288 | 100.080 | 77.71× | 1.69 | 3.06 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 9.830 | 860.871 | 87.58× | 16.28 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.066 | 1.080 | 16.31× | 0.02 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 4.282 | 22.979 | 5.37× | 1.53 | 7.64 | 1.1e-16 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 84.765 | 267.700 | 3.16× | 15.26 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.407 | 2.468 | 6.07× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 13.126 | 81.395 | 6.20× | 8.46 | 58.00 | 1.1e-16 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 179.614 | 946.708 | 5.27× | 83.99 | 579.85 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.100 | 0.401 | 4.03× | 0.02 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 0.409 | 0.763 | 1.87× | 0.76 | 0.29 | 2.5e-14 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 4.390 | 4.683 | 1.07× | 7.63 | 1.91 | 1.4e-14 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.181 | 3.326 | 18.35× | 0.02 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 1.452 | 34.435 | 23.71× | 0.76 | 2.24 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 17.746 | 328.099 | 18.49× | 7.63 | 22.32 | 1.8e-15 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.165 | 0.684 | 4.14× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 9.535 | 19.121 | 2.01× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 133.051 | 168.411 | 1.27× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 1.187 | 17.452 | 14.70× | 0.95 | 1.36 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 3.984 | 75.339 | 18.91× | 1.13 | 4.35 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 30.252 | 616.269 | 20.37× | 2.83 | 34.54 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.058 | 0.070 | 1.21× | 0.04 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 4.320 | 5.944 | 1.38× | 3.82 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 86.361 | 69.898 | 0.81× | 38.15 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.069 | 0.470 | 6.79× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 0.544 | 1.360 | 2.50× | 3.05 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 6.231 | 9.877 | 1.59× | 30.52 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.139 | 1.374 | 9.89× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 12.212 | 65.724 | 5.38× | 0.49 | 3.11 | 3.5e-18 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 118.287 | 577.602 | 4.88× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.233 | 2.589 | 11.09× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 3.057 | 23.816 | 7.79× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 29.347 | 215.237 | 7.33× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.166 | 0.280 | 1.69× | 0.27 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 3.413 | 6.415 | 1.88× | 0.29 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 8.748 | 14.133 | 1.62× | 0.28 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.016 | 0.235 | 15.03× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.016 | 0.188 | 11.96× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.015 | 0.181 | 11.95× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.081 | 0.477 | 5.91× | 0.14 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 6.274 | 25.913 | 4.13× | 13.93 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 115.133 | 305.019 | 2.65× | 139.24 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.069 | 0.468 | 6.74× | 0.02 | 0.02 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 0.553 | 1.030 | 1.86× | 1.53 | 1.53 | 1.4e-42 |
| statistics: Welch t-test | 1,000,000 | SciPy | 4.931 | 7.231 | 1.47× | 15.26 | 15.26 | 9.4e-243 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.111 | 0.245 | 2.22× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 9.657 | 23.328 | 2.42× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 100.587 | 224.261 | 2.23× | 0.50 | 0.16 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (100 programs) | 100,000 | codebleu | 32.284 | 48.859 | 1.51× | 0.37 | 0.25 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (1000 programs) | 1,000,000 | codebleu | 313.887 | 511.292 | 1.63× | 1.98 | 2.06 | 0.0e+00 |
| systems: CodeBLEU n-gram terms (4 programs) | 1,000 | codebleu | 1.271 | 1.806 | 1.42× | 0.07 | 0.07 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000 | scikit-learn | 0.226 | 1.261 | 5.59× | 0.06 | 0.08 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 100,000 | scikit-learn | 12.359 | 24.402 | 1.97× | 5.44 | 7.64 | 0.0e+00 |
| systems: confidence AUROC (n answers) | 1,000,000 | scikit-learn | 181.240 | 279.253 | 1.54× | 54.36 | 76.30 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000 | SciPy | 0.079 | 0.788 | 9.95× | 0.01 | 0.02 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 100,000 | SciPy | 0.532 | 1.410 | 2.65× | 0.76 | 1.53 | 0.0e+00 |
| systems: distribution-shift drop, Welch CI (n scores) | 1,000,000 | SciPy | 8.616 | 9.814 | 1.14× | 7.63 | 15.26 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10 tasks) | 1,000 | jsonschema | 0.076 | 0.277 | 3.66× | 0.00 | 0.00 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (1000 tasks) | 100,000 | jsonschema | 11.064 | 21.992 | 1.99× | 0.70 | 0.05 | 0.0e+00 |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | 1,000,000 | jsonschema | 109.168 | 219.277 | 2.01× | 7.10 | 0.33 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000 | NumPy | 0.035 | 0.038 | 1.09× | 0.01 | 0.01 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 100,000 | NumPy | 0.881 | 1.637 | 1.86× | 0.76 | 0.77 | 0.0e+00 |
| systems: latency p50 / p95 / p99 (n requests) | 1,000,000 | NumPy | 12.868 | 12.543 | 0.97× | 7.63 | 7.63 | 0.0e+00 |
| systems: maintainability index (100 programs) | 100,000 | radon | 8.595 | 9.454 | 1.10× | 0.18 | 0.05 | 0.0e+00 |
| systems: maintainability index (1000 programs) | 1,000,000 | radon | 88.593 | 98.708 | 1.11× | 0.65 | 0.07 | 0.0e+00 |
| systems: maintainability index (4 programs) | 1,000 | radon | 0.326 | 0.351 | 1.08× | 0.04 | 0.04 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 0.224 | 1.355 | 6.06× | 0.01 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 16.933 | 170.383 | 10.06× | 1.13 | 21.57 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 165.031 | 2030.259 | 12.30× | 11.44 | 211.52 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.066 | 0.346 | 5.28× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 5.710 | 36.788 | 6.44× | 0.58 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 63.777 | 363.837 | 5.70× | 5.82 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 0.118 | 0.561 | 4.75× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 22.803 | 59.068 | 2.59× | 0.63 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 152.834 | 613.635 | 4.02× | 6.95 | 6.55 | 0.0e+00 |

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.7.0, Python 3.13.18, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 6.13× | 1.17×–30.41× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 42/42 | 5.84× | 1.10×–35.31× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 46/46 | 8.13× | 1.20×–50.20× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 14/14 | 3.15× | 1.08×–8.99× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 18/18 | 8.90× | 1.20×–43.87× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 172/172 | 6.44× | 1.08×–50.20× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.007 | 0.002 | 0.24× | 0.007 | 0.002 | 0.29× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.009 | 0.003 | 0.35× | 0.009 | 0.004 | 0.38× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 0.855 | 2.632 | 3.08× | 11.037 | 31.658 | 2.87× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.178 | – | – | 1.744 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.092 | 0.010 | 0.10× | 0.936 | 0.064 | 0.07× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.018 | 0.001 | 0.08× | 0.025 | 0.002 | 0.07× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.003 | 0.002 | 0.72× | 0.003 | 0.003 | 0.78× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.128 | – | – | 1.760 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.014 | 0.002 | 0.14× | 0.023 | 0.003 | 0.12× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 0.601 | 0.457 | 0.76× | 7.317 | 5.604 | 0.77× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.446 | 3.144 | 7.05× | 6.390 | 35.282 | 5.52× | 3.3e-16 |
| `calibration.calibration_slope` | statsmodels | 0.818 | 5.610 | 6.86× | 10.265 | 58.785 | 5.73× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.322 | 0.463 | 1.44× | 3.735 | 6.018 | 1.61× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.050 | 0.531 | 10.62× | 0.443 | 4.827 | 10.90× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 0.360 | 1.872 | 5.19× | 6.576 | 21.685 | 3.30× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.344 | 1.686 | 4.90× | 4.037 | 12.964 | 3.21× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.054 | 0.830 | 15.38× | 0.547 | 5.956 | 10.89× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.064 | 1.379 | 21.59× | 0.430 | 9.894 | 23.00× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.292 | 0.483 | 1.65× | 2.865 | 5.020 | 1.75× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.356 | 2.246 | 6.31× | 3.383 | 14.979 | 4.43× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.074 | 1.824 | 24.75× | 0.441 | 11.249 | 25.49× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.467 | 2.974 | 6.36× | 7.885 | 29.970 | 3.80× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.067 | 1.715 | 25.62× | 0.533 | 11.644 | 21.84× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.408 | 2.579 | 6.32× | 7.344 | 32.055 | 4.36× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.345 | 3.060 | 8.87× | 4.023 | 25.310 | 6.29× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.065 | 1.856 | 28.38× | 0.570 | 11.906 | 20.89× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.356 | 2.197 | 6.17× | 3.692 | 15.728 | 4.26× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.348 | 2.212 | 6.36× | 4.845 | 14.936 | 3.08× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 2.474 | 20.573 | 8.32× | 49.158 | 220.5 | 4.49× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.066 | 1.894 | 28.69× | 0.358 | 10.899 | 30.41× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.651 | 2.370 | 3.64× | 6.622 | 20.694 | 3.12× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.058 | 0.035 | 0.60× | 0.423 | 0.311 | 0.74× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.076 | 1.848 | 24.19× | 0.413 | 12.948 | 31.32× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.060 | 1.787 | 29.66× | 0.402 | 14.179 | 35.31× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.141 | 0.017 | 0.12× | 1.627 | 0.166 | 0.10× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.066 | 1.881 | 28.47× | 0.495 | 11.992 | 24.20× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.065 | 1.847 | 28.24× | 0.519 | 11.702 | 22.55× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.059 | 0.045 | 0.76× | 0.465 | 0.358 | 0.77× | 0.0e+00 |
| `code.code_complexity` | radon | 0.745 | 0.818 | 1.10× | 6.979 | 7.544 | 1.08× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 3.730 | 6.791 | 1.82× | 33.894 | 53.628 | 1.58× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.012 | 0.001 | 0.07× | 0.016 | 0.001 | 0.08× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.018 | 0.014 | 0.77× | 0.128 | 0.100 | 0.78× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.003 | 0.002 | 0.72× | 0.003 | 0.003 | 0.78× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.070 | 0.014 | 0.20× | 0.630 | 0.102 | 0.16× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.029 | 0.002 | 0.07× | 0.033 | 0.003 | 0.09× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.078 | 0.007 | 0.09× | 0.637 | 0.047 | 0.07× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.011 | 0.001 | 0.08× | 0.013 | 0.001 | 0.10× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 1.809 | 1.056 | 0.58× | 17.645 | 10.686 | 0.61× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.340 | 0.267 | 0.78× | 6.085 | 2.715 | 0.45× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 0.342 | 15.010 | 43.87× | 3.459 | 70.740 | 20.45× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.042 | 0.022 | 0.53× | 8.800 | 5.369 | 0.61× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 0.350 | 14.115 | 40.31× | 2.570 | 77.972 | 30.34× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.012 | 0.002 | 0.18× | 0.053 | 0.003 | 0.05× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.007 | 0.002 | 0.34× | 0.008 | 0.004 | 0.42× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.011 | 0.003 | 0.27× | 0.015 | 0.004 | 0.28× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.017 | 0.024 | 1.38× | 0.025 | 0.030 | 1.17× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.004 | 0.002 | 0.44× | 0.007 | 0.002 | 0.28× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.064 | 0.001 | 0.02× | 0.087 | 0.001 | 0.01× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.016 | 0.003 | 0.15× | 0.022 | 0.003 | 0.14× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.030 | 0.003 | 0.10× | 0.044 | 0.006 | 0.13× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.022 | 0.002 | 0.09× | 0.041 | 0.004 | 0.10× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.027 | 0.002 | 0.07× | 0.071 | 0.003 | 0.04× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.048 | 0.098 | 2.02× | 1.027 | 0.962 | 0.94× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.460 | 0.429 | 0.93× | 4.471 | 4.384 | 0.98× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.089 | 0.004 | 0.05× | 0.962 | 0.042 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.132 | 0.016 | 0.12× | 1.426 | 0.114 | 0.08× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.017 | 0.009 | 0.50× | 0.043 | 0.015 | 0.36× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.242 | 0.003 | 0.01× | 0.389 | 0.003 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.040 | 0.020 | 0.51× | 0.071 | 0.034 | 0.48× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.031 | 0.002 | 0.07× | 0.047 | 0.003 | 0.06× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 0.409 | 0.264 | 0.65× | 4.222 | 2.659 | 0.63× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.029 | 0.016 | 0.55× | 0.065 | 0.050 | 0.76× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.008 | 0.002 | 0.27× | 0.013 | 0.004 | 0.30× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.109 | 0.018 | 0.16× | 1.258 | 0.140 | 0.11× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.006 | 0.003 | 0.51× | 0.008 | 0.004 | 0.47× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.091 | 0.813 | 8.99× | 0.352 | 0.997 | 2.83× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.083 | 0.004 | 0.05× | 0.401 | 0.006 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.045 | 0.387 | 8.56× | 0.104 | 0.291 | 2.81× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.045 | 0.019 | 0.42× | 0.036 | 0.015 | 0.41× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 1.167 | 7.571 | 6.49× | 9.519 | 63.981 | 6.72× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 1.459 | 6.828 | 4.68× | 15.695 | 69.081 | 4.40× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.377 | 0.429 | 1.14× | 3.900 | 4.618 | 1.18× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.283 | 0.259 | 0.91× | 3.010 | 2.690 | 0.89× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.291 | 0.263 | 0.91× | 3.038 | 2.776 | 0.91× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.021 | – | – | 0.191 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.103 | 0.024 | 0.23× | 0.140 | 0.029 | 0.21× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.005 | 0.002 | 0.42× | 0.006 | 0.003 | 0.40× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.197 | – | – | 2.545 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.372 | 0.111 | 0.30× | 4.182 | 1.066 | 0.25× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.243 | 0.020 | 0.08× | 2.348 | 0.179 | 0.08× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.046 | 0.018 | 0.39× | 0.677 | 0.181 | 0.27× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.046 | 0.234 | 5.09× | 0.548 | 1.222 | 2.23× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.055 | 0.032 | 0.59× | 0.977 | 0.618 | 0.63× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.022 | 0.156 | 6.94× | 0.399 | 0.755 | 1.89× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.035 | 0.189 | 5.33× | 0.669 | 0.972 | 1.45× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.018 | 0.126 | 6.99× | 0.354 | 0.566 | 1.60× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.019 | 0.006 | 0.31× | 0.306 | 0.115 | 0.37× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.123 | 0.265 | 2.14× | 1.342 | 1.563 | 1.17× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.026 | 0.257 | 9.87× | 0.382 | 0.710 | 1.86× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.055 | 0.290 | 5.26× | 0.598 | 1.345 | 2.25× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.028 | 0.260 | 9.22× | 0.599 | 1.130 | 1.89× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.043 | 0.238 | 5.57× | 0.756 | 0.956 | 1.26× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.060 | 0.015 | 0.25× | 1.014 | 0.248 | 0.24× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.029 | 0.006 | 0.20× | 0.441 | 0.041 | 0.09× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.054 | 0.034 | 0.62× | 0.596 | 0.382 | 0.64× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.054 | 0.015 | 0.28× | 0.935 | 0.178 | 0.19× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.052 | 0.020 | 0.38× | 0.922 | 0.400 | 0.43× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.319 | 6.380 | 19.98× | 3.356 | 59.521 | 17.73× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.333 | 6.576 | 19.74× | 3.559 | 71.796 | 20.17× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.318 | 5.994 | 18.86× | 3.313 | 62.618 | 18.90× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.459 | 6.045 | 13.18× | 4.692 | 57.170 | 12.18× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.312 | 5.890 | 18.86× | 3.248 | 57.361 | 17.66× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.315 | 6.077 | 19.30× | 3.404 | 56.443 | 16.58× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.010 | 0.002 | 0.22× | 0.012 | 0.002 | 0.21× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.019 | 0.014 | 0.72× | 0.166 | 0.100 | 0.60× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.071 | 0.436 | 6.12× | 0.076 | 0.459 | 6.08× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.017 | 0.013 | 0.78× | 0.114 | 0.094 | 0.82× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.028 | 0.008 | 0.28× | 0.210 | 0.051 | 0.24× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.008 | 0.002 | 0.25× | 0.010 | 0.003 | 0.26× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.009 | 0.002 | 0.22× | 0.015 | 0.004 | 0.26× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.128 | 0.015 | 0.12× | 1.410 | 0.120 | 0.09× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.044 | 0.004 | 0.09× | 0.051 | 0.005 | 0.09× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.007 | 0.002 | 0.29× | 0.010 | 0.004 | 0.38× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 0.157 | 0.328 | 2.09× | 1.772 | 3.378 | 1.91× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.066 | 0.042 | 0.64× | 0.176 | 0.112 | 0.63× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.003 | 0.002 | 0.77× | 0.004 | 0.004 | 0.89× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 0.233 | 0.018 | 0.08× | 2.844 | 0.110 | 0.04× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.005 | 0.003 | 0.73× | 0.006 | 0.004 | 0.65× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.004 | 0.002 | 0.55× | 0.007 | 0.004 | 0.68× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.134 | 0.040 | 0.30× | 1.241 | 0.350 | 0.28× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.050 | 0.007 | 0.14× | 0.436 | 0.044 | 0.10× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.059 | 0.010 | 0.17× | 0.506 | 0.080 | 0.16× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.010 | 0.002 | 0.22× | 0.014 | 0.004 | 0.27× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.010 | 0.003 | 0.34× | 0.014 | 0.005 | 0.37× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 0.042 | 0.006 | 0.13× | 0.467 | 0.038 | 0.08× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 2.129 | 0.056 | 0.03× | 2.118 | 0.051 | 0.02× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.260 | 0.331 | 1.27× | 3.511 | 4.209 | 1.20× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.225 | – | – | 2.342 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.145 | 1.672 | 11.51× | 1.493 | 11.274 | 7.55× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 0.269 | 0.503 | 1.87× | 3.574 | 6.310 | 1.77× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.137 | 1.602 | 11.71× | 1.433 | 11.268 | 7.86× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.124 | 1.659 | 13.37× | 1.371 | 11.263 | 8.22× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.144 | 1.581 | 10.98× | 1.420 | 11.125 | 7.83× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.022 | 0.457 | 20.78× | 0.233 | 3.317 | 14.26× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.010 | 0.055 | 5.81× | 0.038 | 0.055 | 1.45× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.277 | 0.308 | 1.11× | 3.462 | 3.918 | 1.13× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 65.659 | – | – | 267.3 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.023 | 0.175 | 7.49× | 0.024 | 0.172 | 7.11× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 1.869 | 3.004 | 1.61× | 22.259 | 24.383 | 1.10× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.058 | 0.043 | 0.73× | 0.542 | 0.301 | 0.56× | 0.0e+00 |
| `statistics.compare` | timed alone | 33.677 | – | – | 235.7 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.016 | 0.189 | 11.83× | 0.016 | 0.191 | 12.00× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 3.893 | – | – | 45.962 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.094 | 0.476 | 5.07× | 0.242 | 1.731 | 7.17× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 0.486 | 1.360 | 2.80× | 7.420 | 12.489 | 1.68× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.058 | 0.029 | 0.50× | 0.591 | 0.229 | 0.39× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 0.764 | 2.836 | 3.71× | 10.268 | 35.962 | 3.50× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 0.513 | 1.974 | 3.85× | 7.456 | 24.507 | 3.29× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.014 | 0.054 | 3.81× | 0.097 | 0.274 | 2.83× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 32.663 | – | – | 273.5 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.074 | 0.288 | 3.91× | 0.312 | 0.679 | 2.18× | 1.8e-15 |
| `statistics.proportion_ci` | statsmodels | 0.012 | 0.084 | 6.94× | 0.008 | 0.084 | 10.98× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.023 | – | – | 20.872 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.049 | 0.441 | 9.05× | 0.047 | 0.445 | 9.54× | 0.0e+00 |
| `statistics.t_test` | SciPy | 0.089 | 0.452 | 5.07× | 0.399 | 1.006 | 2.52× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 0.415 | 1.227 | 2.96× | 3.254 | 10.483 | 3.22× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.012 | 0.003 | 0.21× | 0.014 | 0.003 | 0.23× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.014 | 0.004 | 0.28× | 0.018 | 0.006 | 0.34× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 1.005 | 0.768 | 0.76× | 9.643 | 7.626 | 0.79× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.012 | 0.007 | 0.59× | 0.073 | 0.026 | 0.36× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.248 | 0.136 | 0.55× | 2.441 | 1.461 | 0.60× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 0.466 | 4.008 | 8.61× | 5.583 | 44.092 | 7.90× | 0.0e+00 |
| `text.bradley_terry` | choix | 1.753 | 25.069 | 14.30× | 1.920 | 96.378 | 50.20× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 1.327 | 9.840 | 7.41× | 10.039 | 120.5 | 12.00× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 2.654 | 19.139 | 7.21× | 26.332 | 241.0 | 9.15× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.122 | – | – | 1.584 | – | – | – |
| `text.citation_recall` | timed alone | 0.124 | – | – | 1.277 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.118 | 0.003 | 0.03× | 0.928 | 0.010 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.033 | 0.054 | 1.62× | 0.305 | 0.574 | 1.88× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.039 | 0.011 | 0.28× | 0.334 | 0.065 | 0.19× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.152 | 0.262 | 1.72× | 1.587 | 2.983 | 1.88× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.092 | 0.088 | 0.96× | 0.258 | 0.220 | 0.85× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.034 | 0.024 | 0.72× | 0.184 | 0.152 | 0.83× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.265 | – | – | 2.077 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.198 | 0.344 | 1.74× | 1.804 | 3.505 | 1.94× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.032 | 0.086 | 2.68× | 0.087 | 0.635 | 7.28× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.020 | 0.016 | 0.81× | 0.158 | 0.124 | 0.78× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.077 | 0.115 | 1.50× | 0.835 | 1.144 | 1.37× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.183 | 0.123 | 0.67× | 1.717 | 0.933 | 0.54× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.038 | 0.009 | 0.24× | 0.245 | 0.074 | 0.30× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.046 | – | – | 0.401 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 0.943 | 2.368 | 2.51× | 9.842 | 22.785 | 2.32× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.116 | 0.100 | 0.86× | 1.160 | 0.974 | 0.84× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.093 | 0.566 | 6.06× | 0.227 | 0.805 | 3.55× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.042 | 0.010 | 0.24× | 0.394 | 0.152 | 0.39× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.068 | 0.082 | 1.20× | 0.172 | 0.315 | 1.83× | 2.9e-15 |
| `text.mauve` | timed alone | 3.156 | – | – | 569.2 | – | – | – |
| `text.meteor` | NLTK | 0.602 | 3.944 | 6.55× | 6.911 | 42.571 | 6.16× | 0.0e+00 |
| `text.model_score` | timed alone | 0.013 | – | – | 0.071 | – | – | – |
| `text.moverscore` | POT | 0.755 | 1.096 | 1.45× | 8.269 | 11.954 | 1.45× | 0.0e+00 |
| `text.perplexity` | NumPy formula | 0.041 | 0.010 | 0.24× | 0.337 | 0.063 | 0.19× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.051 | 0.003 | 0.05× | 0.425 | 0.005 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.265 | – | – | 2.626 | – | – | – |
| `text.rouge_1` | rouge-score | 0.610 | 2.332 | 3.82× | 4.410 | 23.203 | 5.26× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 0.609 | 2.298 | 3.77× | 4.611 | 22.798 | 4.94× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 0.375 | 2.764 | 7.37× | 4.430 | 28.708 | 6.48× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 0.842 | 4.269 | 5.07× | 10.036 | 43.074 | 4.29× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.014 | 0.004 | 0.28× | 0.031 | 0.007 | 0.23× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 0.523 | 9.176 | 17.55× | 6.715 | 254.7 | 37.93× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 0.344 | – | – | 0.429 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 0.617 | 5.571 | 9.03× | 7.403 | 57.502 | 7.77× | 0.0e+00 |
| `text.spice` | Python stdlib | 0.954 | 0.044 | 0.05× | 8.677 | 0.508 | 0.06× | 0.0e+00 |
| `text.ter` | sacreBLEU | 0.864 | 13.973 | 16.17× | 6.221 | 134.2 | 21.58× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.204 | – | – | 2.663 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.222 | – | – | 2.848 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.171 | 0.026 | 0.15× | 2.158 | 0.156 | 0.07× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 0.747 | – | – | 1.354 | – | – | – |
| `text.win_rate` | NumPy formula | 0.040 | 0.007 | 0.16× | 0.240 | 0.011 | 0.05× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.537 | 0.580 | 1.08× | 3.713 | 3.615 | 0.97× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.158 | 0.128 | 0.81× | 0.864 | 0.684 | 0.79× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.063 | 0.031 | 0.50× | 0.667 | 0.614 | 0.92× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 0.216 | 1.145 | 5.30× | 1.122 | 2.868 | 2.56× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.060 | 0.032 | 0.54× | 0.695 | 0.633 | 0.91× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.048 | 0.023 | 0.48× | 0.684 | 0.572 | 0.84× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.032 | 0.022 | 0.68× | 0.616 | 0.589 | 0.96× | 0.0e+00 |
