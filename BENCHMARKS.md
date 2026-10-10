# Benchmarks

Reproduce with `evalsuite benchmark` (all cases) or one suite: `--suite core` (classification and
regression), `--suite clinical` (clinical, calibration and statistics), `--suite vision` (segmentation and
object detection), `--suite llm` (text generation, retrieval, rater agreement, structured output) or
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
EvalSuite 0.4.0 benchmarks | Python 3.12.3 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | pycocotools 2.0.11 | Linux x86_64 | fastest of 5 runs
```

## Overall

Per suite and in total: how many rows agree with the reference (max |difference| ≤ 1e-9), how many are
faster, and the geometric mean and range of the speed-ups over all sizes.

| Suite | Cases | Rows | Match reference | Faster | Geo-mean speed-up | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 4 | 12 | 12/12 | 10/12 | 5.88× | 0.47×–49.07× |
| Clinical, calibration and statistics | 8 | 24 | 24/24 | 17/24 | 1.71× | 0.25×–11.13× |
| LLM evaluation | 6 | 18 | 18/18 | 10/18 | 1.48× | 0.83×–7.72× |
| Segmentation and object detection | 3 | 9 | 9/9 | 5/9 | 1.97× | 0.64×–10.39× |
| Overall | 21 | 63 | 63/63 | 42/63 | 2.12× | 0.25×–49.07× |

## All cases (alphabetical)

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.101 | 1.342 | 13.25× | 0.05 | 0.03 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 2.995 | 15.218 | 5.08× | 3.05 | 2.18 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 23.809 | 135.515 | 5.69× | 30.52 | 21.79 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 10 items) | 1,000 | krippendorff | 0.049 | 0.045 | 0.92× | 0.01 | 0.01 | 0.0e+00 |
| agreement: Krippendorff's alpha, interval (4 raters × 1000 items) | 100,000 | krippendorff | 0.412 | 0.387 | 0.94× | 0.26 | 0.69 | 3.3e-15 |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | 1,000,000 | krippendorff | 3.511 | 3.372 | 0.96× | 2.26 | 6.32 | 1.8e-14 |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.256 | 9.178 | 35.87× | 0.05 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 2.350 | 115.325 | 49.07× | 3.21 | 3.07 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 28.459 | 1023.820 | 35.98× | 31.54 | 30.53 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.167 | 1.460 | 8.73× | 0.09 | 0.08 | 1.1e-16 |
| binary: ROC AUC | 100,000 | scikit-learn | 17.642 | 27.624 | 1.57× | 9.16 | 7.64 | 0.0e+00 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 187.361 | 327.279 | 1.75× | 91.56 | 76.30 | 0.0e+00 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.559 | 2.578 | 4.61× | 0.10 | 0.60 | 2.8e-16 |
| calibration: slope and intercept | 100,000 | statsmodels | 15.730 | 115.193 | 7.32× | 8.46 | 58.00 | 5.6e-17 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 205.659 | 1173.684 | 5.71× | 83.99 | 579.85 | 3.9e-16 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.235 | 0.523 | 2.23× | 0.05 | 0.01 | 1.1e-14 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 1.951 | 1.602 | 0.82× | 3.05 | 0.29 | 7.1e-15 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 19.570 | 4.878 | 0.25× | 30.52 | 1.91 | 7.1e-15 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.309 | 3.440 | 11.13× | 0.05 | 0.03 | 4.4e-16 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 6.632 | 43.403 | 6.54× | 3.05 | 2.24 | 5.6e-17 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 75.166 | 411.527 | 5.47× | 30.52 | 22.32 | 4.4e-16 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.268 | 1.437 | 5.36× | 0.07 | 0.01 | 5.6e-17 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 11.350 | 15.891 | 1.40× | 6.87 | 0.29 | 5.6e-17 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 165.010 | 201.852 | 1.22× | 68.67 | 1.97 | 5.6e-17 |
| detection: COCO evaluation (10 images) | 1,000 | pycocotools | 13.097 | 21.589 | 1.65× | 0.54 | 1.33 | 0.0e+00 |
| detection: COCO evaluation (100 images) | 100,000 | pycocotools | 102.270 | 93.783 | 0.92× | 1.15 | 4.29 | 0.0e+00 |
| detection: COCO evaluation (1000 images) | 1,000,000 | pycocotools | 838.714 | 848.551 | 1.01× | 6.92 | 34.02 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.045 | 0.077 | 1.71× | 0.05 | 0.05 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 4.867 | 4.675 | 0.96× | 4.58 | 3.97 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 72.148 | 78.964 | 1.09× | 45.78 | 39.17 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.095 | 0.650 | 6.85× | 0.03 | 0.02 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.912 | 1.763 | 0.92× | 2.29 | 1.53 | 0.0e+00 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 22.854 | 10.633 | 0.47× | 22.89 | 15.26 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10 queries) | 1,000 | ranx | 0.182 | 1.407 | 7.72× | 0.01 | 0.04 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (1000 queries) | 100,000 | ranx | 29.821 | 77.001 | 2.58× | 0.49 | 3.12 | 0.0e+00 |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | 1,000,000 | ranx | 170.388 | 861.009 | 5.05× | 5.36 | 31.01 | 6.9e-18 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000 | scikit-learn | 0.314 | 2.882 | 9.19× | 0.16 | 0.10 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 100,000 | scikit-learn | 3.864 | 40.162 | 10.39× | 0.17 | 2.32 | 0.0e+00 |
| segmentation: Dice and IoU per class (n = pixels) | 1,000,000 | scikit-learn | 34.283 | 256.019 | 7.47× | 0.25 | 23.48 | 0.0e+00 |
| segmentation: Hausdorff distance (1 image) | 1,000 | SciPy | 0.515 | 0.397 | 0.77× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (24 images) | 100,000 | SciPy | 21.556 | 13.741 | 0.64× | 0.15 | 0.03 | 0.0e+00 |
| segmentation: Hausdorff distance (50 images) | 1,000,000 | SciPy | 24.165 | 19.832 | 0.82× | 0.15 | 0.03 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.259 | 0.409 | 1.58× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.259 | 0.275 | 1.07× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.231 | 0.232 | 1.00× | 0.00 | 0.00 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.640 | 0.564 | 0.88× | 0.16 | 0.14 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 33.722 | 38.490 | 1.14× | 15.45 | 13.93 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 353.520 | 340.573 | 0.96× | 154.50 | 139.24 | 0.0e+00 |
| statistics: Welch t-test | 1,000 | SciPy | 0.821 | 0.987 | 1.20× | 0.04 | 0.02 | 0.0e+00 |
| statistics: Welch t-test | 100,000 | SciPy | 2.097 | 1.382 | 0.66× | 3.06 | 1.53 | 0.0e+00 |
| statistics: Welch t-test | 1,000,000 | SciPy | 16.124 | 8.349 | 0.52× | 30.52 | 15.26 | 0.0e+00 |
| structured: JSON Schema compliance (10 documents) | 1,000 | jsonschema | 0.157 | 0.421 | 2.67× | 0.00 | 0.00 | 0.0e+00 |
| structured: JSON Schema compliance (1000 documents) | 100,000 | jsonschema | 14.393 | 33.995 | 2.36× | 0.05 | 0.02 | 0.0e+00 |
| structured: JSON Schema compliance (10000 documents) | 1,000,000 | jsonschema | 144.622 | 330.462 | 2.29× | 0.50 | 0.16 | 0.0e+00 |
| text: corpus BLEU and chrF (10 sentences) | 1,000 | sacreBLEU | 2.206 | 1.960 | 0.89× | 0.10 | 0.22 | 1.4e-14 |
| text: corpus BLEU and chrF (1000 sentences) | 100,000 | sacreBLEU | 255.898 | 250.494 | 0.98× | 0.74 | 21.91 | 1.4e-14 |
| text: corpus BLEU and chrF (10000 sentences) | 1,000,000 | sacreBLEU | 2789.790 | 3051.755 | 1.09× | 7.26 | 210.17 | 0.0e+00 |
| text: METEOR, exact and stem matches (10 sentences) | 1,000 | NLTK | 0.459 | 0.501 | 1.09× | 0.01 | 0.01 | 0.0e+00 |
| text: METEOR, exact and stem matches (1000 sentences) | 100,000 | NLTK | 52.275 | 58.965 | 1.13× | 0.12 | 0.04 | 0.0e+00 |
| text: METEOR, exact and stem matches (10000 sentences) | 1,000,000 | NLTK | 488.126 | 626.693 | 1.28× | 1.15 | 0.39 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10 sentences) | 1,000 | rouge-score | 1.145 | 0.947 | 0.83× | 0.01 | 0.01 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (1000 sentences) | 100,000 | rouge-score | 120.724 | 111.884 | 0.93× | 0.12 | 0.64 | 0.0e+00 |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | 1,000,000 | rouge-score | 1290.112 | 1095.609 | 0.85× | 1.16 | 6.55 | 0.0e+00 |

## Reading the results

- **Many metrics at once is where EvalSuite is fastest.** `evaluate()` validates the inputs once and builds
  the confusion matrix once, then derives all eight label metrics from it: 21–33× faster than eight
  separate scikit-learn calls. The same applies to sensitivity, specificity, LR+ and LR− (4–11×) and to
  segmentation Dice and IoU per class (5–17×), which come from one pixel confusion matrix.
- **LLM metrics match their references exactly** and run at about their speed: BLEU and chrF
  (1.09× sacreBLEU at 10,000 sentences), ROUGE (0.85× rouge-score), METEOR
  (1.28× NLTK), ranking metrics (5.1× ranx, which compiles with numba), JSON Schema
  (2.3× jsonschema) and Krippendorff's alpha (0.96×).
- **COCO detection evaluation matches pycocotools exactly and is faster** (1.1–1.9×). IoUs are
  computed once per image and class and reused for every area range and IoU threshold, and the greedy
  matching is vectorised over thresholds.
- **Calibration slope and intercept are 5–6× faster than statsmodels' GLM and use far less memory**
  (84 MiB vs 580 MiB at 1M samples).
- **Hausdorff distance (0.7–0.8×)** uses a Euclidean distance transform per class, which also supports
  anisotropic pixel spacing and HD95; the reference computes the exact point-set distance between boundary
  points of one class. Both give identical results.
- **Hypothesis tests call SciPy**, so they match its speed at best; the Welch t-test also computes the
  confidence interval and Cohen's d.
- **Slower rows, shown on purpose:** the diagnostic report (0.2–0.4× at large n) validates labels and reports
  ten measures where the reference computes seven intervals from counts; regression on a million values
  (0.5×) spends most of its ~20 ms checking every value for NaN, infinity, shape and dtype.

Numbers depend on the machine, Python and NumPy versions; run `evalsuite benchmark` on your own hardware.

## Every metric

One row for each of the 225 registered metrics and statistics functions (`evalsuite benchmark --suite metrics --sizes 10000 100000`; EvalSuite 0.5.0, Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, Linux x86_64, fastest of 5 runs). n is observations for classification, regression, clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 sentence pairs for bitext mining).

Each metric is compared with a reference library where one exists, otherwise with an independent implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised resampling) are timed alone.

| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | Geo-mean speed-up (library) | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Classification and regression | 35 | 26 | 9 | 0 | 35/35 | 52/52 | 3.99× | 1.20×–12.34× |
| Clinical, calibration and statistics | 32 | 21 | 6 | 5 | 27/27 | 23/42 | 1.68× | 0.45×–12.38× |
| LLM evaluation | 70 | 23 | 34 | 13 | 57/57 | 36/46 | 2.30× | 0.09×–45.03× |
| LLM systems | 77 | 7 | 68 | 2 | 75/75 | 12/14 | 1.82× | 0.33×–6.89× |
| Segmentation and object detection | 11 | 9 | 1 | 1 | 10/10 | 14/18 | 4.45× | 0.59×–14.90× |
| **All metrics** | 225 | 86 | 118 | 21 | 204/204 | 137/172 | 2.65× | 0.09×–45.03× |

| Metric | Compared with | EvalSuite ms (n=10,000) | Reference ms (n=10,000) | Speed-up (n=10,000) | EvalSuite ms (n=100,000) | Reference ms (n=100,000) | Speed-up (n=100,000) | Max |difference| |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `agents.agent_cost_per_task` | NumPy formula | 0.018 | 0.003 | 0.18× | 0.050 | 0.004 | 0.08× | 0.0e+00 |
| `agents.human_intervention_rate` | NumPy formula | 0.022 | 0.006 | 0.26× | 0.058 | 0.006 | 0.11× | 0.0e+00 |
| `agents.invalid_tool_call_rate` | jsonschema | 1.450 | 4.042 | 2.79× | 17.778 | 48.445 | 2.73× | 0.0e+00 |
| `agents.plan_adherence` | timed alone | 0.285 | – | – | 2.992 | – | – | – |
| `agents.state_tracking_accuracy` | Python stdlib | 0.168 | 0.016 | 0.09× | 1.802 | 0.100 | 0.06× | 0.0e+00 |
| `agents.steps_per_task` | NumPy formula | 0.099 | 0.003 | 0.03× | 0.141 | 0.003 | 0.02× | 0.0e+00 |
| `agents.task_completion_rate` | NumPy formula | 0.021 | 0.004 | 0.20× | 0.168 | 0.005 | 0.03× | 0.0e+00 |
| `agents.tool_failure_recovery_rate` | timed alone | 0.224 | – | – | 3.118 | – | – | – |
| `agents.tool_use_efficiency` | NumPy formula | 0.042 | 0.004 | 0.09× | 0.117 | 0.005 | 0.04× | 0.0e+00 |
| `agents.unnecessary_tool_call_rate` | Python stdlib | 1.165 | 0.845 | 0.72× | 13.522 | 10.004 | 0.74× | 0.0e+00 |
| `calibration.calibration_intercept` | statsmodels | 0.849 | 5.384 | 6.34× | 8.717 | 47.940 | 5.50× | 1.1e-16 |
| `calibration.calibration_slope` | statsmodels | 1.570 | 9.638 | 6.14× | 17.586 | 91.628 | 5.21× | 3.3e-16 |
| `calibration.maximum_calibration_error` | NumPy formula | 0.482 | 0.671 | 1.39× | 4.968 | 6.698 | 1.35× | 3.1e-16 |
| `classification.accuracy` | scikit-learn | 0.224 | 0.860 | 3.85× | 2.182 | 7.087 | 3.25× | 0.0e+00 |
| `classification.average_precision` | scikit-learn | 1.211 | 2.671 | 2.21× | 18.252 | 22.028 | 1.21× | 1.1e-16 |
| `classification.balanced_accuracy` | scikit-learn | 0.633 | 2.623 | 4.14× | 6.789 | 16.912 | 2.49× | 0.0e+00 |
| `classification.brier_score` | scikit-learn | 0.167 | 1.239 | 7.43× | 1.637 | 8.259 | 5.05× | 0.0e+00 |
| `classification.cohen_kappa` | scikit-learn | 0.234 | 2.147 | 9.16× | 2.376 | 13.231 | 5.57× | 0.0e+00 |
| `classification.expected_calibration_error` | NumPy formula | 0.489 | 0.753 | 1.54× | 4.607 | 8.859 | 1.92× | 1.9e-16 |
| `classification.f1` | scikit-learn | 0.785 | 3.782 | 4.82× | 7.143 | 21.742 | 3.04× | 0.0e+00 |
| `classification.fbeta` | scikit-learn | 0.478 | 4.929 | 10.31× | 2.653 | 15.368 | 5.79× | 0.0e+00 |
| `classification.hamming_loss` | scikit-learn | 0.737 | 4.090 | 5.55× | 11.630 | 36.589 | 3.15× | 0.0e+00 |
| `classification.jaccard` | scikit-learn | 0.245 | 2.632 | 10.72× | 2.253 | 15.311 | 6.80× | 0.0e+00 |
| `classification.log_loss` | scikit-learn | 0.826 | 4.203 | 5.09× | 9.793 | 39.423 | 4.03× | 0.0e+00 |
| `classification.mcc` | scikit-learn | 0.662 | 4.687 | 7.08× | 7.351 | 31.718 | 4.31× | 0.0e+00 |
| `classification.npv` | scikit-learn | 0.236 | 2.913 | 12.34× | 2.894 | 15.324 | 5.29× | 0.0e+00 |
| `classification.precision` | scikit-learn | 0.639 | 3.303 | 5.17× | 6.896 | 18.981 | 2.75× | 0.0e+00 |
| `classification.recall` | scikit-learn | 0.636 | 3.351 | 5.27× | 6.962 | 19.575 | 2.81× | 0.0e+00 |
| `classification.roc_auc` | scikit-learn | 12.330 | 29.444 | 2.39× | 168.2 | 247.0 | 1.47× | 0.0e+00 |
| `classification.specificity` | scikit-learn | 0.271 | 2.999 | 11.07× | 2.324 | 15.597 | 6.71× | 0.0e+00 |
| `classification.top_k_accuracy` | scikit-learn | 0.962 | 3.090 | 3.21× | 11.701 | 24.956 | 2.13× | 0.0e+00 |
| `clinical.diagnostic_odds_ratio` | NumPy formula | 0.226 | 0.054 | 0.24× | 2.433 | 0.445 | 0.18× | 0.0e+00 |
| `clinical.lr_negative` | scikit-learn | 0.223 | 2.628 | 11.79× | 2.217 | 17.369 | 7.83× | 8.3e-17 |
| `clinical.lr_positive` | scikit-learn | 0.224 | 2.697 | 12.04× | 2.196 | 16.924 | 7.71× | 8.9e-16 |
| `clinical.net_benefit` | NumPy formula | 0.274 | 0.027 | 0.10× | 2.400 | 0.222 | 0.09× | 0.0e+00 |
| `clinical.ppv` | scikit-learn | 0.238 | 2.950 | 12.38× | 2.225 | 15.020 | 6.75× | 0.0e+00 |
| `clinical.sensitivity` | scikit-learn | 0.243 | 2.929 | 12.05× | 2.146 | 14.764 | 6.88× | 0.0e+00 |
| `clinical.youden_j` | NumPy formula | 0.356 | 0.074 | 0.21× | 2.187 | 0.449 | 0.21× | 0.0e+00 |
| `code.code_complexity` | radon | 1.143 | 1.197 | 1.05× | 11.163 | 11.901 | 1.07× | 0.0e+00 |
| `code.codebleu` | codebleu (n-gram terms) | 7.669 | 9.799 | 1.28× | 73.239 | 80.815 | 1.10× | 0.0e+00 |
| `code.coverage_rate` | NumPy formula | 0.038 | 0.002 | 0.04× | 0.106 | 0.004 | 0.03× | 0.0e+00 |
| `code.execution_success_rate` | NumPy formula | 0.030 | 0.023 | 0.77× | 0.209 | 0.146 | 0.70× | 0.0e+00 |
| `code.patch_acceptance_rate` | NumPy formula | 0.022 | 0.005 | 0.20× | 0.161 | 0.007 | 0.04× | 0.0e+00 |
| `code.resolved_rate` | Python stdlib | 0.142 | 0.023 | 0.16× | 1.042 | 0.145 | 0.14× | 0.0e+00 |
| `code.runtime_efficiency` | NumPy formula | 0.063 | 0.006 | 0.10× | 0.135 | 0.005 | 0.04× | 0.0e+00 |
| `code.security_vulnerability_rate` | Python stdlib | 0.150 | 0.013 | 0.08× | 1.180 | 0.064 | 0.05× | 0.0e+00 |
| `code.static_analysis_violation_rate` | NumPy formula | 0.034 | 0.002 | 0.05× | 0.105 | 0.003 | 0.02× | 0.0e+00 |
| `code.syntax_validity_rate` | Python stdlib | 2.962 | 1.582 | 0.53× | 26.951 | 15.420 | 0.57× | 0.0e+00 |
| `code.unit_test_pass_rate` | NumPy formula | 0.597 | 0.481 | 0.81× | 6.154 | 5.000 | 0.81× | 0.0e+00 |
| `detection.average_precision_detection` | pycocotools | 3.539 | 25.982 | 7.34× | 42.900 | 150.6 | 3.51× | 0.0e+00 |
| `detection.box_iou` | NumPy formula | 0.061 | 0.032 | 0.53× | 11.676 | 7.928 | 0.68× | 0.0e+00 |
| `detection.mean_average_precision` | pycocotools | 4.083 | 25.745 | 6.31× | 47.155 | 140.1 | 2.97× | 0.0e+00 |
| `efficiency.availability` | NumPy formula | 0.019 | 0.004 | 0.20× | 0.078 | 0.005 | 0.06× | 0.0e+00 |
| `efficiency.energy_per_request` | NumPy formula | 0.016 | 0.004 | 0.27× | 0.052 | 0.008 | 0.15× | 0.0e+00 |
| `efficiency.inference_cost` | NumPy formula | 0.032 | 0.006 | 0.17× | 0.105 | 0.007 | 0.07× | 0.0e+00 |
| `efficiency.latency_percentiles` | NumPy formula | 0.145 | 0.045 | 0.31× | 0.202 | 0.049 | 0.24× | 0.0e+00 |
| `efficiency.requests_per_second` | NumPy formula | 0.012 | 0.004 | 0.31× | 0.049 | 0.005 | 0.10× | 0.0e+00 |
| `efficiency.resource_utilization` | NumPy formula | 0.172 | 0.004 | 0.02× | 0.217 | 0.002 | 0.01× | 0.0e+00 |
| `efficiency.throughput` | NumPy formula | 0.054 | 0.004 | 0.08× | 0.225 | 0.006 | 0.03× | 0.0e+00 |
| `efficiency.time_per_output_token` | NumPy formula | 0.128 | 0.005 | 0.04× | 0.246 | 0.008 | 0.03× | 0.0e+00 |
| `efficiency.time_to_first_token` | NumPy formula | 0.108 | 0.004 | 0.03× | 0.183 | 0.005 | 0.03× | 0.0e+00 |
| `efficiency.token_usage` | NumPy formula | 0.181 | 0.007 | 0.04× | 0.284 | 0.005 | 0.02× | 0.0e+00 |
| `long_context.citation_coverage` | NumPy formula | 0.621 | 0.167 | 0.27× | 5.986 | 1.630 | 0.27× | 0.0e+00 |
| `long_context.compression_ratio` | Python stdlib | 0.580 | 0.504 | 0.87× | 6.056 | 5.893 | 0.97× | 0.0e+00 |
| `long_context.context_utilization` | Python stdlib | 0.140 | 0.010 | 0.07× | 1.559 | 0.060 | 0.04× | 0.0e+00 |
| `long_context.cross_document_consistency` | Python stdlib | 0.745 | 0.023 | 0.03× | 8.485 | 0.184 | 0.02× | 0.0e+00 |
| `long_context.lost_in_the_middle` | NumPy formula | 0.052 | 0.016 | 0.31× | 0.154 | 0.025 | 0.16× | 0.0e+00 |
| `long_context.needle_in_haystack` | NumPy formula | 0.597 | 0.004 | 0.01× | 0.885 | 0.005 | 0.01× | 0.0e+00 |
| `long_context.position_accuracy` | NumPy formula | 0.098 | 0.039 | 0.39× | 0.210 | 0.053 | 0.25× | 0.0e+00 |
| `long_context.retrieval_accuracy_by_length` | NumPy formula | 0.102 | 0.005 | 0.05× | 0.352 | 0.005 | 0.01× | 0.0e+00 |
| `long_context.summary_coverage` | NumPy formula | 1.174 | 0.477 | 0.41× | 11.721 | 4.921 | 0.42× | 0.0e+00 |
| `multilingual.bitext_mining_accuracy` | NumPy formula | 0.056 | 0.033 | 0.58× | 0.121 | 0.080 | 0.66× | 0.0e+00 |
| `multilingual.code_switching_robustness` | NumPy formula | 0.073 | 0.005 | 0.07× | 0.345 | 0.005 | 0.01× | 0.0e+00 |
| `multilingual.cross_lingual_consistency` | Python stdlib | 0.704 | 0.026 | 0.04× | 8.324 | 0.230 | 0.03× | 0.0e+00 |
| `multilingual.cultural_appropriateness` | NumPy formula | 0.022 | 0.005 | 0.25× | 0.099 | 0.007 | 0.07× | 0.0e+00 |
| `multilingual.direct_assessment` | SciPy | 0.165 | 1.134 | 6.89× | 0.647 | 2.464 | 3.81× | 1.0e-17 |
| `multilingual.language_consistency` | NumPy formula | 0.170 | 0.009 | 0.05× | 0.773 | 0.009 | 0.01× | 0.0e+00 |
| `multilingual.language_id_accuracy` | scikit-learn | 0.272 | 0.649 | 2.39× | 1.313 | 0.440 | 0.33× | 0.0e+00 |
| `multilingual.language_parity` | NumPy formula | 0.134 | 0.046 | 0.34× | 0.242 | 0.028 | 0.12× | 0.0e+00 |
| `qa.exact_match` | SQuAD formula | 2.832 | 16.664 | 5.88× | 17.311 | 88.573 | 5.12× | 0.0e+00 |
| `qa.token_f1` | SQuAD formula | 2.247 | 9.762 | 4.34× | 24.069 | 92.207 | 3.83× | 0.0e+00 |
| `rag.context_precision` | NumPy formula | 0.749 | 0.790 | 1.06× | 7.548 | 8.238 | 1.09× | 0.0e+00 |
| `rag.context_recall` | NumPy formula | 0.549 | 0.505 | 0.92× | 5.489 | 5.042 | 0.92× | 0.0e+00 |
| `rag.context_relevance` | NumPy formula | 0.544 | 0.475 | 0.87× | 5.610 | 5.222 | 0.93× | 0.0e+00 |
| `rag.failure_attribution` | timed alone | 0.034 | – | – | 0.261 | – | – | – |
| `rag.latency_summary` | NumPy formula | 0.198 | 0.043 | 0.22× | 0.227 | 0.047 | 0.21× | 0.0e+00 |
| `rag.task_success_rate` | NumPy formula | 0.009 | 0.004 | 0.45× | 0.010 | 0.005 | 0.46× | 0.0e+00 |
| `reasoning.benchmark_accuracy` | timed alone | 0.406 | – | – | 4.026 | – | – | – |
| `reasoning.majority_vote_accuracy` | Python stdlib | 0.907 | 0.187 | 0.21× | 10.001 | 2.865 | 0.29× | 0.0e+00 |
| `reasoning.pass_at_k` | Python stdlib | 0.458 | 0.032 | 0.07× | 5.724 | 0.298 | 0.05× | 0.0e+00 |
| `regression.adjusted_r2` | NumPy formula | 0.086 | 0.024 | 0.27× | 0.838 | 0.304 | 0.36× | 0.0e+00 |
| `regression.explained_variance` | scikit-learn | 0.081 | 0.462 | 5.73× | 0.601 | 1.336 | 2.22× | 0.0e+00 |
| `regression.huber_loss` | NumPy formula | 0.117 | 0.042 | 0.36× | 1.294 | 0.665 | 0.51× | 0.0e+00 |
| `regression.mae` | scikit-learn | 0.036 | 0.266 | 7.30× | 0.268 | 0.559 | 2.08× | 0.0e+00 |
| `regression.mape` | scikit-learn | 0.059 | 0.301 | 5.11× | 0.507 | 0.964 | 1.90× | 0.0e+00 |
| `regression.max_error` | scikit-learn | 0.033 | 0.203 | 6.14× | 0.271 | 0.436 | 1.61× | 0.0e+00 |
| `regression.mean_bias_error` | NumPy formula | 0.034 | 0.009 | 0.27× | 0.214 | 0.116 | 0.54× | 0.0e+00 |
| `regression.median_absolute_error` | scikit-learn | 0.168 | 0.425 | 2.53× | 1.545 | 1.850 | 1.20× | 0.0e+00 |
| `regression.mse` | scikit-learn | 0.043 | 0.307 | 7.19× | 0.728 | 1.385 | 1.90× | 0.0e+00 |
| `regression.msle` | scikit-learn | 0.088 | 0.565 | 6.41× | 0.729 | 1.699 | 2.33× | 0.0e+00 |
| `regression.quantile_loss` | scikit-learn | 0.047 | 0.361 | 7.74× | 0.400 | 1.101 | 2.75× | 0.0e+00 |
| `regression.r2` | scikit-learn | 0.086 | 0.363 | 4.23× | 0.689 | 0.990 | 1.44× | 0.0e+00 |
| `regression.rae` | NumPy formula | 0.096 | 0.024 | 0.25× | 0.890 | 0.241 | 0.27× | 0.0e+00 |
| `regression.rmse` | NumPy formula | 0.045 | 0.009 | 0.20× | 0.277 | 0.069 | 0.25× | 0.0e+00 |
| `regression.rmsle` | NumPy formula | 0.093 | 0.049 | 0.53× | 0.810 | 0.726 | 0.90× | 0.0e+00 |
| `regression.rse` | NumPy formula | 0.104 | 0.021 | 0.21× | 1.127 | 0.297 | 0.26× | 0.0e+00 |
| `regression.smape` | NumPy formula | 0.081 | 0.030 | 0.37× | 0.803 | 0.447 | 0.56× | 0.0e+00 |
| `retrieval.hit_rate_at_k` | ranx | 0.586 | 8.973 | 15.32× | 5.835 | 80.953 | 13.87× | 0.0e+00 |
| `retrieval.mean_average_precision_at_k` | ranx | 0.655 | 9.151 | 13.96× | 6.510 | 81.839 | 12.57× | 3.5e-18 |
| `retrieval.mrr` | ranx | 0.570 | 8.804 | 15.43× | 5.932 | 83.890 | 14.14× | 0.0e+00 |
| `retrieval.ndcg_at_k` | ranx | 0.922 | 9.372 | 10.16× | 8.960 | 84.153 | 9.39× | 3.5e-18 |
| `retrieval.precision_at_k` | ranx | 0.568 | 9.116 | 16.06× | 5.927 | 83.681 | 14.12× | 0.0e+00 |
| `retrieval.recall_at_k` | ranx | 0.566 | 9.285 | 16.41× | 5.789 | 85.631 | 14.79× | 0.0e+00 |
| `robustness.adversarial_robustness` | NumPy formula | 0.052 | 0.004 | 0.08× | 0.349 | 0.004 | 0.01× | 0.0e+00 |
| `robustness.contradiction_rate` | NumPy formula | 0.033 | 0.021 | 0.63× | 0.272 | 0.141 | 0.52× | 0.0e+00 |
| `robustness.distribution_shift_drop` | SciPy | 0.141 | 0.781 | 5.53× | 0.213 | 0.777 | 3.65× | 6.9e-18 |
| `robustness.error_rate` | NumPy formula | 0.026 | 0.020 | 0.77× | 0.179 | 0.139 | 0.77× | 0.0e+00 |
| `robustness.invariance_violation_rate` | Python stdlib | 0.046 | 0.017 | 0.37× | 0.362 | 0.072 | 0.20× | 0.0e+00 |
| `robustness.noise_robustness` | NumPy formula | 0.046 | 0.004 | 0.09× | 0.349 | 0.005 | 0.01× | 0.0e+00 |
| `robustness.ood_accuracy` | NumPy formula | 0.055 | 0.004 | 0.08× | 0.360 | 0.006 | 0.02× | 0.0e+00 |
| `robustness.paraphrase_consistency` | Python stdlib | 0.782 | 0.023 | 0.03× | 11.292 | 0.367 | 0.03× | 0.0e+00 |
| `robustness.prompt_sensitivity` | NumPy formula | 0.098 | 0.007 | 0.07× | 0.535 | 0.013 | 0.02× | 0.0e+00 |
| `robustness.recovery_success_rate` | NumPy formula | 0.046 | 0.004 | 0.10× | 0.680 | 0.012 | 0.02× | 0.0e+00 |
| `robustness.response_stability` | Python stdlib | 1.193 | 0.589 | 0.49× | 12.031 | 6.053 | 0.50× | 0.0e+00 |
| `robustness.truncation_sensitivity` | NumPy formula | 0.158 | 0.105 | 0.66× | 0.293 | 0.151 | 0.52× | 0.0e+00 |
| `safety.attack_success_rate` | NumPy formula | 0.021 | 0.004 | 0.19× | 0.177 | 0.005 | 0.03× | 0.0e+00 |
| `safety.exposure` | NumPy formula | 1.224 | 0.022 | 0.02× | 13.444 | 0.150 | 0.01× | 0.0e+00 |
| `safety.harmful_response_rate` | NumPy formula | 0.051 | 0.006 | 0.11× | 0.329 | 0.007 | 0.02× | 0.0e+00 |
| `safety.over_refusal_rate` | NumPy formula | 0.050 | 0.006 | 0.13× | 0.336 | 0.008 | 0.02× | 0.0e+00 |
| `safety.pii_leakage_rate` | Python stdlib | 0.196 | 0.053 | 0.27× | 1.602 | 0.459 | 0.29× | 0.0e+00 |
| `safety.policy_violation_rate` | Python stdlib | 0.078 | 0.011 | 0.14× | 0.725 | 0.063 | 0.09× | 0.0e+00 |
| `safety.red_team_success_rate` | Python stdlib | 0.093 | 0.015 | 0.16× | 0.790 | 0.108 | 0.14× | 0.0e+00 |
| `safety.refusal_rate` | NumPy formula | 0.051 | 0.004 | 0.08× | 0.356 | 0.010 | 0.03× | 0.0e+00 |
| `safety.stereotype_preference` | NumPy formula | 0.030 | 0.007 | 0.22× | 0.098 | 0.009 | 0.10× | 0.0e+00 |
| `safety.toxicity_score` | NumPy formula | 1.128 | 0.009 | 0.01× | 11.319 | 0.055 | 0.00× | 0.0e+00 |
| `safety.weat_effect_size` | NumPy formula | 4.017 | 0.103 | 0.03× | 4.189 | 0.104 | 0.02× | 0.0e+00 |
| `segmentation.average_surface_distance` | SciPy | 0.983 | 0.584 | 0.59× | 12.303 | 7.575 | 0.62× | 0.0e+00 |
| `segmentation.boundary_iou` | timed alone | 0.362 | – | – | 4.208 | – | – | – |
| `segmentation.dice` | scikit-learn | 0.226 | 2.462 | 10.90× | 2.179 | 14.751 | 6.77× | 0.0e+00 |
| `segmentation.hausdorff_distance` | SciPy | 1.198 | 0.850 | 0.71× | 15.568 | 10.872 | 0.70× | 0.0e+00 |
| `segmentation.iou` | scikit-learn | 0.220 | 2.442 | 11.10× | 2.211 | 15.070 | 6.82× | 0.0e+00 |
| `segmentation.mean_pixel_accuracy` | scikit-learn | 0.205 | 2.717 | 13.27× | 2.243 | 14.714 | 6.56× | 0.0e+00 |
| `segmentation.miou` | scikit-learn | 0.248 | 2.644 | 10.68× | 2.185 | 14.250 | 6.52× | 0.0e+00 |
| `segmentation.pixel_accuracy` | scikit-learn | 0.050 | 0.746 | 14.90× | 0.407 | 4.255 | 10.47× | 0.0e+00 |
| `statistics.accuracy_ci` | statsmodels | 0.019 | 0.099 | 5.08× | 0.110 | 0.096 | 0.87× | 0.0e+00 |
| `statistics.adjust_pvalues` | statsmodels | 0.409 | 0.543 | 1.33× | 5.353 | 5.783 | 1.08× | 2.2e-16 |
| `statistics.bootstrap_ci` | timed alone | 411.6 | – | – | 3,855.5 | – | – | – |
| `statistics.chi_square_test` | SciPy | 0.587 | 0.274 | 0.47× | 0.605 | 0.274 | 0.45× | 0.0e+00 |
| `statistics.cliffs_delta` | SciPy | 2.059 | 2.732 | 1.33× | 26.690 | 29.391 | 1.10× | 2.8e-17 |
| `statistics.cohens_d` | NumPy formula | 0.079 | 0.050 | 0.63× | 0.590 | 0.340 | 0.58× | 0.0e+00 |
| `statistics.compare` | timed alone | 134.3 | – | – | 1,336.6 | – | – | – |
| `statistics.cramers_v` | SciPy | 0.467 | 0.488 | 1.05× | 0.326 | 0.303 | 0.93× | 0.0e+00 |
| `statistics.delong_test` | timed alone | 5.758 | – | – | 56.577 | – | – | – |
| `statistics.fisher_exact_test` | SciPy | 0.967 | 0.781 | 0.81× | 2.309 | 2.201 | 0.95× | 0.0e+00 |
| `statistics.friedman_test` | SciPy | 2.220 | 2.098 | 0.95× | 19.451 | 18.441 | 0.95× | 0.0e+00 |
| `statistics.hedges_g` | NumPy formula | 0.096 | 0.049 | 0.51× | 0.756 | 0.326 | 0.43× | 4.9e-12 |
| `statistics.kruskal_wallis_test` | SciPy | 4.338 | 4.660 | 1.07× | 44.117 | 43.957 | 1.00× | 0.0e+00 |
| `statistics.mann_whitney_test` | SciPy | 3.545 | 2.784 | 0.79× | 28.756 | 28.973 | 1.01× | 0.0e+00 |
| `statistics.mcnemar_test` | statsmodels | 0.108 | 0.094 | 0.87× | 0.330 | 0.462 | 1.40× | 0.0e+00 |
| `statistics.paired_bootstrap_test` | timed alone | 134.5 | – | – | 1,351.6 | – | – | – |
| `statistics.paired_t_test` | SciPy | 0.697 | 0.440 | 0.63× | 1.833 | 0.970 | 0.53× | 0.0e+00 |
| `statistics.proportion_ci` | statsmodels | 0.146 | 0.162 | 1.11× | 0.130 | 0.143 | 1.10× | 0.0e+00 |
| `statistics.roc_auc_ci` | timed alone | 2.761 | – | – | 27.557 | – | – | – |
| `statistics.shapiro_wilk_test` | SciPy | 0.695 | 0.642 | 0.92× | 0.761 | 0.662 | 0.87× | 0.0e+00 |
| `statistics.t_test` | SciPy | 1.076 | 0.824 | 0.77× | 3.542 | 1.991 | 0.56× | 0.0e+00 |
| `statistics.wilcoxon_test` | SciPy | 3.125 | 1.654 | 0.53× | 29.612 | 14.508 | 0.49× | 0.0e+00 |
| `text.abstention_accuracy` | NumPy formula | 0.020 | 0.005 | 0.24× | 0.025 | 0.006 | 0.23× | 0.0e+00 |
| `text.answer_correctness` | NumPy formula | 0.026 | 0.012 | 0.48× | 0.033 | 0.011 | 0.34× | 0.0e+00 |
| `text.answer_relevance` | NumPy formula | 1.742 | 1.415 | 0.81× | 18.676 | 14.585 | 0.78× | 0.0e+00 |
| `text.api_call_success_rate` | NumPy formula | 0.017 | 0.008 | 0.47× | 0.117 | 0.036 | 0.31× | 0.0e+00 |
| `text.bertscore` | NumPy formula | 0.475 | 0.282 | 0.59× | 5.312 | 3.106 | 0.58× | 0.0e+00 |
| `text.bleu` | sacreBLEU | 7.623 | 5.969 | 0.78× | 80.002 | 76.359 | 0.95× | 0.0e+00 |
| `text.bradley_terry` | choix | 2.155 | 44.807 | 20.79× | 4.222 | 190.1 | 45.03× | 8.7e-10 |
| `text.chrf` | sacreBLEU | 15.809 | 15.607 | 0.99× | 159.5 | 168.1 | 1.05× | 1.4e-14 |
| `text.cider` | pycocoevalcap | 23.188 | 35.977 | 1.55× | 247.6 | 396.4 | 1.60× | 0.0e+00 |
| `text.citation_precision` | timed alone | 0.208 | – | – | 2.501 | – | – | – |
| `text.citation_recall` | timed alone | 0.160 | – | – | 1.959 | – | – | – |
| `text.claim_verification_accuracy` | NumPy formula | 0.207 | 0.010 | 0.05× | 1.559 | 0.014 | 0.01× | 0.0e+00 |
| `text.constraint_satisfaction_rate` | NumPy formula | 0.114 | 0.134 | 1.17× | 0.494 | 0.993 | 2.01× | 0.0e+00 |
| `text.cross_entropy` | NumPy formula | 0.065 | 0.015 | 0.24× | 0.637 | 0.111 | 0.17× | 0.0e+00 |
| `text.distinct_n` | Python stdlib | 0.758 | 0.222 | 0.29× | 7.608 | 3.756 | 0.49× | 0.0e+00 |
| `text.elo_ratings` | Python stdlib | 0.552 | 0.145 | 0.26× | 1.306 | 0.362 | 0.28× | 0.0e+00 |
| `text.embedding_similarity` | NumPy formula | 0.059 | 0.036 | 0.61× | 0.286 | 0.226 | 0.79× | 0.0e+00 |
| `text.extra_content_rate` | timed alone | 0.479 | – | – | 3.321 | – | – | – |
| `text.faithfulness` | NumPy formula | 0.342 | 1.046 | 3.05× | 3.127 | 6.517 | 2.08× | 0.0e+00 |
| `text.fleiss_kappa` | statsmodels | 0.467 | 0.158 | 0.34× | 4.914 | 1.286 | 0.26× | 0.0e+00 |
| `text.format_compliance` | Python stdlib | 0.036 | 0.030 | 0.83× | 0.270 | 0.212 | 0.79× | 0.0e+00 |
| `text.groundedness` | NumPy formula | 0.652 | 0.241 | 0.37× | 6.765 | 2.200 | 0.33× | 0.0e+00 |
| `text.hallucination_rate` | NumPy formula | 0.300 | 0.129 | 0.43× | 2.959 | 1.428 | 0.48× | 0.0e+00 |
| `text.instruction_compliance_rate` | NumPy formula | 0.041 | 0.014 | 0.33× | 0.385 | 0.096 | 0.25× | 0.0e+00 |
| `text.instruction_retention` | timed alone | 0.066 | – | – | 0.548 | – | – | – |
| `text.json_schema_compliance` | jsonschema | 1.596 | 5.879 | 3.68× | 15.881 | 39.550 | 2.49× | 0.0e+00 |
| `text.json_validity` | Python stdlib | 0.215 | 0.179 | 0.83× | 3.922 | 3.318 | 0.85× | 0.0e+00 |
| `text.judge_agreement` | scikit-learn | 0.142 | 0.923 | 6.49× | 0.364 | 1.281 | 3.52× | 0.0e+00 |
| `text.knowledge_consistency` | Python stdlib | 0.089 | 0.027 | 0.30× | 0.651 | 0.221 | 0.34× | 0.0e+00 |
| `text.krippendorff_alpha` | krippendorff | 0.130 | 0.145 | 1.12× | 0.497 | 0.481 | 0.97× | 2.9e-15 |
| `text.mauve` | timed alone | 5.511 | – | – | 867.6 | – | – | – |
| `text.meteor` | NLTK | 5.387 | 6.357 | 1.18× | 61.231 | 64.797 | 1.06× | 0.0e+00 |
| `text.model_score` | timed alone | 0.019 | – | – | 0.104 | – | – | – |
| `text.moverscore` | POT | 20.463 | 2.037 | 0.10× | 233.3 | 21.991 | 0.09× | 1.1e-16 |
| `text.perplexity` | NumPy formula | 0.068 | 0.014 | 0.21× | 0.598 | 0.107 | 0.18× | 0.0e+00 |
| `text.position_consistency` | NumPy formula | 0.093 | 0.005 | 0.05× | 0.755 | 0.010 | 0.01× | 0.0e+00 |
| `text.required_field_accuracy` | timed alone | 0.423 | – | – | 4.494 | – | – | – |
| `text.rouge_1` | rouge-score | 3.154 | 3.556 | 1.13× | 30.701 | 36.723 | 1.20× | 0.0e+00 |
| `text.rouge_2` | rouge-score | 3.152 | 3.579 | 1.14× | 31.005 | 36.772 | 1.19× | 0.0e+00 |
| `text.rouge_l` | rouge-score | 4.056 | 4.597 | 1.13× | 41.765 | 47.098 | 1.13× | 0.0e+00 |
| `text.rouge_lsum` | rouge-score | 6.358 | 7.168 | 1.13× | 65.866 | 72.932 | 1.11× | 0.0e+00 |
| `text.rubric_score` | NumPy formula | 0.023 | 0.007 | 0.32× | 0.044 | 0.012 | 0.28× | 0.0e+00 |
| `text.self_bleu` | sacreBLEU | 10.420 | 14.342 | 1.38× | 256.2 | 390.6 | 1.52× | 0.0e+00 |
| `text.self_preference_bias` | timed alone | 1.201 | – | – | 0.771 | – | – | – |
| `text.sentence_bleu` | sacreBLEU | 8.840 | 9.023 | 1.02× | 82.806 | 89.790 | 1.08× | 0.0e+00 |
| `text.spice` | Python stdlib | 1.425 | 0.063 | 0.04× | 14.865 | 0.799 | 0.05× | 0.0e+00 |
| `text.ter` | sacreBLEU | 37.723 | 23.399 | 0.62× | 374.3 | 229.0 | 0.61× | 3.6e-15 |
| `text.tool_argument_accuracy` | timed alone | 0.688 | – | – | 4.419 | – | – | – |
| `text.tool_call_f1` | timed alone | 0.755 | – | – | 4.962 | – | – | – |
| `text.tool_selection_accuracy` | Python stdlib | 0.570 | 0.055 | 0.10× | 3.547 | 0.264 | 0.07× | 0.0e+00 |
| `text.verbosity_bias` | timed alone | 1.442 | – | – | 2.401 | – | – | – |
| `text.win_rate` | NumPy formula | 0.105 | 0.014 | 0.14× | 0.415 | 0.018 | 0.04× | 0.0e+00 |
| `text.xml_validity` | Python stdlib | 0.981 | 0.967 | 0.99× | 5.760 | 5.597 | 0.97× | 0.0e+00 |
| `uncertainty.adaptive_calibration_error` | NumPy formula | 0.809 | 0.256 | 0.32× | 3.055 | 0.852 | 0.28× | 0.0e+00 |
| `uncertainty.aurc` | NumPy formula | 0.323 | 0.039 | 0.12× | 3.012 | 0.758 | 0.25× | 0.0e+00 |
| `uncertainty.confidence_accuracy_correlation` | scikit-learn | 1.473 | 1.930 | 1.31× | 6.332 | 4.169 | 0.66× | 1.1e-16 |
| `uncertainty.coverage_at_risk` | NumPy formula | 0.359 | 0.039 | 0.11× | 3.139 | 0.761 | 0.24× | 0.0e+00 |
| `uncertainty.risk_at_coverage` | NumPy formula | 0.287 | 0.027 | 0.09× | 3.119 | 0.718 | 0.23× | 0.0e+00 |
| `uncertainty.selective_risk` | NumPy formula | 0.259 | 0.027 | 0.10× | 3.015 | 0.727 | 0.24× | 0.0e+00 |
