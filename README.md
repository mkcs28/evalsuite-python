# EvalSuite

[![CI](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml/badge.svg)](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![Python](https://img.shields.io/pypi/pyversions/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Unified, reproducible evaluation for machine learning, LLM and research.**

EvalSuite brings classification, regression, clinical, statistical, segmentation and object-detection
evaluation into one consistent, validated, documented framework.

> **Status: stable (0.5.0).** Every item on the 0.1.0–0.5.0 roadmaps is implemented and verified, including SPICE.
> **LLM systems arrive in 0.5.0**: safety, robustness, calibration and uncertainty, agents and tool use,
> multilingual, code generation, long context and summarization, and inference efficiency and cost.

## Installation

```bash
pip install evalsuite-python
```

The package is installed as `evalsuite-python` and imported as `evalsuite`:

```python
import evalsuite as es
```

## Why EvalSuite

- **One consistent API.** Every metric returns a result object that behaves like a number and exports to
  JSON, pandas, Markdown and LaTeX.
- **Explicit conventions.** Averaging, label order, the positive class and zero-division behaviour are stated
  and recorded in every result, never silently assumed.
- **Validated.** Each metric is tested against scikit-learn where definitions coincide, plus property-based
  tests and edge cases.
- **Documented.** Every metric carries its definition, formula, range, input requirements and references,
  available programmatically through `metric_info()`.
- **Efficient.** `evaluate()` validates inputs once and computes the confusion matrix once for all metrics.
- **Lightweight.** Requires only NumPy, SciPy and pandas.

## Quick start

```python
import evalsuite as es

y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]
y_prob = [0.1, 0.9, 0.4, 0.2, 0.8, 0.6]

result = es.evaluate(y_true, y_pred, y_prob=y_prob)
print(result.summary())

result["f1"]  # MetricResult(f1=0.666667)
f"{result['mcc']:.3f}"  # '0.333'
result.to_latex(caption="Test-set performance")
result.to_dataframe()

es.f1(y_true, y_pred)  # individual metrics
es.roc_auc(y_true, y_prob)
es.metric_info("classification.mcc").formula  # documentation
es.list_metrics("regression")
```

## Comparing models

```python
result = es.compare(
    y_true,
    {"logistic": pred_lr, "forest": pred_rf, "boosting": pred_gb},
    probabilities={"logistic": prob_lr, "forest": prob_rf, "boosting": prob_gb},
    random_state=0,
)
print(result.summary())  # estimates with 95% CIs, paired tests, Holm-adjusted p-values
result.to_latex(label="tab:models")

es.bootstrap_ci("f1", y_true, y_pred, average="macro", random_state=0)  # BCa interval for any metric
es.accuracy_ci(y_true, y_pred)  # Wilson interval
es.delong_test(y_true, prob_a, prob_b)  # two correlated AUCs
es.mcnemar_test(y_true, pred_a, pred_b)
```

Every model is evaluated on the same bootstrap resamples, so differences are paired. Accuracy is compared
with McNemar's test, binary ROC AUC with DeLong's test and other metrics with a paired bootstrap test;
p-values are adjusted for multiple comparisons (Holm by default).

## Clinical evaluation

```python
report = es.diagnostic_report(y_true, y_pred)  # binary test vs reference standard
print(report)
# Sensitivity, specificity, PPV, NPV (Wilson CIs), LR+ and LR− (log CIs, Simel 1991),
# diagnostic odds ratio (Woolf), Youden's J, accuracy and prevalence

es.lr_positive(y_true, y_pred)
es.youden_j(y_true, y_pred)

dca = es.decision_curve(y_true, {"model": y_prob})  # net benefit vs treat all / treat none
dca.useful_range()  # thresholds where the model beats both
es.plot.decision_curve(y_true, {"model": y_prob})
```

Ratios that divide by zero are `inf` or NaN with a warning, never 0.

## Calibration

```python
es.calibration_report(y_true, y_prob)  # Brier, ECE, MCE, intercept, slope, Hosmer–Lemeshow
es.calibration_slope(y_true, y_prob)  # ideal 1; < 1 means predictions are too extreme
es.calibration_intercept(y_true, y_prob)  # ideal 0 (calibration-in-the-large)
es.hosmer_lemeshow(y_true, y_prob, n_groups=10)
```

## Statistical tests

```python
es.t_test(scores_a, scores_b)  # Welch by default; mean difference with CI and Cohen's d
es.paired_t_test(fold_scores_a, fold_scores_b)
es.wilcoxon_test(fold_scores_a, fold_scores_b)  # with matched-pairs rank-biserial r
es.mann_whitney_test(a, b)  # with rank-biserial r
es.friedman_test(scores_a, scores_b, scores_c)  # with Kendall's W
es.kruskal_wallis_test(g1, g2, g3)
es.shapiro_wilk_test(residuals)
es.chi_square_test(table)  # with Cramér's V
es.fisher_exact_test([[8, 2], [1, 5]])
es.adjust_pvalues(p_values, method="hochberg")  # also holm, bonferroni, bh, by
```

Every test returns a `TestResult` with the statistic, p-value and an effect size, computed with SciPy and
checked against SciPy and statsmodels in the test suite.

## Segmentation

```python
# label masks: one 2-D image, or images on the first axis (N, H, W) / (N, D, H, W), or a list of masks
es.dice(y_true, y_pred)  # macro over classes, pixel counts summed over the dataset
es.iou(y_true, y_pred, average=None)  # per class; classes absent from both masks are NaN, not 0
es.miou(y_true, y_pred, ignore_index=255)
es.dice(y_true, y_pred, aggregate="image")  # mean of per-image scores (medical imaging convention)
es.boundary_iou(y_true, y_pred)  # Cheng et al. 2021
es.hausdorff_distance(y_true, y_pred, percentile=95, spacing=(0.8, 0.8))  # HD95 in mm

report = es.segmentation_report(y_true, y_pred, class_names={0: "background", 1: "liver"})
print(report)  # mIoU, Dice, pixel accuracy, Boundary IoU, HD95, ASSD + per-class table
es.plot.segmentation(image, y_true[0], y_pred[0])  # prediction fill, truth outline
es.plot.per_class(report, metric="iou")  # per-class bars (also takes a detection report)
```

## Object detection

```python
y_true = [{"boxes": [[x1, y1, x2, y2], ...], "labels": [3, ...]}, ...]  # one dict per image
y_pred = [{"boxes": [...], "labels": [...], "scores": [...]}, ...]

report = es.detection_report(y_true, y_pred)  # the 12 COCO numbers + AP per class
report["map"], report["map_50"], report["mar_100"]
es.mean_average_precision(y_true, y_pred, iou_threshold=0.5)  # mAP@.50
es.average_precision_detection(y_true, y_pred, interpolation="voc")  # per class, VOC-style
es.box_iou(boxes_a, boxes_b, box_format="xywh")
y_true, y_pred = es.from_coco("instances_val.json", "detections.json")
es.plot.detection_pr(y_true, y_pred)
```

The COCO protocol (crowd regions, area ranges, max detections, 101-point interpolation) matches
`pycocotools` to the last digit in the test suite.

Models can be compared over the same images with intervals and paired tests, as for every other task:
`es.compare(y_true, {"unet": masks_a, "deeplab": masks_b})` resamples images; for detection it compares
mAP.

## LLM evaluation

```python
es.bleu(references, predictions)  # sacreBLEU-identical; also chrf, ter, rouge_l, meteor, cider
es.text_report(references, predictions)  # BLEU, chrF(++), TER, ROUGE, METEOR, EM, token F1 at once
es.bertscore(ref_token_embs, pred_token_embs)  # from your encoder's embeddings; also mauve, moverscore
es.model_score(refs, preds, scorer=comet_fn)  # any learned metric or judge, with intervals and compare
es.faithfulness(claim_verdicts)  # also hallucination_rate, citation_recall, answer_correctness
es.bradley_terry(comparisons, scale="elo")  # also win_rate, elo_ratings, krippendorff_alpha
es.ndcg_at_k(relevant, retrieved, k=10)  # also mrr, context_precision, context_recall
es.json_schema_compliance(outputs, schema)  # also tool_call_f1, instruction_compliance_rate
es.pass_at_k(n_samples, n_correct, k=10)  # also benchmark_accuracy(..., style="gsm8k")
es.compare(references, {"a": preds_a, "b": preds_b}, metrics=["bleu", "rouge_l"])  # paired over examples
es.plot.ratings(comparisons)  # Bradley–Terry leaderboard with intervals; also win_matrix, text_scores
es.spice(reference_tuples, candidate_tuples)  # scene-graph F-score; parser= for captions, synonyms="wordnet"
```

## LLM systems

Safety, robustness, uncertainty, agents, multilingual, code, long context and serving cost. The verdicts
these take (harmful or not, test passed or not, confidence) come from your own judge, classifier or sandbox;
EvalSuite turns them into rates with Wilson intervals and breakdowns. PII detection, CodeBLEU, cyclomatic
complexity and the maintainability index (identical to radon) are computed from the text itself, and
EvalSuite never executes generated code.

```python
import evalsuite as es

es.harmful_response_rate([True, False, False, True], harmful_prompt=[True, True, False, True])
es.over_refusal_rate(refused=[True, False, True], should_refuse=[True, False, False])
es.pii_leakage_rate(["mail me at a@b.io", "nothing here"])  # e-mail, phone, Luhn-checked cards, IPs
es.paraphrase_consistency([["Paris", "paris"], ["4", "5", "4"]])
es.aurc(correct=[True, True, False, True], confidence=[0.9, 0.8, 0.7, 0.4])  # also selective_risk
es.plan_adherence(plans=[["search", "read", "answer"]], executed=[["search", "answer"]])
es.language_parity({"en": [1, 1, 0, 1], "sw": [1, 0, 0, 1]})
es.codebleu(["def add(a, b):\n    return a + b"], ["def add(x, y):\n    return x + y"])
es.code_complexity(["def f(x):\n    return 1 if x else 0"])  # cyclomatic complexity + maintainability
es.needle_in_haystack(correct=[1, 0, 1, 1], context_lengths=[4000, 4000, 8000, 8000], depths=[0, 0.5, 0, 0.5])
es.time_to_first_token(request_times=[0.0, 1.0], first_token_times=[0.21, 1.35])
es.inference_cost([1200, 800], [300, 150], input_price=3.0, output_price=15.0)  # prices per 1M tokens
```

## Classification report

```python
report = es.classification_report(y_true, y_pred)
print(report)  # per-class precision, recall, F1, specificity, support + averages
report.save("report.html")  # also .csv .md .tex .json .txt
```

## Plots

```bash
pip install "evalsuite-python[plot]"   # adds matplotlib; importing evalsuite never loads it
```

```python
es.plot.roc(y_true, {"logistic": prob_lr, "forest": prob_rf})  # AUC in the legend
es.plot.pr(y_true, prob)  # AP and the prevalence line
es.plot.calibration(y_true, prob)  # reliability diagram, ECE, Brier
es.plot.confusion_matrix(y_true, y_pred, normalize="true")
es.plot.residuals(y_reg, pred_reg)  # or kind="predicted"
es.plot.comparison(es.compare(...))  # forest plot with CIs
```

Each function returns a matplotlib `Axes` (pass `ax=` to draw into your own figure). The numbers shown are
computed with EvalSuite's metrics, so plots and tables always agree. Several models get distinct colours
*and* line styles, so figures stay readable in greyscale print.

## Exports

Every result (`evaluate`, `classification_report`, `compare`, single metrics) exports to `summary()`,
`to_json()`, `to_csv()`, `to_dataframe()`, `to_markdown()`, `to_latex()` and `to_html()`, and `save(path)` picks
the format from the extension. HTML pages are standalone (inline CSS, no scripts) and escape all text.

## Command line

```bash
evalsuite evaluate predictions.csv --y-true label --y-pred pred --y-prob prob
evalsuite report predictions.csv --y-true label --y-pred pred -o report.html
evalsuite compare predictions.csv --y-true label --pred lr=pred_lr --pred rf=pred_rf \
    --prob lr=p_lr --prob rf=p_rf --plot comparison.png
evalsuite plot roc predictions.csv --y-true label --y-prob prob -o roc.png
evalsuite diagnostic predictions.csv --y-true label --y-pred pred   # sensitivity, LR+, DOR... with CIs
evalsuite calibration predictions.csv --y-true label --y-prob prob  # slope, intercept, ECE, HL
evalsuite plot decision predictions.csv --y-true label --y-prob prob -o dca.png
evalsuite segmentation true_masks.npy pred_masks.npy --ignore-index 255 --plot per_class.png
evalsuite detection instances_val.json detections.json --plot pr_curves.png
evalsuite metrics --category clinical
evalsuite info classification.mcc
evalsuite benchmark --quick
```

Input files can be CSV, TSV, Parquet or JSON. Output format follows `--format` or the `-o` extension
(text, json, csv, markdown, latex, html). Errors are reported in one line with exit code 2.

## Performance

Benchmarked against reference implementations on the same data (fastest of 5 runs; Python 3.12, NumPy 2.5,
Linux x86_64). **Overall: 63 of 63 rows agree with the reference** (largest
difference 1.8e-14), 42 are faster, and the geometric-mean speed-up across all
21 cases and three sizes is **2.12×**. At 1,000,000 samples (pixels for
segmentation; 1,000 images for detection and 50 for Hausdorff; 10,000 examples for the LLM cases), in
alphabetical order:

| Case | Reference | EvalSuite (ms) | Reference (ms) | Speed-up |
| --- | --- | ---: | ---: | ---: |
| 10 classes: macro F1 | scikit-learn | 23.8 | 135.5 | **5.7×** |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | krippendorff | 3.5 | 3.4 | 0.96× |
| binary: 8 label metrics via evaluate() | scikit-learn | 28.5 | 1023.8 | **36.0×** |
| binary: ROC AUC | scikit-learn | 187.4 | 327.3 | **1.7×** |
| calibration: slope and intercept | statsmodels | 205.7 | 1173.7 | **5.7×** |
| clinical: diagnostic report (7 CIs) | statsmodels | 19.6 | 4.9 | 0.25× |
| clinical: sensitivity, specificity, LR+, LR− | scikit-learn | 75.2 | 411.5 | **5.5×** |
| decision curve: 99 thresholds | NumPy loop | 165.0 | 201.9 | **1.2×** |
| detection: COCO evaluation (1000 images) | pycocotools | 838.7 | 848.6 | **1.0×** |
| multiple testing: Hochberg (n p-values) | statsmodels | 72.1 | 79.0 | **1.1×** |
| regression: MAE, MSE, RMSE, R² via evaluate() | scikit-learn | 22.9 | 10.6 | 0.47× |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | ranx | 170.4 | 861.0 | **5.1×** |
| segmentation: Dice and IoU per class (n = pixels) | scikit-learn | 34.3 | 256.0 | **7.5×** |
| segmentation: Hausdorff distance (50 images) | SciPy | 24.2 | 19.8 | 0.82× |
| statistics: Cramér's V (5×5 table) | SciPy | 0.2 | 0.2 | **1.0×** |
| statistics: Mann–Whitney U | SciPy | 353.5 | 340.6 | 0.96× |
| statistics: Welch t-test | SciPy | 16.1 | 8.3 | 0.52× |
| structured: JSON Schema compliance (10000 documents) | jsonschema | 144.6 | 330.5 | **2.3×** |
| text: corpus BLEU and chrF (10000 sentences) | sacreBLEU | 2789.8 | 3051.8 | **1.1×** |
| text: METEOR, exact and stem matches (10000 sentences) | NLTK | 488.1 | 626.7 | **1.3×** |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | rouge-score | 1290.1 | 1095.6 | 0.85× |

**Every metric is benchmarked too** (`evalsuite benchmark --suite metrics`): all 225 registered metrics and
statistics functions at 10,000 and 100,000 samples. 86 are compared with a reference library, 118 with an
independent textbook formula and 21 (learned, judge-dependent or randomised) are timed alone; **all 204
comparisons agree**, and against the libraries the geometric-mean speed-up is **2.65×** (137 of 172
measurements faster). Per-metric table in BENCHMARKS.md.

`evaluate()` validates inputs once and builds the confusion matrix once for all metrics, which is where most
of the speed-up comes from. Hypothesis tests use SciPy underneath, so they match its speed at best; rows
below 1× pay for input validation and the extra intervals and effect sizes EvalSuite reports. Reproduce on
your machine with `evalsuite benchmark`; full table (1k, 100k and 1M samples, peak memory) and notes in
[BENCHMARKS.md](https://github.com/mkcs28/evalsuite-python/blob/main/BENCHMARKS.md).

## Metrics in this release

**Classification** (binary, multiclass, multilabel; micro/macro/weighted/samples/per-class averaging;
sample weights): accuracy, balanced accuracy, precision, recall, specificity, NPV, F1, F-beta, Jaccard,
MCC, Cohen's kappa (unweighted, linear, quadratic), Hamming loss, confusion matrix, ROC AUC (binary,
one-vs-rest, one-vs-one), average precision, ROC and PR curves, log loss, Brier score, top-k accuracy,
calibration curve and expected calibration error.

**Clinical** (binary; `pos_label`; sample weights): sensitivity, specificity, PPV, NPV, positive and
negative likelihood ratios, diagnostic odds ratio, Youden's J, net benefit and decision curves, and a
diagnostic report with confidence intervals for all of them.

**Calibration**: calibration curve, Brier score, expected and maximum calibration error, calibration slope
and intercept, Hosmer–Lemeshow test.

**Statistics**: confidence intervals (bootstrap percentile/basic/BCa, Wilson, Clopper–Pearson, DeLong),
paired tests (McNemar, DeLong, paired bootstrap), t-tests (Welch, Student, paired), Mann–Whitney, Wilcoxon,
Kruskal–Wallis, Friedman, Shapiro–Wilk, χ², Fisher's exact; effect sizes (Cohen's d, Hedges' g, Cliff's
delta, Cramér's V); multiple-testing corrections (Bonferroni, Holm, Hochberg, Benjamini–Hochberg,
Benjamini–Yekutieli).

**Segmentation** (2-D and 3-D label masks; `ignore_index`; dataset or per-image aggregation): Dice, IoU,
mIoU, pixel accuracy, mean pixel accuracy, Boundary IoU, Hausdorff distance and HD95, average symmetric
surface distance (with pixel spacing), confusion matrix and a full report.

**Object detection**: box IoU (xyxy, xywh, cxcywh), COCO mAP@[.50:.95], mAP@.50, mAP@.75, mAP and mAR by
object size, AP per class with COCO or VOC interpolation, precision-recall curves, COCO file import.

**Regression** (single and multi-output; sample weights): MAE, MSE, RMSE, R², adjusted R², MAPE, sMAPE,
MSLE, RMSLE, median absolute error, explained variance, max error, mean bias error, quantile (pinball)
loss, Huber loss, relative absolute error, relative squared error.

**LLM evaluation** (0.4.0): BLEU, sentence BLEU, chrF/chrF++, TER, ROUGE-1/2/L/Lsum, METEOR, CIDEr-D, SPICE,
perplexity, cross-entropy, Distinct-n, Self-BLEU, MAUVE, BERTScore, embedding similarity, MoverScore, learned
metrics via `model_score`; faithfulness, hallucination rate, groundedness, citation precision/recall, claim
verification, knowledge consistency, answer correctness/relevance, abstention, exact match, token F1; win
rate, Bradley–Terry, Elo, Krippendorff's alpha, Fleiss' kappa, judge agreement and biases, rubric scores;
pass@k, majority vote, benchmark accuracy; Precision/Recall/Hit rate@k, MRR, MAP, NDCG, context precision,
recall and relevance; JSON / JSON Schema / XML validity, required fields, tool calls, instruction following.

**LLM systems** (0.5.0): harmful-response, refusal, over-refusal, attack-success and red-team success rates,
expected maximum toxicity, CrowS-Pairs stereotype preference, WEAT, PII leakage, memorization exposure,
policy violations; adversarial, typo-noise, OOD and code-switching robustness, distribution-shift drop,
paraphrase / counterfactual / cross-lingual consistency, response stability, contradiction rate, failure and
recovery rates, prompt sensitivity, truncation sensitivity; adaptive calibration error, risk-coverage curve,
AURC and E-AURC, selective risk, risk at coverage, coverage at risk, confidence AUROC; task completion,
invalid tool calls, tool-use efficiency, steps per task, plan adherence, state tracking, tool-failure
recovery, loop rate, human intervention, cost per task; language ID, bitext mining, language parity, direct
assessment, cultural appropriateness, language consistency; unit-test pass rate, syntax validity, static
analysis and security findings, execution success, test coverage, patch acceptance, SWE-bench resolved rate,
CodeBLEU, cyclomatic complexity and maintainability index, runtime efficiency; long-context accuracy by
length, needle-in-a-haystack grid, position accuracy, lost-in-the-middle gap, context utilization, summary
coverage, compression ratio, citation coverage, cross-document consistency; TTFT, TPOT, latency percentiles,
throughput, token usage, cost, resource utilization, energy per request, requests per second, availability.

## Conventions

- `average="auto"` resolves to `"binary"` for binary targets and `"macro"` otherwise; the resolved value
  is stored in `result.params["average"]`.
- Labels are sorted unless you pass `labels=[...]`; that order defines per-class outputs and the columns
  of 2-D `y_prob`.
- Undefined ratios (zero denominators) return 0 **with an `UndefinedMetricWarning`**; pass
  `zero_division=np.nan` to propagate NaN, or `0`/`1` to choose silently.
- Domain violations raise clear errors instead of being patched over (for example MAPE with zero targets).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest --cov=evalsuite
ruff check . && ruff format --check . && mypy
```

## Links

- PyPI: https://pypi.org/project/evalsuite-python/
- Website and documentation: https://evalsuite-nine.vercel.app
- Website source: https://github.com/mkcs28/evalsuite

## Credits

Authors and maintainers: **Manoj Kumar C S** and **Nikhil D Bharadwaj**.

## License

MIT. See [LICENSE](LICENSE).
