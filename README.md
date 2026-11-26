# EvalSuite

[![CI](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml/badge.svg)](https://github.com/mkcs28/evalsuite-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![Python](https://img.shields.io/pypi/pyversions/evalsuite-python)](https://pypi.org/project/evalsuite-python/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Unified, reproducible evaluation for machine learning, LLM and research.**

EvalSuite brings classification, regression, clinical, statistical, segmentation and object-detection
evaluation into one consistent, validated, documented framework.

> **On an older version? Upgrade once:** `pip install -U evalsuite-python`. From 1.1.2 on, EvalSuite tells you
> when a newer release is out (`evalsuite upgrade`).

> **Status: stable (1.2.0).** [Example notebooks](notebooks/README.md) (Colab) ·
> [in-browser tools](https://evalsuite-nine.vercel.app/tools) · [changelog](CHANGELOG.md)

**What's new**

- **1.2** — LLM evaluation and research-grade statistics: benchmark presets (GSM8K, MMLU, ARC, TruthfulQA,
  HumanEval) for the leaderboard, a judge report (win rates, position / length / self-preference bias, agreement
  with humans), rubric reports, an end-to-end RAG report that blames each failure on retrieval or generation, a
  single-file HTML report; sample-size and power planning, cluster-bootstrap intervals (`clusters=`), a fairness
  report (gaps with intervals, intersectional groups) and survival metrics (C-index, time-dependent AUC,
  integrated Brier score); a GitHub Action that comments results on pull requests.
- **1.1.2** — a once-a-day notice when a newer release exists, and a one-command upgrade (`evalsuite upgrade`
  or `es.upgrade()`); every README refreshed with current benchmark numbers.
- **1.1.1** — hardening: every engine metric fuzzed (no crashes on malformed input), NaN / Infinity preserved in
  every language, bug fixes in `evaluate()`'s task detection, 2.4–6× faster engine bootstrap intervals.
- **1.1** — the engine everywhere: clinical, calibration, retrieval and segmentation metrics with confidence
  intervals in JavaScript, C, R and Julia; an R source package; Colab notebooks; BLEU and HD95 faster than the
  Python path.
- **1.0** — `evaluate()` reports confidence intervals by default and tests against a baseline; an LLM
  leaderboard that knows when models are tied; scikit-learn, Hugging Face, Keras, Lightning, MLflow and W&B
  integrations; monitoring with drift alerts; TRIPOD+AI and model-card report cards; npm (WebAssembly), C, R
  and Julia bindings.
- **0.8** — a full Rust edition on crates.io ([`evalsuite`](https://crates.io/crates/evalsuite)) and the
  `evalsuite-cli` command. **0.6–0.7** — a compiled, multithreaded Rust core with identical results.
  **0.5** — LLM systems (safety, robustness, agents, code, long context, efficiency). **0.1–0.4** —
  classification, regression, clinical, statistics, vision and LLM evaluation.

## Installation

```bash
pip install evalsuite-python
```

To move to the newest release later, run `evalsuite upgrade` (or `pip install -U evalsuite-python`). pip never
replaces an installed package on its own, so EvalSuite prints a one-line notice, at most once a day, when a
newer release is on PyPI; the check runs in the background and never slows `import evalsuite`
(`EVALSUITE_NO_UPDATE_CHECK=1` turns it off; it is always off in CI). `es.check_for_update()` and
`evalsuite version --check` report the installed and latest versions.

The package is installed as `evalsuite-python` and imported as `evalsuite`:

```python
import evalsuite as es
```

EvalSuite has a **compiled core written in Rust**. pip installs a prebuilt wheel on Linux (x86_64, aarch64;
glibc and musl), macOS (Apple silicon, Intel) and Windows (x64, arm64) for Python 3.9 to 3.14, so no compiler
is needed. Anywhere else pip falls back to a pure-Python wheel that gives exactly the same results, more slowly.
`evalsuite._native.backend()` says which one is in use (`"rust"` or `"python"`), and
`EVALSUITE_PURE_PYTHON=1` selects the Python implementations on any install.

The core runs independent examples (sentences, images, classes, resamples) in parallel on all CPU cores; set
`EVALSUITE_NUM_THREADS` to cap the number of threads. Results never depend on the thread count: sums are always
taken in example order. Bootstrap functions (`bootstrap_ci`, `paired_bootstrap_test`, `compare`) and
`weat_effect_size` accept `resampling="native"` to draw resamples (or permutations) in the core (seeded
xoshiro256++): reproducible with `random_state` and faster on large inputs, but a different random stream from
the default NumPy one, so interval endpoints and p-values differ slightly (statistically equivalent).

### Rust

The same metrics are available as a pure-Rust crate, verified against this package on 36,514 recorded cases,
and as a standalone command (`cargo install evalsuite-cli`):

```toml
[dependencies]
evalsuite = "1.1"
```

```rust
let acc = evalsuite::classification::accuracy(&[1, 0, 1, 1], &[1, 0, 0, 1], None).unwrap(); // 0.75
```

```bash
evalsuite evaluate preds.csv --y-true label --y-pred pred --y-prob score
```

See [rust/evalsuite/README.md](rust/evalsuite/README.md) for the module list, performance and the few
documented differences, and [rust/evalsuite-cli/README.md](rust/evalsuite-cli/README.md) for the command
(`evalsuite upgrade` installs its latest release).

### JavaScript, C, R and Julia

One engine, the same numbers in every language ([bindings/README.md](bindings/README.md)):

```js
// npm install evalsuite@latest
const es = require("evalsuite");
es.evaluate("roc_auc", { y_true: [0, 0, 1, 1], y_prob: [0.1, 0.4, 0.35, 0.8], ci: "analytic" });
// { metric: "roc_auc", value: 0.75, ci_low: ..., ci_high: ..., ci_method: "delong" }
```

```r
library(evalsuite)  # install.packages from the release tarball or R-universe
es_metric("f1", y_true = c(0, 1, 1, 0), y_pred = c(0, 1, 0, 0))
```

C (`evalsuite.h`, `-levalsuite_c`) and Julia (`EvalSuite.metric("f1"; y_true=..., y_pred=...)`) take the same
arguments. Metrics: classification, clinical, calibration, regression, text, retrieval, segmentation, drift
and the leaderboard, with `ci: "analytic"` or `"bootstrap"` intervals.

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

## Confidence intervals and baselines by default

```python
result = es.evaluate(y_true, new_pred, y_prob=new_prob, baseline_pred=old_pred, baseline_prob=old_prob)
print(result.summary())
# Random forest vs logistic regression, breast-cancer test split (notebooks/01):
#   Accuracy     0.9430  (0.9049–0.9664)  -0.0044 vs baseline (p=1.000, mcnemar-exact)
#   ROC AUC      0.9833  (0.9637–1.0000)  -0.0097 vs baseline (p=0.285, delong)
#   Brier score  0.0375  (0.0216–0.0535)  -0.0001 vs baseline (p=0.989, paired-bootstrap)
```

Intervals are analytic by default (Wilson, DeLong, Student-t; no extra cost), `ci="bootstrap"` for any metric,
`ci=False` to turn them off; baseline tests are paired and Holm-adjusted.

## Report cards (TRIPOD+AI, model cards)

```python
card = es.report_card(y_true, y_prob, threshold=0.2, groups=site, group_name="site")
print(card.paragraph())  # results paragraph with intervals, ready for a paper
card.to_markdown()
card.to_latex()
# card.to_docx("performance.docx")  # Word: pip install "evalsuite-python[docx]"
```

Discrimination, calibration, Brier score, threshold measures, net benefit and subgroup performance, plus the
TRIPOD+AI checklist; `standard="model-card"` fills a model card's quantitative sections.

## LLM leaderboards

```python
board = es.leaderboard({"model-a": "runs/a.jsonl", "model-b": "runs/b.jsonl"}, scorer="gsm8k")
print(board.summary())  # ranks shared when differences are not significant (McNemar, Holm)
```

Scorers (exact match, GSM8K, boxed, choice, numeric, ...), per-category tables, contamination and LLM-judge
position checks; also `evalsuite leaderboard runs/*.jsonl` and the [in-browser tool](https://evalsuite-nine.vercel.app/tools).

## Monitoring

```python
mon = es.Monitor(
    window=1000, alerts={"roc_auc": ("<", 0.8), "psi": (">", 0.25)}, reference_scores=validation_probabilities
)
alerts = mon.update(y_true_batch, y_pred=pred_batch, y_prob=prob_batch)
```

Sliding-window metrics, PSI and Kolmogorov–Smirnov drift (also before labels arrive, with `observe`), alert
callbacks and `snapshot()` with intervals.

## Integrations

```python
from evalsuite import integrations as ei

cross_val_score(model, X, y, scoring=ei.sklearn_scorer("mcc"))
Trainer(..., compute_metrics=ei.hf_compute_metrics(["accuracy", "f1"]))
model.fit(X, y, callbacks=[ei.keras_callback(X_val, y_val)])
ei.log_mlflow(result)
ei.log_wandb(result)
```

## Planning a test set

```python
es.sample_size_two_proportions(0.80, 0.83, discordance=0.10)  # items to detect +3 points between two models
es.sample_size_proportion(0.9, 0.02)  # items for accuracy ± 2 points
es.sample_size_auc(0.85, prevalence=0.2, half_width=0.03)  # positives and negatives for an AUC interval
es.power_two_proportions(500, 0.80, 0.83, discordance=0.10)  # power with the items you have
```

## Clustered data

Several scans per patient, several questions per document or several sites make ordinary intervals too narrow.
Pass the cluster of each row and every interval and baseline test resamples whole clusters:

```python
patient = np.repeat(np.arange(40), 5)  # 200 rows, 5 per patient
result = es.evaluate(y_true, y_pred, clusters=patient)
es.bootstrap_ci("roc_auc", y_true, y_prob=y_prob, clusters=patient)
```

## Fairness

```python
group = np.where(np.arange(200) % 3 == 0, "A", "B")
report = es.fairness_report(y_true, y_pred, y_prob, groups={"group": group})
print(report.summary())  # per-group rates with CIs, parity / equalized-odds gaps with CIs, Holm-adjusted tests
```

Several attributes give intersectional groups (`groups={"sex": sex, "age": age}`); gap definitions match
Fairlearn's.

## Survival analysis

```python
time = np.arange(1, 201) * 0.5
event = np.arange(200) % 4 != 0
risk = -time + np.sin(np.arange(200))
es.concordance_index(time, event, risk)  # Harrell's C (ipcw=True: Uno's C)
es.time_dependent_auc(time, event, risk, [20, 40, 60])  # cumulative/dynamic AUC
```

`brier_score_survival` and `integrated_brier_score` take predicted survival probabilities. All agree with
scikit-survival to 1e-15 (Harrell's C is about 17× faster).

## LLM judges, rubrics and RAG

```python
judgements = [
    {"model_a": "m1", "model_b": "m2", "verdict": "A", "verdict_swapped": "A", "length_a": 120, "length_b": 80},
    {"model_a": "m2", "model_b": "m1", "verdict": "B", "verdict_swapped": "tie", "length_a": 60, "length_b": 150},
    {"model_a": "m1", "model_b": "m2", "verdict": "tie", "verdict_swapped": "tie", "length_a": 90, "length_b": 95},
]
print(es.judge_report(judgements, n_resamples=200).summary())
# win rates, Bradley–Terry, position consistency, length bias, self-preference, agreement with humans

rag = es.rag_report(
    [
        {"retrieved": ["d1", "d7"], "relevant": ["d1"], "answer": "Paris", "reference": "Paris"},
        {"retrieved": ["d3"], "relevant": ["d2"], "answer": "Rome", "reference": "Oslo"},
    ]
)
print(rag.summary())  # retrieval / grounding / answer metrics, and each failure blamed on a stage
```

Leaderboards know common benchmarks: `es.leaderboard(runs, benchmark="gsm8k", references=es.load_benchmark("gsm8k"))`
(also `mmlu`, `arc`, `truthfulqa`, `humaneval`; datasets via `pip install "evalsuite-python[benchmarks]"`).

## One HTML report

```python
page = es.html_report(
    {"Test set": es.evaluate(y_true, y_pred, y_prob=y_prob), "Notes": "Model v3."},
    title="Model evaluation",
    path="report.html",
)
```

Any EvalSuite result, Matplotlib figure or Markdown text becomes a section of one self-contained file (no
scripts, no external requests).

## Evaluation in pull requests

```yaml
- uses: mkcs28/evalsuite-python@v1.2.0
  with:
    predictions: predictions.csv
    y-true: label
    y-pred: pred
    baseline-pred: base_pred        # e.g. the main branch's predictions
    fail-on-regression: "true"
```

The action comments metrics with intervals and paired tests on the pull request and fails it on a significant
regression ([full example](examples/evalsuite-action.yml)).

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
evalsuite leaderboard runs/*.jsonl --scorer gsm8k -o leaderboard.md
evalsuite benchmark --quick
evalsuite upgrade                      # install the latest release
```

Input files can be CSV, TSV, Parquet or JSON. Output format follows `--format` or the `-o` extension
(text, json, csv, markdown, latex, html). Errors are reported in one line with exit code 2.

## Performance

Benchmarked against reference implementations on the same data (fastest of 5 runs; EvalSuite 1.1.2, Python 3.13,
NumPy 2.5, Linux x86_64 with 2 cores). **Every one of the 78 comparisons agrees with the reference** (largest
difference 2.5e-14), 77 are faster (the other, Hochberg's correction at 100,000 p-values, is level at 0.98×),
and the geometric-mean speed-up across 27 cases at three sizes is **3.86×**. `evaluate()` now also computes
the default confidence intervals inside these timings and is still 18× faster than scikit-learn's separate
metric calls. At 1,000,000 samples (pixels for segmentation; 1,000 images for detection and 50 for Hausdorff;
10,000 examples for the LLM cases), in alphabetical order:

| Case | Reference | EvalSuite (ms) | Reference (ms) | Speed-up |
| --- | --- | ---: | ---: | ---: |
| 10 classes: macro F1 | scikit-learn | 8.6 | 35.6 | **4.16×** |
| agreement: Krippendorff's alpha, interval (4 raters × 10000 items) | krippendorff | 1.6 | 3.2 | **2.01×** |
| binary: 8 label metrics via evaluate() | scikit-learn | 9.2 | 172.0 | **18.70×** |
| binary: ROC AUC | scikit-learn | 84.0 | 211.6 | **2.52×** |
| calibration: slope and intercept | statsmodels | 218.2 | 1217.0 | **5.58×** |
| clinical: diagnostic report (7 CIs) | statsmodels | 5.5 | 6.5 | **1.17×** |
| clinical: sensitivity, specificity, LR+, LR− | scikit-learn | 19.5 | 76.4 | **3.91×** |
| decision curve: 99 thresholds | NumPy loop | 169.5 | 196.8 | **1.16×** |
| detection: COCO evaluation (1000 images) | pycocotools | 40.2 | 831.8 | **20.71×** |
| multiple testing: Hochberg (n p-values) | statsmodels | 77.4 | 96.0 | **1.24×** |
| regression: MAE, MSE, RMSE, R² via evaluate() | scikit-learn | 9.7 | 11.4 | **1.18×** |
| retrieval: MRR, MAP@20, NDCG@10 (10000 queries) | ranx | 106.7 | 777.2 | **7.28×** |
| segmentation: Dice and IoU per class (n = pixels) | scikit-learn | 34.9 | 72.2 | **2.07×** |
| segmentation: Hausdorff distance (50 images) | SciPy | 7.6 | 18.5 | **2.43×** |
| statistics: Cramér's V (5×5 table) | SciPy | 0.0 | 0.3 | **15.93×** |
| statistics: Mann–Whitney U | SciPy | 127.5 | 385.5 | **3.02×** |
| statistics: Welch t-test | SciPy | 5.4 | 8.1 | **1.50×** |
| structured: JSON Schema compliance (10000 documents) | jsonschema | 133.2 | 295.5 | **2.22×** |
| systems: CodeBLEU n-gram terms (1000 programs) | – | 435.4 | – | – |
| systems: confidence AUROC (n answers) | scikit-learn | 181.8 | 242.6 | **1.33×** |
| systems: distribution-shift drop, Welch CI (n scores) | SciPy | 6.3 | 7.5 | **1.20×** |
| systems: invalid tool calls vs JSON Schema (10000 tasks) | jsonschema | 127.6 | 285.0 | **2.23×** |
| systems: latency p50 / p95 / p99 (n requests) | NumPy | 9.0 | 15.3 | **1.71×** |
| systems: maintainability index (1000 programs) | radon | 118.3 | 131.9 | **1.11×** |
| text: METEOR, exact and stem matches (10000 sentences) | NLTK | 92.4 | 493.8 | **5.34×** |
| text: ROUGE-1, ROUGE-2, ROUGE-L (10000 sentences) | rouge-score | 219.5 | 889.0 | **4.05×** |
| text: corpus BLEU and chrF (10000 sentences) | sacreBLEU | 266.7 | 2399.8 | **9.00×** |

**Every metric is benchmarked too** (`evalsuite benchmark --suite metrics`, re-run on 1.2.0): all 225 registered
metrics and statistics functions at 10,000 and 100,000 samples. 82 are compared with a reference library, 118
with an independent textbook formula and 25 (learned, judge-dependent, randomised, or whose reference library was
not installed) are timed alone; **all 200 comparisons agree**, and against the libraries the geometric-mean
speed-up is **5.50×** (164 of 164 measurements faster). Per-metric table in BENCHMARKS.md.

**Nearly every workload is faster than its reference.** The text metrics (BLEU, chrF, TER, ROUGE, METEOR, CIDEr,
CodeBLEU's n-gram terms, distinct-n), the whole COCO evaluation (IoU, score sort, matching, accumulation),
ROC AUC and average precision (one sort per column, shared by both inside `evaluate()`), the LLM-systems
consistency, groundedness, citation, toxicity and exposure metrics, Elo ratings, bootstrap resampling (including
whole bootstrap and jackknife runs of ROC AUC / AP), DeLong, QA exact match and F1,
Hausdorff and surface distances, calibration bins and the WEAT permutation test run in the compiled,
multithreaded Rust core; bootstrap intervals, paired bootstrap tests and
`compare()` evaluate confusion-matrix metrics from counts of each resample's (true, predicted) pairs instead of
copies of the data. `evaluate()` validates inputs once and builds the confusion
matrix once for all metrics, and integer labels are encoded without sorting. The rank tests rank the data once
with a single unstable sort; t-tests, χ², Fisher and McNemar compute their statistics directly, and
Shapiro–Wilk caches its weights per sample size. Percentiles select exactly from a small bracketed window
instead of partitioning the whole array; COCO matching runs across all images of a class at once; surface
distances work on boundary points only; TER, BLEU, ROUGE and METEOR reuse tokenisation, stems and
edit-distance rows. Every one of these is tested to give the reference's result. Reproduce on
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

Building from source needs a Rust toolchain ([rustup](https://rustup.rs)); the compiled core is in `rust/`
(`evalsuite-core`, pure Rust, and `evalsuite-py`, the PyO3 bindings), built by maturin.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # builds the Rust core into evalsuite._core
pytest --cov=evalsuite
EVALSUITE_PURE_PYTHON=1 pytest -q       # the same suite without the compiled core
ruff check . && ruff format --check . && mypy
cargo test --manifest-path rust/Cargo.toml -p evalsuite-core
```

Every compiled kernel is a port of a Python implementation that stays in the package as the reference and
fallback; `tests/test_native.py` checks that the two give identical results. See CONTRIBUTING.md.

## Links

- PyPI: https://pypi.org/project/evalsuite-python/
- crates.io: https://crates.io/crates/evalsuite (library), https://crates.io/crates/evalsuite-cli (command)
- Website and documentation: https://evalsuite-nine.vercel.app
- Website source: https://github.com/mkcs28/evalsuite

## Credits

Authors and maintainers: **Manoj Kumar C S** and **Nikhil D Bharadwaj**.

## License

Dual-licensed under either of

- MIT license ([LICENSE](LICENSE))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you explicitly state otherwise, any contribution you submit for inclusion in this
project shall be dual-licensed as above, without any additional terms or conditions.
