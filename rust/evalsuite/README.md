# evalsuite

Evaluation metrics and statistics in pure Rust, for machine learning, NLP, computer vision, clinical models and
LLM systems. It is the Rust edition of [EvalSuite](https://github.com/mkcs28/evalsuite-python)
(`pip install evalsuite-python`): the same metrics, the same conventions and, case for case, the same numbers.

```toml
[dependencies]
evalsuite = "0.8"
```

```rust
use evalsuite::{classification, regression, retrieval};

let acc = classification::accuracy(&[1, 0, 1, 1], &[1, 0, 0, 1], None)?;
assert_eq!(acc, 0.75);

let rmse = regression::rmse(&[3.0, -0.5, 2.0, 7.0], &[2.5, 0.0, 2.0, 8.0], None)?;
assert!((rmse - 0.6123724356957945).abs() < 1e-15);

let relevant = vec![retrieval::judgements(["a", "c"]), retrieval::judgements(["x"])];
let ranked = vec![vec!["b", "a", "c"], vec!["x", "y"]];
assert_eq!(retrieval::mrr(&relevant, &ranked, None)?.mean, 0.75);
# Ok::<(), evalsuite::Error>(())
```

No Python, no C, no `unsafe`. Dependencies: `rayon` (parallel resampling), `libm`, `regex` / `fancy-regex`,
`serde_json` and `roxmltree`, plus `evalsuite-core`, the kernels shared with the Python package.

## What is in it

| Module | Metrics |
| --- | --- |
| `classification` | accuracy, precision, recall, F1 / Fβ, specificity, NPV, Jaccard, balanced accuracy, MCC, Cohen's κ, Hamming loss, confusion matrix; `binary::` ROC AUC, average precision, ROC / PR curves, log loss, Brier, calibration curve, ECE; multiclass (OvR / OvO) and `multilabel::` variants, top-k accuracy |
| `regression` | MAE, MSE, RMSE, R², adjusted R², MAPE, sMAPE, MSLE, RMSLE, median AE, explained variance, max error, bias, quantile and Huber loss, RAE, RSE; `multioutput::` |
| `clinical` | sensitivity, specificity, PPV, NPV, likelihood ratios, diagnostic odds ratio, Youden's J, net benefit, decision curves |
| `calibration`, `uncertainty` | MCE, calibration slope / intercept, Hosmer–Lemeshow; risk–coverage, AURC, selective risk, adaptive ECE |
| `stats` | t-tests, Mann–Whitney (exact), Wilcoxon (exact), Kruskal–Wallis, Friedman, Shapiro–Wilk, χ², Cramér's V, Fisher's exact, McNemar, DeLong, effect sizes, p-value adjustment, proportion intervals, bootstrap (percentile / basic / BCa, stratified, paired) |
| `retrieval` | Precision / Recall / Hit rate @k, MRR, MAP, NDCG |
| `segmentation`, `detection` | Dice, IoU, mIoU, pixel accuracy, boundary IoU, Hausdorff / HD95, ASSD; box IoU, COCO mAP and the 12-number report, VOC AP |
| `generation` | BLEU, sentence BLEU, self-BLEU, chrF / chrF++, TER, ROUGE-N / L / Lsum, METEOR, CIDEr, distinct-n, cross-entropy, perplexity |
| `qa`, `semantic`, `spice` | exact match, token F1, majority vote, pass@k, answer extraction, benchmark accuracy; BERTScore, MoverScore (exact EMD), MAUVE, embedding similarity, learned-metric adapter; SPICE |
| `judge` | win rate, Bradley–Terry, Elo, Krippendorff's α, Fleiss' κ, judge–human agreement, position / verbosity / self-preference bias, rubric scores |
| `structured`, `json_schema` | JSON / XML validity, JSON Schema compliance (built-in Draft 2020-12 validator), required fields, tool selection / arguments, tool-call F1, API success, format and instruction compliance |
| `rag`, `factuality` | context precision / recall / relevance, latency, task success, failure attribution; faithfulness, hallucination rate, groundedness, citation precision / recall, claim verification, answer correctness / relevance, abstention |
| `agents`, `efficiency`, `longcontext`, `multilingual`, `robustness`, `safety`, `code` | the LLM-systems metrics: tool use, plans, state tracking, cost; TTFT, TPOT, throughput, energy, availability; needle-in-a-haystack, lost in the middle; language ID, bitext mining, parity, direct assessment; adversarial / noise robustness, consistency, prompt sensitivity; harmful / refusal / attack rates, toxicity, WEAT, PII detection, exposure; unit-test pass rate, resolved rate, CodeBLEU |

Every function validates its inputs and returns `evalsuite::Result`: `Error::InvalidInput` for bad input,
`Error::Undefined` where the metric does not exist for the data, as in the Python package. Rates come with
their 95 % Wilson interval, distributions with their percentiles.

## How it is verified

Every metric is checked against the Python package: `scripts/rust_fixtures.py` runs EvalSuite on randomised
inputs (ties, empty groups, degenerate and invalid cases included) and stores the inputs with the results, and
the crate's tests replay all 36,054 cases. Values agree to a relative tolerance of 1e-12 for most metrics and
1e-9 for the iterative ones, and every input the Python package rejects is rejected here too. The fixtures live
in the repository (not in the published crate); run `cargo test -p evalsuite --release` there.

## Differences from the Python package

- **Python-only parsing.** `code_complexity` (cyclomatic complexity, maintainability index) uses CPython's own
  `ast` and `tokenize` and is not ported. `syntax_validity_rate` takes your checker, and `codebleu` computes the
  n-gram and keyword-weighted n-gram parts itself while the syntax and data-flow match scores come from your
  parser.
- **No bundled linguistic resources.** METEOR and ROUGE take an optional stemmer function, and SPICE an
  optional synonym function, instead of NLTK's Porter stemmer and WordNet.
- **Random streams.** Bootstrap resampling uses the same seeded generator as the Python package's
  `resampling="native"` option. MAUVE's k-means, `add_typos` and WEAT's random re-partitions use it too, so
  they differ from the Python defaults, which draw from NumPy. Exact computations (WEAT with exact
  enumeration, exact Mann–Whitney and Wilcoxon) are identical.
- **`stats::compare`** uses paired bootstrap tests for every metric; the Python version switches to McNemar and
  DeLong where they apply (both are available on their own in `stats`).
- **Surface distances** support 2-D and 3-D masks.
- **Regular expressions** you pass (`format_compliance`, JSON Schema `pattern`) use the `regex` crate: linear
  time, no look-around or back-references. **JSON** is parsed by `serde_json`, which rejects the `NaN` and
  `Infinity` literals that Python's `json` accepts.
- `judge_agreement_ordinal` and `judge_agreement_continuous` need at least two ratings (the Python version
  fails with a SciPy error).
- The `evaluate()` pipeline, report objects and the metric registry are Python features and are not part of
  the crate.

## License

Licensed under either of MIT or Apache-2.0, at your option.
