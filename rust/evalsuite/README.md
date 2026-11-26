# evalsuite

Evaluation metrics and statistics in pure Rust, for machine learning, NLP, computer vision, clinical models and
LLM systems. It is the Rust edition of [EvalSuite](https://github.com/mkcs28/evalsuite-python)
(`pip install evalsuite-python`): the same metrics, the same conventions and, case for case, the same numbers.

```toml
[dependencies]
evalsuite = "0.8"
```

For the terminal there is also the `evalsuite` command (`cargo install evalsuite-cli`): classification,
regression, diagnostic, calibration and text metrics straight from CSV, TSV or JSON files.

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
`serde_json`, `roxmltree` and `rustpython-parser` (Python source analysis for the code metrics; the default
`python` feature), plus `evalsuite-core`, the kernels shared with the Python package.

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
the crate's tests replay all 36,514 cases. Values agree to a relative tolerance of 1e-12 for most metrics and
1e-9 for the iterative ones, and every input the Python package rejects is rejected here too. Property tests check the invariants on top
(perfect predictions, bounds, symmetry, row-order and label-name invariance, p-value calibration under the null). The fixtures live
in the repository (not in the published crate); run `cargo test -p evalsuite --release` there.

## Differences from the Python package

- **Python source analysis** (`code_complexity`, `codebleu_python`, `python_syntax_ok`) parses with
  `rustpython-parser` and tokenizes as Python 3.12's `tokenize`, with radon's and CodeBLEU's rules; results
  match the Python package on Python 3.12+. Syntax newer than that parser supports counts as invalid, and
  `python_syntax_ok` covers the common compile-time errors (`return` / `yield` / `await` outside a function,
  `break` / `continue` outside a loop, module-level `nonlocal`), not every check `compile()` makes. For other
  languages `codebleu` takes the syntax and data-flow scores from your parser.
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

## Performance

Medians from `cargo bench -p evalsuite` (Criterion) against the Python package 0.8.5 with its compiled core,
on the same workloads and machine (2 vCPUs, Linux x86_64, Python 3.13, NumPy 2.5):

| Workload | Rust | Python | Speed-up |
| --- | ---: | ---: | ---: |
| Accuracy, 100,000 / 1,000,000 labels | 0.033 / 0.73 ms | 0.69 / 7.9 ms | 21× / 10.8× |
| Macro F1, 5 classes, 100,000 / 1,000,000 | 1.1 / 13.5 ms | 1.5 / 9.9 ms | 1.3× / 0.7× |
| ROC AUC, 100,000 / 1,000,000 | 6.9 / 121 ms | 9.4 / 125 ms | 1.4× / 1.0× |
| RMSE, 1,000,000 | 4.3 ms | 6.0 ms | 1.4× |
| R², 1,000,000 | 8.3 ms | 9.5 ms | 1.2× |
| Median absolute error, 1,000,000 | 8.5 ms | 17.5 ms | 2.1× |
| BCa bootstrap of MAE, 10,000 obs × 2,000 resamples | 100 ms | 362 ms | 3.6× |
| Mann–Whitney U, 10,000 | 0.68 ms | 0.83 ms | 1.2× |
| Welch t-test, 10,000 | 0.064 ms | 0.105 ms | 1.6× |
| BLEU, 1,000 sentences (1.1) | 6.0 ms | 7.7 ms | 1.3× |
| chrF, 1,000 sentences (1.0) | 9.4 ms | 11.4 ms | 1.2× |
| TER, 1,000 sentences (1.0) | 106 ms | 128 ms | 1.2× |
| ROUGE-L, 1,000 sentences | 5.2 ms | 5.0 ms | 1.0× |
| NDCG@10, 10,000 queries | 4.7 ms | 71 ms | 15.3× |
| MRR, 10,000 queries | 4.1 ms | 42 ms | 10.4× |
| Dice, 512 × 512 | 2.2 ms | 4.5 ms | 2.0× |
| HD95, 512 × 512 (1.1) | 1.3 ms | 2.1 ms | 1.6× |

Where the Python package is faster it already runs NumPy's SIMD loops, or the same compiled kernels with a
tokenizer written in C.

## License

Licensed under either of MIT or Apache-2.0, at your option.
