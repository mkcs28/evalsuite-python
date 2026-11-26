# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-11-26

### Added
- **The engine everywhere**: npm, C, R and Julia now cover clinical metrics (sensitivity, PPV, NPV, likelihood
  ratios, diagnostic odds ratio, Youden's J, net benefit), calibration (slope, intercept, ECE, MCE,
  Hosmer–Lemeshow), retrieval (NDCG, MRR, MAP, precision / recall / hit rate at k) and segmentation (Dice, IoU),
  in addition to the 1.0 set.
- **Confidence intervals in every language**: `ci: "analytic"` (Wilson for proportions, DeLong for ROC AUC,
  bootstrap otherwise) or `ci: "bootstrap"` (seeded BCa) returns `ci_low` / `ci_high` alongside the value.
- **Leaderboard in the engine** (`leaderboard`): Wilson intervals, McNemar tests with Holm adjustment and
  significance-aware ranks, available from JavaScript, C, R and Julia.
- **R package distribution**: a self-contained source package (Rust sources vendored, offline build, Windows
  support) checked with `R CMD check`, attached to each release and published to the `r-package` branch for
  R-universe. **Julia**: a BinaryBuilder recipe for `EvalSuite_jll`.
- **Example notebooks** (Colab): classification quickstart, clinical TRIPOD+AI, LLM leaderboard, monitoring,
  vision.

### Changed
- BLEU 1.3× and HD95 1.6× faster than the Python reference path (memoised 13a tokenization, byte-level
  tokenizer, per-segment vocabularies, packed n-gram matching; row-parallel surface scan, allocation-free label
  checks). Results unchanged.

## [1.0.0] - 2026-11-25

EvalSuite 1.0: answers with uncertainty by default, and one engine in every language.

### Added
- **Confidence intervals by default in `evaluate()`** (`ci="analytic"`: Wilson for proportions, DeLong for
  ROC AUC, t-intervals for error metrics; `ci="bootstrap"` or `ci=False`) and **baseline tests**
  (`baseline_pred=` / `baseline_prob=`: paired difference with interval and p-value, Holm-adjusted). Summaries,
  Markdown, LaTeX and DataFrame exports show the interval and the comparison.
- **LLM leaderboard harness** (`es.leaderboard`, `evalsuite leaderboard`): scorers (exact match, GSM8K, boxed,
  choice, numeric, ...), Wilson / bootstrap intervals, McNemar or paired-bootstrap tests with Holm adjustment,
  significance-aware ranks, per-category tables, extraction-failure, contamination (n-gram overlap) and
  judge position-consistency checks; Markdown, LaTeX, HTML, CSV and JSON output.
- **Integrations** (`evalsuite.integrations`): scikit-learn scorers, Hugging Face `compute_metrics`, Keras and
  PyTorch Lightning callbacks, MLflow and Weights & Biases logging (optional extras).
- **Monitoring** (`es.Monitor`, `evalsuite::monitor`): sliding-window metrics, PSI and Kolmogorov–Smirnov drift
  against reference scores, threshold alerts with callbacks, `snapshot()` with intervals.
- **Report cards** (`es.report_card`): the TRIPOD+AI performance section (sample, discrimination, calibration,
  Brier, threshold measures, net benefit, subgroups) with intervals, a results paragraph and the checklist; model
  cards from `context`; Markdown, LaTeX, HTML, JSON, CSV and Word (`[docx]` extra).
- **One engine, every language**: `evalsuite::bridge` (JSON in, JSON out); the `evalsuite` npm package
  (WebAssembly, Node and browser); a C API (`evalsuite-capi`, `evalsuite.h`); R (`bindings/r`) and Julia
  (`bindings/julia`) packages. CI checks that every language gives the engine's value.

### Changed
- Faster Rust text metrics: an allocation-free 13a tokenizer and parallel sentence statistics (TER 1.9×,
  chrF 1.6×, BLEU 1.2× versus 0.8.5); HD95 percentiles by selection instead of sorting.

## [0.8.5] - 2026-11-25

The Rust edition is complete, gains a command-line tool, and the Python retrieval metrics run on it.

### Added
- **`code_complexity`, full CodeBLEU and a Python syntax check in the crate** (`code::code_complexity`,
  `code::codebleu_python`, `code::python_syntax_ok`, `code::strip_python_comments`): Python source is parsed
  with `rustpython-parser` into a CPython-shaped tree and tokenized as Python 3.12's `tokenize`, with radon's
  complexity, Halstead and raw-line rules and CodeBLEU's sub-tree and data-flow matching. Every registered metric
  of the Python package now has a Rust counterpart.
- **`evalsuite-cli`** (`cargo install evalsuite-cli`): the `evalsuite` command evaluates CSV, TSV, JSON and
  JSON-lines files (`evaluate` for classification or regression, `diagnostic`, `calibration`, `text`,
  `metrics`) as text, JSON, CSV or Markdown, with the same numbers as the Python CLI.
- **`clinical::diagnostic_report`** in the crate: the 2×2 table and every measure with its interval.
- **Property tests** for the crate (invariants independent of the fixtures) and **Criterion benchmarks**
  (`cargo bench -p evalsuite`); results in BENCHMARKS.md.
- **Python retrieval metrics on Rust**: Precision, Recall and Hit rate @k, MRR, MAP and NDCG validate and look
  up ids with Python's own dict and set semantics from Rust and repeat the arithmetic exactly (bit-identical,
  checked by `tests/test_native.py` on mixed id types and graded judgements): 3–3.5× faster.

### Changed
- Faster crate kernels: label encoding in one hashed pass, unweighted accuracy without encoding (11–21× the
  Python speed), selection instead of sorting for the median (2×), parallel BLEU tokenization.
- The crate's parity fixtures cover 36,514 cases (460 new ones for Python source analysis); `crates.yml`
  publishes `evalsuite-core`, `evalsuite` and `evalsuite-cli` together.

### Fixed
- `proportion_ci(method="wilson")` (Python and Rust) keeps the estimate inside the interval at 0 and n
  successes: rounding gave an upper bound of 0.9999999999999999 at n of n (Rust: a lower bound of 3e-17 at 0).

### Not moved to Rust
- Regression metrics: NumPy already handles the arrays and most of the time is input validation that stays in
  Python, so a Rust call would gain little.
- Judge statistics (Bradley–Terry, Krippendorff's alpha) and other NumPy-vectorised groups: their results depend
  on NumPy's SIMD `log` / `exp` and BLAS matrix products, which Rust cannot reproduce bit for bit on every CPU,
  and the package's rule is that native results equal the Python ones exactly.

## [0.8.0] - 2026-11-25

A full Rust edition of EvalSuite: the new [`evalsuite`](https://crates.io/crates/evalsuite) crate on crates.io.
The Python package is unchanged apart from its version number.

### Added
- **`evalsuite` crate** (pure Rust, `#![forbid(unsafe_code)]`): every registered metric and statistics function
  of the Python package except `code_complexity` — classification, regression, clinical, calibration and
  uncertainty, hypothesis tests and bootstrap intervals, retrieval, segmentation, detection, text generation,
  QA and reasoning, embedding metrics, SPICE, LLM-judge statistics, structured outputs and tool use, RAG,
  factuality, and the LLM-systems groups (agents, efficiency, long context, multilingual, robustness, safety,
  code). It includes a built-in JSON Schema validator, PII detectors and an exact earth mover's distance.
- **Verified against the Python package**: `scripts/rust_fixtures.py` records 36,054 randomised cases (ties,
  degenerate and invalid inputs included) and the crate's tests replay them all, to a relative tolerance of
  1e-12 for most metrics (1e-9 for iterative ones); every input Python rejects is rejected too. CI runs them.
- **Publishing workflow** `.github/workflows/crates.yml`: on a version tag it tests, then publishes
  `evalsuite-core` and `evalsuite` with the `CARGO_REGISTRY_TOKEN` secret, behind the `crates-io` environment.
  Versions already on crates.io are skipped.

### Differences in the crate (documented in its README)
- `code_complexity` is not ported (it relies on CPython's `ast` and `tokenize`); `syntax_validity_rate` takes a
  checker and `codebleu` takes the syntax / data-flow match scores from the caller's parser.
- METEOR, ROUGE and SPICE take optional stemmer / synonym functions instead of NLTK's Porter stemmer and WordNet.
- MAUVE's clustering, `add_typos` and WEAT's random re-partitions use the crate's seeded generator (the Python
  package's `resampling="native"` stream), not NumPy's.
- `stats::compare` uses paired bootstrap tests throughout; surface distances support 2-D and 3-D masks.

## [0.7.1] - 2026-11-25

More statistics, vision and QA work in the compiled core. Every default result is unchanged (bit-identical to
the Python fallback, checked by `tests/test_native.py`); the new `resampling="native"` option of
`weat_effect_size` is opt-in.

### Added
- **Bootstrap of ROC AUC and average precision** (`bootstrap_ci`, `paired_bootstrap_test`, `compare`): every
  resample's curve in one parallel compiled call. With unit weights a resample is one linear pass over the data's
  presorted order with multiplicities (counts are exact integers, so no re-sort); with weights each resample is
  gathered and stably sorted. The BCa jackknife reuses one presort for every left-out block. A 2,000-resample BCa
  interval on 100,000 observations: 28.9 s → 5.9 s (4.0 s with `resampling="native"`, 2 cores).
- **DeLong test and DeLong AUC intervals**: one sort per model yields all three mid-rankings (exact
  half-integers) instead of three SciPy `rankdata` calls: 3×.
- **QA metrics**: SQuAD answer normalisation, exact match and token F1 in the core for ASCII text (Unicode text
  keeps Python's case mapping): `token_f1` 6.3×, `exact_match` 2.4×, and `majority_vote_accuracy` normalises in
  one batch.
- **Hausdorff distance / HD95 / average symmetric surface distance**: surfaces extracted in parallel and exact
  nearest-surface distances from a dense-bitmap shell search with an exact k-d tree fallback, using the same
  arithmetic as the direct comparison: 1.2–6× faster on 2-D and 3-D masks, identical results.
- **Calibration** (`calibration_curve`, `expected_calibration_error`, and the metrics built on them): bin search
  and the three weighted bin sums in one pass: 4.4× at 1,000,000 samples.
- **WEAT permutation test**: exact enumeration of the splits in the core (53×, identical p-values), and
  opt-in `resampling="native"` for the random splits (26×; seeded and thread-count independent, a different
  random stream from NumPy's).
- Tool-call metrics reuse one JSON encoder for their call keys (byte-identical keys, 1.3×).
- Parity tests for every new kernel, and the thread-count invariance test now covers the new parallel paths.

### Not ported
- No concordance / C-index metric exists in EvalSuite, so there was nothing to port (adding one is a feature,
  not a port).
- Turning words into ids for the text metrics: a compiled interner gave no gain, because the Python 3.9 stable
  ABI the wheels target has to copy every string, which costs as much as Python's dict lookup.
- EvalSuite has no fairness module; the multilingual per-language split is a dictionary lookup.
- Multiple-testing corrections (`adjust_pvalues`): a compiled sort was slower than NumPy 2's SIMD sort at
  10,000–100,000 p-values and only marginally faster at 1,000,000, so the NumPy code stays.

## [0.7.0] - 2026-11-24

More of EvalSuite runs in the compiled Rust core. Every result is unchanged: each new kernel is bit-identical to
the Python implementation it replaces (checked by `tests/test_native.py`), which stays as the fallback.

### Added
- **ROC AUC and average precision** in the core: one stable sort per column, columns in parallel, with NumPy's
  pairwise summation reproduced exactly. `evaluate()` now shares that sort between ROC AUC and average precision
  (cached on its context), so it sorts each probability column once instead of twice. `evaluate()` with
  probabilities: 2.5× faster (binary) to 4.3× (5-class, 1M samples); `roc_auc` 2.6×, `average_precision` 2×.
- **LLM-systems consistency metrics** (`paraphrase_consistency`, `response_stability`,
  `cross_document_consistency`, `cross_lingual_consistency`, `majority_vote_accuracy`): answer groups are
  counted in the core (equal pairs from answer counts, O(m) instead of O(m²) pair comparisons): 1.4–4.8×.
- **Rows of numbers read directly by the core** (no per-row NumPy arrays): `groundedness` 5×,
  `citation_coverage` 7.8×, `toxicity_score` 6×, `exposure` 3×.
- **Pairwise preferences**: sequential `elo_ratings` 3.6×, and the Bradley–Terry win matrix.
- Parity tests for every new kernel (ties, sample weights, micro/macro/weighted/per-class, one-vs-one,
  non-string answers, invalid inputs raising the same errors).

### Not ported, on purpose
- Retrieval metrics: already 11–15× faster than ranx in Python; mapping document ids to the core cost more than
  it saved.
- Krippendorff's alpha: already one vectorised NumPy pass; a compiled version was only 1.5× faster and could not
  reproduce NumPy's matrix-multiply rounding, so results would have changed in the last digit.
- Fleiss' kappa and judge agreement are single vectorised integer passes; metrics built on Python's `ast`, JSON
  or regular expressions (syntax validity, tool-call keys, answer normalisation) stay in Python.

## [0.6.1] - 2026-11-24

Finishes the Rust port. Results are unchanged (bit-identical to the Python reference) unless you opt in to
`resampling="native"`.

### Added
- Whole COCO evaluation in the core: IoU matrices (with the crowd rule), stable descending score sort, ignored
  ground truth, greedy matching, precision/recall accumulation and the 101-point interpolation, in parallel
  over classes (`detection_report`, `mean_average_precision`).
- Compiled CIDEr-D, CodeBLEU (n-gram and keyword-weighted n-gram terms) and distinct-n.
- Multithreading: every per-example kernel (BLEU, chrF, TER, ROUGE, METEOR, CIDEr, CodeBLEU, COCO, resample
  counting) runs on all cores; `EVALSUITE_NUM_THREADS` caps the threads. Sums stay in example order, so the
  thread count never changes a result.
- Opt-in `resampling="native"` for `bootstrap_ci`, `paired_bootstrap_test` and `compare`: resample indices are
  drawn in the core (seeded xoshiro256++, one stream per resample, written straight into the index array).
  About 3× faster bootstrap on 100,000 observations; reproducible with `random_state` and independent of the
  thread count, but a different random stream from the default `"numpy"`. The default is unchanged.
- Parity tests for the new kernels, tied scores and user areas in COCO, thread-count invariance and native
  resampling.

### Fixed
- The ported float sums follow the running interpreter's built-in `sum()` (compensated summation on Python
  3.12+, sequential before), so CIDEr and CodeBLEU agree with the Python path to the last bit on every Python.

## [0.6.0] - 2026-11-24

EvalSuite now has a **compiled core written in Rust**. The public API is unchanged and every result is identical:
each kernel is a direct port of the Python implementation it replaces, which stays as the reference and as the
fallback, and the two are tested against each other for every kernel.

### Added
- `evalsuite._core`, built from `rust/` (crates `evalsuite-core`, pure Rust, and `evalsuite-py`, the PyO3
  bindings) with maturin. Binary wheels for Linux (x86_64, aarch64; glibc and musl), macOS (arm64, x86_64) and
  Windows (x64, arm64), each built for the stable ABI so one wheel serves Python 3.9 to 3.14.
- A pure-Python wheel (`py3-none-any`): pip installs it only where no binary wheel fits, and EvalSuite then runs
  its Python implementations with the same results. `EVALSUITE_PURE_PYTHON=1` selects them on any install;
  `evalsuite._native.backend()` reports `"rust"` or `"python"`.
- Compiled kernels: TER (beam edit distance with block shifts), BLEU / self-BLEU / sentence-BLEU statistics,
  chrF and chrF++ statistics, ROUGE-N overlaps, ROUGE-L and ROUGE-Lsum longest common subsequences, METEOR
  alignment, COCO detection matching, and bootstrap resample counting.
- Bootstrap intervals, paired bootstrap tests and `compare()` evaluate confusion-matrix metrics (accuracy,
  balanced accuracy, Cohen's kappa, F1, F-beta, Jaccard, MCC, NPV, precision, recall, specificity) from counts
  of each resample's (true, predicted) pairs instead of the resampled observations, and the BCa jackknife from
  the full counts minus each block's: identical values, no per-resample copies of the data.
- `tests/test_native.py`: parity of every kernel with its Python reference; CI runs the whole suite with and
  without the core and requires the binary wheel, the pure wheel and the sdist to give identical results.

### Changed
- Benchmarks: overall geometric-mean speed-up 3.29× → **3.70×** (81 of 81 rows faster), per-metric speed-up
  against reference libraries 4.97× → **5.99×** (172 of 172 faster); every comparison still agrees.
- Against 0.5.5: TER about 12× faster (now about 19× sacreBLEU), self-BLEU 10×, chrF 7×, BLEU and ROUGE 2–5×,
  METEOR 2×, bootstrap confidence intervals about 5×, paired bootstrap tests and `compare()` about 2×.
  Corpus BLEU and chrF together are now 8.3× sacreBLEU, METEOR 4.9× NLTK and COCO evaluation 2.3× pycocotools.
- Build backend: maturin (was hatchling); building from the sdist needs a Rust toolchain.

## [0.5.5] - 2026-11-09

Every benchmark row is now faster than its reference library: 81 of 81 in the main benchmark (geometric mean
3.07× → **3.29×**) and 172 of 172 per-metric library comparisons (4.67× → **4.97×**). No API changes; results
are unchanged (all 81 + 204 comparisons agree with the reference, as before).

### Changed
- Latency percentiles (and every LLM-systems latency/step/token summary): exact selection from a small window
  bracketed by a sorted sample instead of partitioning the whole array; bit-identical to `np.percentile`
  (0.69× → 1.1–2.0× NumPy). Falls back to a full partition if a bracket ever misses.
- COCO detection: COCOeval's greedy matching runs across all images of a class at once (0.96× → 1.85×
  pycocotools; identical matches).
- Shapiro–Wilk: SciPy's algorithm (SciPy ≥ 1.12) with Royston's weights cached per sample size, bit-identical
  (0.90× → 11× SciPy); older SciPy keeps its own Fortran implementation.
- MoverScore: cost matrix computed exactly as POT's `ot.dist`, POT's solver called without redundant marginal
  checks (0.9× → 1.4× POT).
- Surface distances: erosion by array shifts and direct nearest-point search for small boundaries (k-d trees
  for large ones); average surface distance 0.75× → 1.2× and Hausdorff 1.1× → 1.9× SciPy.
- METEOR: Porter stems memoised per word (0.74× → 3.5× NLTK).
- Krippendorff's alpha and CodeBLEU: lower fixed cost on tiny inputs (now faster at every size).
- README and BENCHMARKS.md rerun on 0.5.5.

## [0.5.4] - 2026-11-09

Speed release and a second licence. No API changes; every result is unchanged (all 81 benchmark rows and
204 per-metric comparisons agree with their reference libraries, as before).

### Added
- Dual licence: EvalSuite is now available under **MIT OR Apache-2.0**, at your option (`LICENSE`,
  `LICENSE-APACHE`). Both texts ship in the wheel and sdist; the PyPI licence expression is `MIT OR Apache-2.0`.

### Changed
- Overall benchmark speed-up 2.06× → **3.07×** (75 of 81 rows faster, was 58); per-metric speed-up against
  reference libraries 2.80× → **4.67×** (165 of 172 faster, was 138).
- Classification: integer labels are found and encoded without sorting or `searchsorted`, and binary label sets
  come straight from min/max; the confusion matrix counts without unit weights (binary `evaluate()` 37× → 98×
  faster than scikit-learn, 10-class macro F1 3.4× → 16.5×, diagnostic report 0.22× → 1.1×).
- Statistics: Wilcoxon, Mann–Whitney, Kruskal–Wallis and Friedman rank once with an unstable sort and bincount
  rank sums (2–4× faster than SciPy, identical p-values including SciPy's exact/asymptotic switching); paired
  t-test, χ² and Cramér's V, McNemar, proportion and accuracy intervals compute directly; Fisher's exact test sums
  the hypergeometric pmf in log space (5–7× faster, within 1e-11 of SciPy).
- Text: TER reuses edit-distance rows across shift candidates (0.61× → 1.5× sacreBLEU); BLEU, self-BLEU and ROUGE
  (0.86× → 1.9× rouge-score)
  build n-grams in C and memoise tokenisation like sacreBLEU and rouge-score; chrF has a tighter matching loop;
  Fleiss' kappa (0.23× → 6×), Krippendorff's alpha (0.77× → 2.0×) and language-ID accuracy (0.38× → 3–7×) are vectorised.
- MoverScore uses POT's network simplex when POT is installed and an exact assignment formulation otherwise
  (0.07× → 0.9× POT).
- Hausdorff and surface distances use k-d trees on the surface points instead of whole-image distance transforms.
- Regression `evaluate()` shares squared errors between MSE, RMSE and R², and task inference checks a short leading
  chunk for fractional values first (0.69× → 1.8× scikit-learn).
- Benchmarks and README tables rerun on 0.5.4.

## [0.5.3] - 2026-11-09

Performance release; benchmarks rerun. No API changes.

### Changed
- `t_test` computes the t statistic and its p-value directly (Student-t tail via `scipy.special.stdtr`) from
  one pass over each sample instead of calling `scipy.stats.ttest_ind` and recomputing means and variances;
  identical to SciPy within 1e-15 and now faster than it (was 0.43×).
- `latency_percentiles` and every LLM-systems latency/step/token summary take min, max and all percentiles
  from a single `np.partition` instead of separate `np.percentile` calls (identical values, ~1.4× faster).
- Statistical tests no longer copy inputs that are already float64.
- CI: Python 3.9 tests run on ubuntu-22.04 (ubuntu-latest no longer ships 3.9); GitHub Actions updated
  (checkout 7.0.1, setup-python 7.1.0, upload-artifact 7.0.2, download-artifact 8.0.2).
- README and BENCHMARKS.md: all suites and the per-metric table rerun on 0.5.3.

## [0.5.2] - 2026-11-07

Security and performance hardening. No API changes.

### Security
- The built-in PII e-mail detector no longer backtracks quadratically: `detect_pii` / `pii_leakage_rate`
  on 60 kB of adversarial text went from ~6 s to ~10 ms.
- `codebleu`, `code_complexity` and `syntax_validity_rate` handle source nested too deeply for CPython's
  parser (MemoryError / RecursionError) instead of crashing: invalid for syntax validity, an
  `InputValidationError` for complexity, no syntax match for CodeBLEU.
- Internal `assert` type checks replaced by explicit errors, so behaviour is the same under `python -O`.
- CI and release workflows pin every action to a commit SHA; Dependabot keeps them and the dependencies current.
- Documented that user-supplied regular expressions (`format_compliance`) run on a backtracking engine.

### Performance
- `import evalsuite` no longer imports SciPy; it is loaded on first use (import time ~0.6 s → ~0.07 s).
- CodeBLEU syntax match uses interned subtree ids instead of serialised strings: linear in the size of the
  program rather than quadratic.

## [0.5.1] - 2026-10-10

### Added
- `evalsuite benchmark --suite llmsys` (and `run_benchmarks(suite="llmsys")`): v0.5 LLM-systems workloads
  against radon, the codebleu package, scikit-learn, jsonschema, SciPy and NumPy; included in `--suite all`
  (27 cases, 81 measurements, all agreeing with their reference).

### Changed
- LLM-systems metrics accept NumPy arrays without converting them to Python lists first (much faster on
  large inputs); `confidence_accuracy_correlation` computes Spearman from the ranks it already has.

## [0.5.0] - 2026-10-10

LLM systems (the v0.5.0 roadmap: 8 areas, 85 items) and SPICE. No existing API changed.

### Added
- `es.llmsys` (all functions also at top level): 77 new registered metrics. Items on the roadmap that an
  existing function already covers (ECE, MCE, Brier, log loss, abstention, pass@k, BLEU/chrF/COMET,
  ROUGE-L/BERTScore, tool-call precision/recall, Recall@k/MRR) are documented as such.
  - Safety: harmful-response / unsafe-compliance, refusal and appropriate refusal, over-refusal, jailbreak /
    prompt-injection attack success, red-team success@k, expected maximum toxicity and toxicity probability
    (RealToxicityPrompts) with per-attribute scores, CrowS-Pairs stereotype preference, WEAT effect size with
    a permutation test, PII leakage (built-in e-mail, phone, Luhn-checked card, IPv4, SSN detectors plus
    protected strings; `detect_pii`), memorization exposure (Carlini et al.), policy violations.
  - Robustness: adversarial and typo-noise robustness (`add_typos`), OOD accuracy, distribution-shift drop
    with a Welch interval (= SciPy), paraphrase consistency, counterfactual invariance violations, response
    stability, contradiction rate, failure / timeout / error rate, recovery rate, prompt sensitivity, accuracy
    slope against context length.
  - Uncertainty: adaptive calibration error, risk-coverage curve, AURC and E-AURC, selective risk, risk at
    coverage, coverage at risk, confidence AUROC (= scikit-learn).
  - Agents: task completion and progress, invalid tool calls checked against each tool's JSON Schema,
    execution failures, tool-use efficiency, steps per task, plan adherence (LCS), joint goal and slot
    accuracy, tool-failure recovery, repeated-call and loop rates, human intervention, cost per success.
  - Multilingual: language-ID accuracy and macro-F1, bitext mining (cosine or margin), language parity,
    code-switching robustness, z-normalised direct assessment, cultural appropriateness, language consistency,
    cross-lingual consistency.
  - Code: unit-test pass rate, syntax validity (parsed, never run), static-analysis violations per KLOC,
    execution success, test coverage, patch acceptance and bug reproduction, SWE-bench resolved rate,
    security findings by severity, EffiBench NET / NMU, CodeBLEU (n-gram terms identical to the `codebleu`
    package; syntax and data-flow terms from Python's `ast`), cyclomatic complexity and maintainability index
    (identical to radon on every module of EvalSuite and radon).
  - Long context: accuracy by length with effective length (RULER), needle-in-a-haystack grid, accuracy by
    position, lost-in-the-middle gap, context utilization, summary coverage and factual consistency,
    compression ratio, citation coverage, cross-document consistency.
  - Efficiency: TTFT, TPOT, latency percentiles, throughput, token usage, cost (with cached-token pricing),
    resource utilization, energy per request, requests per second, availability and error budget.
- `es.spice`: SPICE from scene-graph tuples or any `parser`, with exact, WordNet or custom synonym matching
  and object / attribute / relation F-scores. Runs without Java.
- Benchmarks: `suite="metrics"` covers all 225 metrics and statistics functions; new "LLM systems" group.

## [0.4.1] - 2026-10-09

Benchmarks for every metric. No API changed.

### Added
- `run_benchmarks(suite="metrics")` and `evalsuite benchmark --suite metrics`: one row for each of the 147
  registered metrics and statistics functions, compared with a reference library (scikit-learn, SciPy,
  statsmodels, pycocotools, sacreBLEU, rouge-score, NLTK, pycocoevalcap, ranx, krippendorff, choix, POT,
  jsonschema) where one exists, otherwise with an independent NumPy / standard-library implementation of
  the textbook formula; learned, judge-dependent and randomised procedures are timed alone. All 128
  comparisons agree to 1e-9. A test keeps the suite in step with the registry.
- `scripts/bench_metrics_report.py` renders the per-metric table for BENCHMARKS.md and the website.

### Changed
- `net_benefit` at a single threshold computes in one pass instead of sorting (about 3× faster).
- `perplexity` and `cross_entropy` validate the pooled tokens once instead of each sequence separately
  (5–9× faster for many short sequences); errors still name the offending sequence.

## [0.4.0] - 2026-10-09

LLM evaluation (the v0.4.0 roadmap). No existing API changed.

### Added
- `es.text` and top-level functions for language-model evaluation; 69 new registered metrics, each checked
  against a reference implementation where one exists:
  - Text generation: BLEU, sentence BLEU, chrF/chrF++, TER (sacreBLEU), ROUGE-1/2/L/Lsum (rouge-score),
    METEOR (NLTK, synonyms pluggable), CIDEr-D (pycocoevalcap), perplexity, cross-entropy, Distinct-n,
    Self-BLEU, MAUVE (mauve-text divergence frontier, NumPy k-means).
  - Semantic: BERTScore, embedding cosine / Euclidean / Manhattan, MoverScore (exact transport, = POT), and
    `model_score` to run COMET, BLEURT, BARTScore, AlignScore or any scorer with intervals and comparison.
  - Factuality: faithfulness, hallucination (unsupported-claim) rate, groundedness, citation precision and
    recall (ALCE), claim-verification accuracy, knowledge consistency, answer correctness and relevance
    (RAGAS), abstention accuracy, exact match and token F1 (SQuAD).
  - LLM-as-a-judge: win rate, Bradley–Terry (= choix), Elo, Krippendorff's alpha (= krippendorff), Fleiss'
    kappa (= statsmodels), judge–human agreement, position consistency, verbosity and self-preference bias,
    rubric scores.
  - Reasoning: pass@k, majority-vote accuracy, answer extraction and benchmark accuracy (GSM8K, MATH,
    multiple choice).
  - RAG: Precision/Recall/Hit rate@k, MRR, MAP, NDCG (= ranx), context precision, recall and relevance,
    latency summary, task success rate, failure attribution.
  - Structured output: JSON validity, JSON Schema compliance (built-in validator, agrees with jsonschema),
    XML validity, required-field accuracy, tool selection / argument accuracy and tool-call F1, API-call
    success, instruction compliance, constraint satisfaction, format compliance, multi-turn retention,
    extra-content rate.
- `es.text_report` and `evalsuite text` (CLI) for the reference-based text metrics in one report.
- `bootstrap_ci`, `compare` and the paired tests resample whole items (texts, reference lists, rankings) for
  these metrics, so corpus metrics such as BLEU get proper intervals.
- Benchmark suite `llm` against sacreBLEU, rouge-score, NLTK, ranx, krippendorff and jsonschema.
- `llm` extra (`nltk`) for METEOR's default Porter stemmer.
- Plots: `es.plot.ratings` (Bradley–Terry or Elo leaderboard with bootstrap intervals), `es.plot.win_matrix`
  (pairwise win rates) and `es.plot.text_scores` (per-example score distributions by system).
- Benchmark overall summary counts cases without their size suffixes.

### Not included
- SPICE (needs a Java scene-graph parser); BLEURT, COMET, BARTScore and AlignScore run through
  `es.model_score` with your model rather than being bundled.

## [0.3.1] - 2026-10-09

Quality-assurance release: release pipeline, packaging checks, edge-case fixes and an overall benchmark.
No API was removed or renamed; code written for 0.3.0 runs unchanged.

### Fixed
- Labels that mix numbers and strings (`[0, "a"]`, or string `y_true` with integer `y_pred`) now raise
  `InputValidationError` instead of being silently compared as text.
- `rmse` no longer overflows to `inf` when errors exceed ~1e154; the value is computed with rescaling.
- `bootstrap_ci`, `paired_bootstrap_test` and `compare` raise `StatisticalTestError` for fewer than two
  observations instead of returning a zero-width interval.
- A 2-D mask given as a nested Python list (`[[1, 1], [0, 0]]`) is read as one image, as a NumPy array is.
- Release workflow: the GitHub release step no longer fails when the release already exists (for example
  when the tag was created from the GitHub UI); it attaches the files instead.

### Added
- Benchmarks: an overall summary first (per suite and in total: cases, agreement with the reference,
  how many are faster, geometric-mean and range of speed-ups) and rows in alphabetical order;
  `BenchmarkResult.overall()`, `overall_markdown()` and `sorted_rows()`; `overall` in the JSON export.
- Release workflow: refuses a tag that does not match the package version or a version already on PyPI,
  checks the built files, uses `--verify-tag`, and can be run by hand to attach a version's PyPI files
  (hash-verified) to its GitHub release.
- CI: wheel and sdist installed in clean environments with `pip check` and a smoke test that must give the
  same results for both, a distribution check (metadata, forbidden files, credentials), a minimum-dependency
  job (NumPy 1.22, SciPy 1.8, pandas 1.4), README examples run as tests, and repository hygiene tests
  (no credentials, no unsafe calls, least-privilege workflows, single version source).
- QA tests against scikit-learn and statsmodels for the edge cases in the release checklist.
- `SECURITY.md` with the reporting process.

## [0.3.0] - 2026-10-09

Computer vision (the v0.3.0 roadmap), with comparison, plots, reporting and benchmarks extended to it.

### Added
- Segmentation: `dice`, `iou`, `miou`, `pixel_accuracy`, `mean_pixel_accuracy`, `boundary_iou`,
  `hausdorff_distance` (HD and HD95, anisotropic `spacing`), `average_surface_distance`,
  `segmentation_confusion`, `per_image_scores` and `segmentation_report`. 2-D and 3-D masks, lists of
  differently sized masks, `ignore_index`, `aggregate="dataset"|"image"`, undefined classes reported as NaN.
- Object detection: `box_iou`, `detection_report` (the 12 COCO numbers plus AP per class),
  `mean_average_precision`, `average_precision_detection` (COCO or VOC interpolation), `detection_pr_curve`,
  `from_coco`. Matches pycocotools exactly, including crowd regions, area ranges and detection limits.
- Comparison for vision: `compare`, `bootstrap_ci` and `paired_bootstrap_test` resample images for
  segmentation masks and per-image detections.
- Plots: `es.plot.segmentation` (prediction fill vs truth outline), `es.plot.per_class` (any per-class
  result, a segmentation report or a detection report), `es.plot.detection_pr`.
- CLI: `evalsuite segmentation` (.npy/.npz or a folder of mask images) and `evalsuite detection` (COCO JSON).
- Benchmarks: `--suite vision` against scikit-learn, SciPy and pycocotools.
- `vision` extra (Pillow, for reading mask image folders); `CITATION.cff`.

### Changed
- `evaluate()` points segmentation masks and detection annotations to the right functions instead of
  failing with a shape error.
- CI and release workflows use the Node 24 versions of the GitHub actions.

### Fixed
- Calibration slope and intercept no longer emit an overflow warning before reporting a perfectly
  separated outcome.
- The CLI no longer fails on consoles that cannot print Unicode (Windows code pages).

## [0.2.1] - 2026-10-09

### Added
- Benchmarks for the v0.2.0 functions (`evalsuite benchmark --suite clinical`): diagnostic metrics and
  report, calibration slope and intercept, decision curves, t-test, Mann–Whitney, Cramér's V and Hochberg,
  each against its reference (scikit-learn, statsmodels, SciPy or the textbook NumPy loop). Benchmark rows
  now name their reference library.

### Changed
- Faster label handling: integer class labels are found with one marking pass instead of a sort
  (8 label metrics at 1M samples: 34× faster than scikit-learn, up from 10×; macro F1 5.8×, up from 1.6×).
- Decision curves sort the risks once and use cumulative sums (O((n + k) log n), no n × k matrix).

## [0.2.0] - 2026-10-08

Clinical and statistical evaluation (the v0.2.0 roadmap).

### Added
- Clinical: `sensitivity`, `ppv`, `lr_positive`, `lr_negative`, `diagnostic_odds_ratio`, `youden_j`,
  `net_benefit`; `diagnostic_report` with confidence intervals for every measure (Wilson or Clopper–Pearson
  for proportions, log method for likelihood ratios, Woolf for the DOR, Wald for Youden's J);
  `decision_curve` (net benefit, treat all, treat none, useful threshold range).
- Calibration: `maximum_calibration_error`, `calibration_slope`, `calibration_intercept`,
  `hosmer_lemeshow`, `calibration_report`.
- Statistical tests: `t_test` (Welch/Student), `paired_t_test`, `mann_whitney_test`, `wilcoxon_test`,
  `kruskal_wallis_test`, `friedman_test`, `shapiro_wilk_test`, `chi_square_test`, `fisher_exact_test`, each
  with an effect size; `cramers_v` (optional bias correction); Hochberg correction in `adjust_pvalues` and
  `compare`.
- `es.plot.decision_curve`; CLI commands `evalsuite diagnostic`, `evalsuite calibration` and
  `evalsuite plot decision`.
- Validated against statsmodels (GLM, Table2x2, proportion_confint, multipletests, CompareMeans) and SciPy.

## [0.1.2] - 2026-10-08

### Changed
- LICENSE: copyright held by Manoj Kumar C S and Nikhil D Bharadwaj.

## [0.1.1] - 2026-10-08

### Changed
- Credits: Manoj Kumar C S and Nikhil D Bharadwaj listed as authors and maintainers (README and package metadata).

## [0.1.0] - 2026-10-08

First stable release. Everything from the 0.1.0 roadmap: classification and regression metrics, result
system, input validation, metric registry, model comparison with confidence intervals and paired tests,
plots, HTML/CSV/LaTeX/Markdown reports, classification report, command-line tool and benchmarks.

### Changed
- Development status: Production/Stable.
- README (PyPI description) now includes the benchmark table against scikit-learn.

## [0.1.0b2]

### Changed
- `es.plot.calibration`: the legend now sits below the axes by default so it no longer covers the
  curves; new `legend_loc` argument (`"below"` or any matplotlib location).

## [0.1.0b1]

Feature-complete for 0.1.0.

### Added

- Plots (`es.plot`, optional `[plot]` extra): ROC, precision-recall, calibration, confusion matrix, residuals /
  predicted-vs-true, and model-comparison forest plots. Values come from EvalSuite's metrics; several models
  get distinct colours and line styles. Importing `evalsuite` never imports matplotlib.
- Reporting: `classification_report` (per-class precision, recall, F1, specificity, support; accuracy, micro,
  macro and weighted averages; matches scikit-learn); `to_html()` and `to_csv()` on every result; `save(path)`
  choosing the format from the extension (.json .csv .md .tex .html .txt).
- `calibration_curve` and `expected_calibration_error` (matches scikit-learn's calibration curve).
- Command line: `evalsuite evaluate | report | compare | plot | metrics | info | benchmark`, and
  `python -m evalsuite`. Reads CSV, TSV, Parquet and JSON; clear one-line errors with exit code 2.
- Benchmarks (`evalsuite.benchmarks.run_benchmarks`, `evalsuite benchmark`): time and peak memory against
  scikit-learn, with a check that both libraries return the same numbers. Results in `BENCHMARKS.md`.

### Changed

- Regression `evaluate()` validates inputs once for all metrics and uses a faster unweighted mean (2× faster
  on large arrays).

## [0.1.0a2]

### Added

- Model comparison (`es.compare`): per-model confidence intervals from paired bootstrap resamples, pairwise
  tests (McNemar for accuracy, DeLong for binary ROC AUC, paired bootstrap otherwise), multiple-comparison
  correction, best model per metric, and summary/pandas/Markdown/LaTeX/JSON export.
- Confidence intervals: `bootstrap_ci` (percentile, basic, BCa; stratified and reproducible),
  `proportion_ci` and `accuracy_ci` (Wilson, Clopper-Pearson, normal), `roc_auc_ci` (DeLong).
- Paired tests: `mcnemar_test`, `delong_test`, `paired_bootstrap_test`.
- Effect sizes: `cohens_d` (independent and paired), `hedges_g`, `cliffs_delta`; `adjust_pvalues`
  (Holm, Bonferroni, Benjamini-Hochberg, Benjamini-Yekutieli).
- Reference tests against statsmodels and SciPy, brute-force DeLong checks and coverage simulations.
- Python 3.14 support and CI.

### Fixed

- 0.1.0a1 installed an `evalsuite` console command although the CLI is not implemented yet, so the command
  failed with `ModuleNotFoundError`. The entry point is removed until the CLI ships.

## [0.1.0a1]

First alpha, published to reserve the name and test the release pipeline. Distribution name
`evalsuite-python` (`pip install --pre evalsuite-python`), imported as `evalsuite`.

### Added

- Core: input validation with actionable errors, exception hierarchy, immutable result objects with
  JSON/pandas/Markdown/LaTeX export, metric registry (`list_metrics`, `metric_info`), and an evaluation
  context that computes the confusion matrix once per evaluation.
- Classification metrics for binary, multiclass and multilabel targets with all averaging modes and
  sample weights.
- Regression metrics for single- and multi-output targets with sample weights.
- `evaluate()` high-level API with task inference and default metric sets.
- Reference tests against scikit-learn and property-based tests; Python 3.9 to 3.13.
