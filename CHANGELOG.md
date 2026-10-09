# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (v0.4.0, LLM evaluation, in development)
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
