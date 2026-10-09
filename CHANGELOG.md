# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
  result), `es.plot.detection_pr`.
- CLI: `evalsuite segmentation` (.npy/.npz or a folder of mask images) and `evalsuite detection` (COCO JSON).
- Benchmarks: `--suite vision` against scikit-learn, SciPy and pycocotools.
- `vision` extra (Pillow, for reading mask image folders); `CITATION.cff`.

### Changed
- `evaluate()` points segmentation masks and detection annotations to the right functions instead of
  failing with a shape error.
- CI and release workflows use the Node 24 versions of the GitHub actions.

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
