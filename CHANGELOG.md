# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
