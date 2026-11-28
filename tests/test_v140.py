"""1.4.0: package-wide input validation and per-area lazy imports."""

import os
import subprocess
import sys

import numpy as np
import pytest

import evalsuite as es

E = es.InputValidationError
STR = np.array(["a", "b", "a", "b", "b", "a"])
BIN = np.array([0, 1, 1, 0, 1, 0])


@pytest.mark.parametrize(
    "call",
    [
        lambda: es.roc_auc(BIN, y_prob=STR),
        lambda: es.log_loss(BIN, y_prob=STR),
        lambda: es.t_test(STR, STR),
        lambda: es.mann_whitney_test(BIN, STR),
        lambda: es.cohens_d(STR, STR),
        lambda: es.adjust_pvalues(STR),
        lambda: es.kaplan_meier(STR, BIN),
        lambda: es.population_stability_index(STR, STR),
        lambda: es.recalibrate(BIN, STR),
        lambda: es.bayes_correlated_ttest(STR, STR),
        lambda: es.latency_summary(STR),
        lambda: es.krippendorff_alpha(np.array([["a", "b"], ["b", "a"]])),
        lambda: es.chi_square_test(np.array([["a", "b"], ["c", "d"]])),
    ],
)
def test_non_numeric_input_is_an_input_validation_error(call):
    with pytest.raises(E, match="must contain numbers"):
        call()


def test_wrong_argument_types_get_clear_errors():
    with pytest.raises(E, match="map model names"):
        es.compare(BIN, BIN)
    with pytest.raises(E, match="qrels must map"):
        es.ranking_report(BIN, BIN)
    with pytest.raises(E, match="single number"):
        es.sample_size_proportion(BIN, 0.05)
    with pytest.raises(E, match="name such as"):
        es.list_metrics(BIN)
    with pytest.raises(E, match="one string"):
        es.normalize_answer(BIN)
    with pytest.raises(E, match="one string"):
        es.extract_json(BIN)
    with pytest.raises(E, match="question string"):
        es.format_multiple_choice(BIN, BIN)


def test_infinite_labels_do_not_leak_numpy_warnings():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(es.EvalSuiteError):
            es.evaluate(np.array([0.0, 1.0, np.inf, 0.0]), np.array([0.1, 0.9, 0.8, 0.2]))


AREAS = [
    "text",
    "llmsys",
    "vision",
    "clinical",
    "survival",
    "forecasting",
    "drift",
    "judging",
    "rag_report",
    "fairness",
    "multilabel",
    "agreement",
    "bayes",
    "recalibration",
    "calibration",
    "html_report",
    "cache",
    "cli.main",
]


@pytest.mark.parametrize("area", AREAS)
def test_feature_areas_load_no_heavy_library(area):
    code = (
        f"import sys, evalsuite.{area}; "
        "print(','.join(m for m in ('scipy', 'pandas', 'matplotlib', 'sklearn', 'torch') if m in sys.modules))"
    )
    out = subprocess.run(  # noqa: S603 - fixed interpreter, test-only code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "EVALSUITE_NO_UPDATE_CHECK": "1"},
    )
    assert out.stdout.strip() == ""
