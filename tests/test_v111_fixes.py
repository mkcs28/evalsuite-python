"""Regression tests for bugs fixed in 1.1.1."""

import numpy as np
import pandas as pd
import pytest

import evalsuite as es
from evalsuite.core.exceptions import InputValidationError


def test_integer_truth_with_continuous_predictions_is_regression():
    r = es.evaluate([1.0, 2.0, 3.0], [1.0, 2.5, 2.0])
    assert r.task == "regression"
    assert r["mae"].value == pytest.approx(0.5)
    # integer predictions keep the classification reading
    assert es.evaluate([1, 2, 3], [1, 2, 2]).task == "classification"


def test_probabilities_passed_as_predictions_get_a_clear_error():
    with pytest.raises(InputValidationError, match="y_prob"):
        es.evaluate([0, 1, 1], [0.2, 0.9, 0.4])
    assert es.evaluate([0, 1, 1], [0.2, 0.9, 0.4], task="regression").task == "regression"


def test_single_column_inputs_are_vectors():
    col = np.array([[0], [1], [1]])
    assert es.evaluate(col, np.array([[0], [1], [0]]))["accuracy"].value == pytest.approx(2 / 3)
    assert es.evaluate(pd.DataFrame({"y": [0, 1, 1]}), [0, 1, 0])["accuracy"].value == pytest.approx(2 / 3)
    assert es.evaluate([[0], [1]], [[0], [1]])["accuracy"].value == 1.0
    # real multilabel matrices are untouched
    assert es.evaluate(np.array([[1, 0], [0, 1]]), np.array([[1, 0], [1, 1]])).target_type == "multilabel"
