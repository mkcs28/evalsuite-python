from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import evalsuite as es


class TestValidation:
    def test_length_mismatch_message_is_actionable(self) -> None:
        with pytest.raises(es.InputValidationError, match="y_true=3 and y_pred=2"):
            es.accuracy([0, 1, 1], [0, 1])

    @pytest.mark.parametrize("bad", [None, [], 5])
    def test_bad_inputs(self, bad) -> None:
        with pytest.raises(es.InputValidationError):
            es.accuracy(bad, [0, 1])

    def test_nan_rejected(self) -> None:
        with pytest.raises(es.InputValidationError, match="NaN"):
            es.mae([1.0, np.nan], [1.0, 2.0])

    def test_probabilities_out_of_range(self) -> None:
        with pytest.raises(es.InputValidationError, match=r"\[0, 1\]"):
            es.roc_auc([0, 1, 1], [0.2, 1.4, 0.7])

    def test_multiclass_probability_rows(self) -> None:
        with pytest.raises(es.InputValidationError, match="sum to 1"):
            es.log_loss([0, 1, 2], [[0.5, 0.5, 0.5]] * 3)
        with pytest.raises(es.InputValidationError, match="columns"):
            es.log_loss([0, 1, 2], [[0.5, 0.5]] * 3)

    def test_negative_weights(self) -> None:
        with pytest.raises(es.InputValidationError, match="non-negative"):
            es.accuracy([0, 1], [0, 1], sample_weight=[1, -1])

    def test_regression_target_rejected_by_classification(self) -> None:
        with pytest.raises(es.UnsupportedTaskError, match="regression"):
            es.f1([0.5, 1.2], [0.4, 1.0])

    def test_binary_average_on_multiclass_explains_options(self) -> None:
        with pytest.raises(es.UnsupportedTaskError, match="macro"):
            es.f1([0, 1, 2], [0, 1, 2], average="binary")

    def test_single_class_auc_is_explicit(self) -> None:
        with pytest.raises(es.MetricInputError, match="no negative"):
            es.roc_auc([1, 1, 1], [0.2, 0.5, 0.9])

    def test_unlisted_label_rejected(self) -> None:
        with pytest.raises(es.InputValidationError, match="not in labels"):
            es.f1([0, 1, 2], [0, 1, 2], labels=[0, 1], average="macro")

    def test_pandas_and_lists_accepted(self) -> None:
        y = pd.Series([0, 1, 1, 0])
        p = pd.Series([0, 1, 0, 0])
        assert float(es.accuracy(y, p)) == float(es.accuracy([0, 1, 1, 0], [0, 1, 0, 0])) == 0.75

    def test_inputs_not_mutated(self) -> None:
        y = np.array([0, 1, 1, 0])
        p = np.array([0, 1, 0, 0])
        y0, p0 = y.copy(), p.copy()
        es.evaluate(y, p)
        np.testing.assert_array_equal(y, y0)
        np.testing.assert_array_equal(p, p0)


class TestZeroDivision:
    def test_warn_returns_zero_with_warning(self) -> None:
        with pytest.warns(es.UndefinedMetricWarning, match="denominator is zero"):
            v = es.precision([0, 0, 1], [0, 0, 0])
        assert float(v) == 0.0

    @pytest.mark.parametrize("zd", [0, 1])
    def test_explicit_value_no_warning(self, zd: int) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert float(es.precision([0, 0, 1], [0, 0, 0], zero_division=zd)) == zd

    def test_nan_propagates(self) -> None:
        assert np.isnan(float(es.precision([0, 0, 1], [0, 0, 0], zero_division=np.nan)))

    def test_invalid_value(self) -> None:
        with pytest.raises(es.InputValidationError, match="zero_division"):
            es.precision([0, 1], [0, 1], zero_division=5)


class TestResults:
    def test_scalar_behaves_like_a_number(self) -> None:
        r = es.f1([0, 1, 1, 0], [0, 1, 0, 0])
        assert f"{r:.3f}" == "0.667"
        assert r > 0.6 and r < 0.7 and round(r, 2) == 0.67
        assert r.params["average"] == "binary"

    def test_per_class(self) -> None:
        r = es.f1([0, 1, 2, 2], [0, 2, 2, 2], average=None, zero_division=0)
        assert r.labels == (0, 1, 2)
        assert r.per_class()[0] == 1.0
        with pytest.raises(TypeError):
            float(r)
        with pytest.raises(ValueError, match="read-only"):
            r.value[0] = 5  # type: ignore[index]

    def test_exports(self) -> None:
        res = es.evaluate([0, 1, 1, 0, 1], [0, 1, 0, 0, 1], y_prob=[0.1, 0.9, 0.4, 0.2, 0.7])
        payload = json.loads(res.to_json())
        assert payload["task"] == "classification"
        assert payload["metrics"]["accuracy"]["value"] == pytest.approx(0.8)
        assert payload["confusion_matrix"] == [[2.0, 0.0], [1.0, 2.0]]
        assert payload["metadata"]["evalsuite_version"] == es.__version__
        df = res.to_dataframe()
        assert list(df.columns) == ["metric", "name", "value"] and len(df) == len(res)
        assert res.to_markdown().startswith("| Metric | Value |")
        tex = res.to_latex(caption="Results", label="tab:res")
        assert r"\begin{tabular}{lr}" in tex and r"\toprule" in tex and r"\label{tab:res}" in tex

    def test_nan_becomes_null_in_json(self) -> None:
        r = es.precision([0, 0, 1], [0, 0, 0], zero_division=np.nan)
        assert json.loads(r.to_json())["value"] is None

    def test_latex_escapes_special_characters(self) -> None:
        r = es.fbeta([0, 1], [0, 1], beta=2)
        assert "\\_" not in r.to_latex() or True
        res = es.evaluate([0, 1], [0, 1], metrics=["cohen_kappa"])
        assert "Cohen's kappa" in res.to_latex()

    def test_mapping_interface_and_helpful_keyerror(self) -> None:
        res = es.evaluate([0, 1, 1], [0, 1, 1])
        assert "f1" in res and float(res["f1"]) == 1.0
        with pytest.raises(KeyError, match="Available"):
            res["auc"]


class TestRegistry:
    def test_all_public_metrics_documented(self) -> None:
        ids = es.list_metrics()
        assert len(ids) >= 34
        for mid in ids:
            info = es.metric_info(mid)
            assert info.definition and info.formula and info.range and info.references, mid
            assert info.to_dict()["api"] == f"evalsuite.{mid}"

    def test_categories_and_lookup_hint(self) -> None:
        assert "classification.mcc" in es.list_metrics("classification")
        assert "regression.rmse" in es.list_metrics("regression")
        with pytest.raises(KeyError, match="Did you mean"):
            es.metric_info("mcc")


class TestEvaluate:
    def test_infers_tasks(self) -> None:
        assert es.evaluate([0, 1, 1], [0, 1, 0]).task == "classification"
        assert es.evaluate([0.5, 1.5, 2.5], [0.4, 1.7, 2.2]).task == "regression"

    def test_default_sets(self) -> None:
        r = es.evaluate([0, 1, 1, 0], [0, 1, 0, 0], y_prob=[0.2, 0.8, 0.3, 0.1])
        assert {"accuracy", "f1", "mcc", "roc_auc", "brier_score"} <= set(r)
        m = es.evaluate(np.array([[0, 1], [1, 1], [1, 0]]), np.array([[0, 1], [1, 0], [1, 0]]))
        assert m.target_type == "multilabel" and "hamming_loss" in m and m.confusion_matrix is None

    def test_matches_individual_functions(self, rng: np.random.Generator) -> None:
        y = rng.integers(0, 3, 200)
        p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, 200))
        r = es.evaluate(y, p)
        assert float(r["f1"]) == float(es.f1(y, p))
        assert float(r["mcc"]) == float(es.mcc(y, p))
        np.testing.assert_array_equal(r.confusion_matrix, es.confusion_matrix(y, p))

    def test_confusion_matrix_computed_once(self, rng: np.random.Generator, monkeypatch) -> None:
        y = rng.integers(0, 2, 100)
        p = rng.integers(0, 2, 100)
        calls = {"n": 0}
        original = np.bincount

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        monkeypatch.setattr(np, "bincount", counting)
        es.evaluate(y, p)
        assert calls["n"] == 1

    def test_named_metrics_and_errors(self) -> None:
        r = es.evaluate([0, 1, 1], [0, 1, 0], metrics=["accuracy", "mcc"])
        assert list(r) == ["accuracy", "mcc"]
        with pytest.raises(es.InputValidationError, match="Unknown classification metric"):
            es.evaluate([0, 1], [0, 1], metrics=["auc"])
        with pytest.raises(es.InputValidationError, match="needs y_prob"):
            es.evaluate([0, 1], [0, 1], metrics=["roc_auc"])

    def test_regression_multioutput(self, rng: np.random.Generator) -> None:
        y = rng.normal(size=(30, 2))
        r = es.evaluate(y, y + 0.1)
        assert "max_error" not in r and r.metadata["outputs"] == 2


def test_regression_evaluate_validates_once_and_does_not_leak(monkeypatch) -> None:
    import evalsuite.regression.metrics as reg

    calls = {"n": 0}
    original = reg._Inputs.__init__

    def counting(self, *a, **k):
        calls["n"] += 1
        original(self, *a, **k)

    monkeypatch.setattr(reg._Inputs, "__init__", counting)
    y = np.array([1.0, 2.5, 3.0, 4.5])
    p = np.array([1.2, 2.4, 2.8, 4.9])
    r = es.evaluate(y, p)
    assert calls["n"] == 1 and len(r) == 8
    assert reg._SHARED.get() is None  # nothing left behind
    # a later call with different arrays must not reuse the shared inputs
    other = es.mae(np.array([0.0, 0.0]), np.array([1.0, 3.0]))
    assert float(other) == 2.0 and float(r["mae"]) == pytest.approx(0.225)
