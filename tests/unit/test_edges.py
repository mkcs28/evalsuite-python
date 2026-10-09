"""Error paths and export formats: every user-facing message and branch."""

from __future__ import annotations

import json

import numpy as np
import pytest

import evalsuite as es
from evalsuite.core.context import ClassificationContext
from evalsuite.core.exceptions import OptionalDependencyError
from evalsuite.core.validation import to_numpy, validate_sample_weight


class TestValidationEdges:
    def test_shapes(self) -> None:
        with pytest.raises(es.InputValidationError, match="scalar"):
            to_numpy(3, "y")
        with pytest.raises(es.InputValidationError, match="shape"):
            to_numpy(np.zeros((2, 2, 2)), "y", allow_2d=True)
        with pytest.raises(es.InputValidationError, match="1-D"):
            to_numpy(np.zeros((3, 2)), "y")
        assert to_numpy(np.zeros((3, 1)), "y").shape == (3,)

    def test_weights(self) -> None:
        with pytest.raises(es.InputValidationError, match="one weight per observation"):
            validate_sample_weight([1, 2], 3)
        with pytest.raises(es.InputValidationError, match="sum to zero"):
            validate_sample_weight([0, 0], 2)

    def test_labels(self) -> None:
        with pytest.raises(es.InputValidationError, match="duplicates"):
            es.f1([0, 1], [0, 1], labels=[0, 0, 1], average="macro")
        with pytest.raises(es.InputValidationError, match="comparable"):
            es.f1(np.array([1, "a"], dtype=object), np.array(["a", 1], dtype=object), average="macro")

    def test_bad_2d_target(self) -> None:
        with pytest.raises(es.InputValidationError, match="indicator"):
            es.f1(np.array([[0, 2], [1, 1]]), np.array([[0, 1], [1, 1]]))


class TestContextEdges:
    def test_dimension_and_multilabel_mismatch(self) -> None:
        with pytest.raises(es.InputValidationError, match="dimension"):
            ClassificationContext(np.array([[0, 1], [1, 0]]), np.array([0, 1]))
        with pytest.raises(es.InputValidationError, match="label columns"):
            ClassificationContext(np.array([[0, 1], [1, 0]]), np.array([[0, 1, 1], [1, 0, 0]]))
        with pytest.raises(es.InputValidationError, match="indicator"):
            ClassificationContext(np.array([[0, 1], [1, 0]]), np.array([[0, 2], [1, 0]]))

    def test_unsorted_explicit_labels_and_objects(self) -> None:
        r = es.recall(
            ["b", "a", "c", "a"], ["b", "a", "a", "a"], labels=["c", "b", "a"], average=None, zero_division=0
        )
        assert r.per_class() == {"c": 0.0, "b": 1.0, "a": 1.0}

    def test_probability_requirements(self) -> None:
        with pytest.raises(es.InputValidationError, match="needs predicted probabilities"):
            _ = ClassificationContext([0, 1], [0, 1]).y_prob
        with pytest.raises(es.InputValidationError, match="needs y_pred"):
            _ = ClassificationContext([0, 1], None).pred_idx
        assert float(es.roc_auc([0, 1, 1, 0], [[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]])) == 1.0
        with pytest.raises(es.InputValidationError, match="two columns"):
            es.roc_auc([0, 1, 0], [[0.2, 0.3, 0.5]] * 3)
        with pytest.raises(es.InputValidationError, match="one column per class"):
            es.roc_auc([0, 1, 2], [0.2, 0.5, 0.3])
        with pytest.raises(es.InputValidationError, match="one probability column per label"):
            es.roc_auc(np.array([[0, 1], [1, 0]]), np.array([[0.2, 0.3, 0.1], [0.3, 0.1, 0.2]]))
        with pytest.raises(es.UnsupportedTaskError, match="multilabel"):
            _ = ClassificationContext(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [1, 0]])).confusion_matrix

    def test_unsupported_multilabel_metrics(self) -> None:
        Y = np.array([[0, 1], [1, 0], [1, 1]])
        for fn in (es.mcc, es.balanced_accuracy, es.cohen_kappa):
            with pytest.raises(es.UnsupportedTaskError):
                fn(Y, Y)
        with pytest.raises(es.UnsupportedTaskError):
            es.log_loss(Y, Y * 0.5)
        with pytest.raises(es.UnsupportedTaskError):
            es.brier_score(Y, Y * 0.5)


class TestAveragingEdges:
    def test_invalid_average_and_samples(self) -> None:
        with pytest.raises(es.UnsupportedTaskError, match="not supported"):
            es.f1([0, 1], [0, 1], average="mean")
        with pytest.raises(es.UnsupportedTaskError, match="multilabel"):
            es.f1([0, 1, 2], [0, 1, 2], average="samples")

    def test_pos_label_absent(self) -> None:
        with pytest.raises(es.InputValidationError, match="not one of the labels"):
            es.f1([0, 1], [0, 1], pos_label=7)
        # all-negative data: the positive class never occurs, precision/recall are undefined
        assert float(es.recall([0, 0, 0], [0, 0, 0], zero_division=0)) == 0.0

    def test_unsupported_options(self) -> None:
        y = [0, 1, 2, 0, 1, 2]
        prob = np.full((6, 3), 1 / 3)
        with pytest.raises(es.UnsupportedTaskError, match="macro"):
            es.roc_auc(y, prob, multi_class="ovo", average="weighted")
        with pytest.raises(es.UnsupportedTaskError, match="ovr"):
            es.roc_auc(y, prob, multi_class="bad")  # type: ignore[arg-type]
        with pytest.raises(es.UnsupportedTaskError, match="micro"):
            es.roc_auc(y, prob, average="micro")
        with pytest.raises(es.InputValidationError, match="k must be"):
            es.top_k_accuracy(y, prob, k=4)
        with pytest.raises(es.UnsupportedTaskError):
            es.top_k_accuracy([0, 1], [0.2, 0.8])
        with pytest.raises(es.InputValidationError, match="beta"):
            es.fbeta([0, 1], [0, 1], beta=0)
        with pytest.raises(es.InputValidationError, match="weights"):
            es.cohen_kappa([0, 1], [0, 1], weights="cubic")  # type: ignore[arg-type]
        with pytest.raises(es.InputValidationError, match="normalize"):
            es.confusion_matrix([0, 1], [0, 1], normalize="rows")  # type: ignore[arg-type]
        with pytest.raises(es.UnsupportedTaskError, match="binary"):
            es.roc_curve([0, 1, 2], prob[:3])
        with pytest.raises(es.UnsupportedTaskError, match="binary"):
            es.pr_curve([0, 1, 2], prob[:3])
        with pytest.raises(es.MetricInputError, match="no positive"):
            es.pr_curve([0, 0, 0], [0.1, 0.2, 0.3])
        with pytest.raises(es.MetricInputError, match="no positive"):
            es.average_precision([0, 0, 0], [0.1, 0.2, 0.3])

    def test_mcc_and_kappa_undefined(self) -> None:
        with pytest.warns(es.UndefinedMetricWarning):
            assert float(es.mcc([1, 1, 1], [1, 1, 1])) == 0.0
        assert np.isnan(float(es.cohen_kappa([1, 1], [1, 1], zero_division=np.nan)))

    def test_balanced_accuracy_adjusted_single_class(self) -> None:
        assert float(es.balanced_accuracy([1, 1, 1], [1, 1, 1], adjusted=True)) == 0.0


class TestRegressionEdges:
    def test_inputs(self) -> None:
        with pytest.raises(es.InputValidationError, match="numeric"):
            es.mae(["a", "b"], ["a", "b"])
        with pytest.raises(es.InputValidationError, match="shape"):
            es.mae(np.zeros((3, 2)), np.zeros((3, 3)))
        with pytest.raises(es.InputValidationError, match="multioutput"):
            es.mae(np.zeros((3, 2)), np.zeros((3, 2)), multioutput="mean")
        with pytest.raises(es.InputValidationError, match="weights"):
            es.mae(np.zeros((3, 2)), np.zeros((3, 2)), multioutput=[1.0])
        with pytest.raises(es.InputValidationError, match="single-output"):
            es.max_error(np.zeros((3, 2)), np.zeros((3, 2)))
        with pytest.raises(es.InputValidationError, match="single-output"):
            es.adjusted_r2(np.zeros((3, 2)), np.zeros((3, 2)), n_features=1)
        with pytest.raises(es.InputValidationError, match="n_features"):
            es.adjusted_r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_features=-1)
        with pytest.raises(es.InputValidationError, match="alpha"):
            es.quantile_loss([1.0], [1.0], alpha=1.5)
        with pytest.raises(es.InputValidationError, match="delta"):
            es.huber_loss([1.0], [1.0], delta=0)

    def test_explained_variance_constant(self) -> None:
        with pytest.warns(es.UndefinedMetricWarning):
            assert float(es.explained_variance([2.0, 2.0], [2.0, 2.0])) == 1.0

    def test_multioutput_raw_values_labels(self) -> None:
        r = es.mae(np.zeros((4, 2)), np.ones((4, 2)), multioutput="raw_values")
        assert r.labels == (0, 1) and r.params["multioutput"] == "raw_values"


class TestResultFormats:
    def test_per_class_exports(self) -> None:
        r = es.f1([0, 1, 2, 2], [0, 2, 2, 2], average=None, zero_division=0)
        assert "MetricResult(f1: {0: 1.0000" in repr(r)
        assert r.to_markdown().splitlines()[0] == "| Label | F1 |"
        assert r"\begin{tabular}" in r.to_latex()
        df = r.to_dataframe()
        assert list(df.columns) == ["label", "f1"] and len(df) == 3
        d = json.loads(r.to_json())
        assert d["labels"] == [0, 1, 2] and d["params"]["average"] is None

    def test_scalar_exports_and_errors(self) -> None:
        r = es.accuracy([0, 1], [0, 1])
        assert repr(r) == "MetricResult(accuracy=1.000000)"
        assert r.to_markdown().endswith("| Accuracy | 1.0000 |")
        assert r.to_dataframe().iloc[0]["value"] == 1.0
        assert f"{r}" == repr(r)
        with pytest.raises(TypeError, match="single value"):
            r.per_class()
        assert r == 1.0 and r != es.accuracy([0, 1], [0, 1]) and hash(r)

    def test_evaluation_summary_with_per_class_and_value(self) -> None:
        res = es.evaluate([0, 1, 2, 2], [0, 2, 2, 2], metrics=["accuracy", "f1"], average=None, zero_division=0)
        text = res.summary()
        assert "0=1.0000" in text and repr(res) == text
        assert set(res.value) == {"accuracy", "f1"}
        assert len(res.to_dataframe()) == 1  # per-class rows are excluded from the scalar table
        assert res.to_dict()["labels"] == [0, 1, 2]

    def test_special_characters_escaped_in_latex(self) -> None:
        from evalsuite.core.result import _latex_escape

        assert _latex_escape("a_b & 50% #1 $x$ {y} ~ ^") == (
            r"a\_b \& 50\% \#1 \$x\$ \{y\} \textasciitilde{} \textasciicircum{}"
        )

    def test_optional_dependency_message(self) -> None:
        err = OptionalDependencyError("matplotlib", "plot", "Plotting")
        assert 'pip install "evalsuite-python[plot]"' in str(err) and err.extra == "plot"


class TestEvaluateEdges:
    def test_task_and_input_errors(self) -> None:
        with pytest.raises(es.UnsupportedTaskError):
            es.evaluate([0, 1], [0, 1], task="segmentation")
        with pytest.raises(es.InputValidationError, match="needs y_pred"):
            es.evaluate([0.5, 1.5], task="regression")
        with pytest.raises(es.InputValidationError, match="y_pred, y_prob"):
            es.evaluate([0, 1], task="classification")
        with pytest.raises(es.InputValidationError, match="Unknown regression metric"):
            es.evaluate([0.5, 1.5], [0.4, 1.6], metrics=["auc"])
        with pytest.raises(es.InputValidationError, match="does not support sample_weight"):
            es.evaluate([0.5, 1.5], [0.4, 1.6], metrics=["max_error"], sample_weight=[1, 2])
        with pytest.raises(es.InputValidationError, match="needs y_pred"):
            es.evaluate([0, 1], y_prob=[0.2, 0.8], metrics=["accuracy"])

    def test_probability_only_evaluation(self) -> None:
        r = es.evaluate([0, 1, 1, 0], y_prob=[0.1, 0.8, 0.7, 0.3])
        assert set(r) == {"roc_auc", "average_precision", "log_loss", "brier_score"}
        assert r.confusion_matrix is None

    def test_multilabel_with_probabilities_and_hamming(self) -> None:
        Y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
        S = np.array([[0.2, 0.7], [0.8, 0.3], [0.6, 0.9], [0.1, 0.2]])
        r = es.evaluate(Y, (S > 0.5).astype(int), y_prob=S)
        assert {"roc_auc", "average_precision", "hamming_loss"} <= set(r)
        r2 = es.evaluate([0, 1, 2], [0, 2, 2], metrics=["hamming_loss", "npv", "jaccard"], zero_division=0)
        assert float(r2["hamming_loss"]) == pytest.approx(1 / 3)


def test_unique_labels_matches_numpy_for_every_integer_kind() -> None:
    from evalsuite.core.validation import unique_labels

    cases = [
        np.array([-128, 127, 0], dtype=np.int8),
        np.array([2**64 - 1, 2**64 - 3], dtype=np.uint64),
        np.array([5, 3, 5], dtype=np.uint8),
        np.array([10**15, -(10**15)]),
        np.array([], dtype=np.int64),
        np.array([[1, 0], [0, 1]]),
        np.array([1.5, 0.5]),
        np.array(["b", "a", "b"]),
    ]
    for a in cases:
        u = unique_labels(a)
        assert np.array_equal(u, np.unique(a)) and u.dtype == np.unique(a).dtype
