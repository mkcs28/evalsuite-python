from __future__ import annotations

import json

import numpy as np
import pytest

import evalsuite as es


@pytest.fixture
def three_models():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 250)
    preds = {
        name: np.where(rng.random(250) < acc, y, 1 - y)
        for name, acc in (("strong", 0.9), ("medium", 0.8), ("weak", 0.65))
    }
    probs = {
        name: np.clip(y * w + rng.random(250) * (1 - w), 0, 1)
        for name, w in (("strong", 0.6), ("medium", 0.4), ("weak", 0.1))
    }
    return y, preds, probs


def test_compare_defaults_tests_and_reproducibility(three_models) -> None:
    y, preds, probs = three_models
    r = es.compare(y, preds, probabilities=probs, random_state=0, n_resamples=400)
    assert r.models == ("strong", "medium", "weak")
    assert r.metrics == ("accuracy", "f1", "mcc", "roc_auc", "brier_score")
    assert len(r.estimates) == 15 and len(r.tests) == 15  # 3 pairs x 5 metrics
    tests = {(t["metric"], t["model_a"], t["model_b"]): t for t in r.tests}
    assert tests[("accuracy", "strong", "weak")]["test"].startswith("mcnemar")
    assert tests[("roc_auc", "strong", "weak")]["test"] == "delong"
    assert tests[("f1", "strong", "weak")]["test"] == "paired-bootstrap"
    assert all(t["p_adjusted"] >= t["p_value"] - 1e-12 for t in r.tests)
    assert tests[("accuracy", "strong", "weak")]["significant"]
    assert r.best("accuracy") == "strong" and r.best("brier_score") == "strong"
    for row in r.estimates:
        assert row["low"] <= row["estimate"] <= row["high"] or row["metric"] == "brier_score"
    again = es.compare(y, preds, probabilities=probs, random_state=0, n_resamples=400)
    assert again.to_json() == r.to_json()


def test_compare_matches_individual_functions(three_models) -> None:
    y, preds, probs = three_models
    r = es.compare(y, preds, probabilities=probs, random_state=1, n_resamples=200)
    assert r.estimate("medium", "mcc")["estimate"] == float(es.mcc(y, preds["medium"]))
    t = next(
        t for t in r.tests if t["metric"] == "roc_auc" and t["model_a"] == "strong" and t["model_b"] == "weak"
    )
    assert t["p_value"] == es.delong_test(y, probs["strong"], probs["weak"]).p_value


def test_compare_baseline_and_exports(three_models) -> None:
    y, preds, _ = three_models
    r = es.compare(
        y,
        preds,
        baseline="weak",
        metrics=["accuracy", "f1"],
        random_state=0,
        n_resamples=200,
        metric_kwargs={"f1": {"average": "macro"}},
        correction="bh",
    )
    assert {(t["model_a"], t["model_b"]) for t in r.tests} == {("strong", "weak"), ("medium", "weak")}
    md = r.to_markdown()
    assert md.splitlines()[0] == "| Model | accuracy | f1 |" and "**" in md
    tex = r.to_latex(label="tab:cmp")
    assert r"\textbf{" in tex and r"\label{tab:cmp}" in tex and "95\\% CI" in tex
    assert len(r.to_dataframe()) == 6 and len(r.to_dataframe("tests")) == 4
    payload = json.loads(r.to_json())
    assert payload["settings"]["correction"] == "bh" and payload["settings"]["baseline"] == "weak"
    assert "significant at" in r.summary() and repr(r) == r.summary()


def test_compare_regression() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(10, 2, 200)
    r = es.compare(
        y, {"good": y + rng.normal(0, 0.5, 200), "bad": y + rng.normal(0, 2, 200)}, random_state=0, n_resamples=300
    )
    assert r.metrics == ("mae", "rmse", "r2")
    assert r.best("mae") == "good" and r.best("r2") == "good"
    assert all(t["test"] == "paired-bootstrap" and t["significant"] for t in r.tests)


def test_compare_errors(three_models) -> None:
    y, preds, _ = three_models
    with pytest.raises(es.InputValidationError, match="at least two models"):
        es.compare(y, {"a": preds["strong"]})
    with pytest.raises(es.InputValidationError, match="baseline"):
        es.compare(y, preds, baseline="nope")
    with pytest.raises(es.InputValidationError, match="needs probabilities"):
        es.compare(y, preds, metrics=["roc_auc"])
    with pytest.raises(es.InputValidationError, match="which"):
        es.compare(y, preds, n_resamples=100, random_state=0).to_dataframe("everything")
    with pytest.raises(KeyError):
        es.compare(y, preds, n_resamples=100, random_state=0).best("auc")


def test_result_objects() -> None:
    ci = es.proportion_ci(8, 10)
    low, high = ci
    assert (low, high) == (ci.low, ci.high) and float(ci) == 0.8 and ci.width == high - low
    assert "wilson" in repr(ci) and json.loads(ci.to_json())["method"] == "wilson"
    t = es.mcnemar_test([0, 1, 1, 0] * 10, [0, 1, 1, 0] * 10, [0, 1, 0, 0] * 10)
    assert t.significant(0.05) == (t.p_value < 0.05) and "TestResult(mcnemar" in repr(t)
    assert json.loads(t.to_json())["params"]["b"] == 10


@pytest.mark.parametrize(
    "call,match",
    [
        (lambda: es.bootstrap_ci("f1", [0, 1], [0, 1], level=1.5), "level"),
        (lambda: es.bootstrap_ci("f1", [0, 1], [0, 1], n_resamples=10), "n_resamples"),
        (lambda: es.bootstrap_ci("f1", [0, 1], [0, 1], method="student"), "method"),
        (lambda: es.bootstrap_ci("nonsense", [0, 1], [0, 1]), "Unknown metric"),
        (lambda: es.bootstrap_ci(42, [0, 1], [0, 1]), "metric function"),
        (lambda: es.bootstrap_ci("f1", [0, 1], [0, 1], y_prob=[0.1, 0.9]), "exactly one"),
        (lambda: es.bootstrap_ci("max_error", [0.1, 1.2], [0.2, 1.0], sample_weight=[1, 1]), "sample_weight"),
        (lambda: es.proportion_ci(11, 10), "between 0 and n"),
        (lambda: es.proportion_ci(1, 0), "positive"),
        (lambda: es.proportion_ci(1, 10, method="agresti"), "method"),
        (lambda: es.accuracy_ci([0, 1], [0, 1, 1]), "shape"),
        (lambda: es.mcnemar_test([0, 1], [0, 1], [0]), "same shape"),
        (lambda: es.roc_auc_ci([0, 1, 2], [0.1, 0.5, 0.9]), "binary"),
        (lambda: es.roc_auc_ci(["a", "b"], [0.1, 0.9]), "pos_label"),
        (lambda: es.adjust_pvalues([0.1, 1.2]), "between 0 and 1"),
        (lambda: es.adjust_pvalues([0.1], method="sidak"), "method"),
        (lambda: es.cohens_d([1, 2], [1, 2, 3], paired=True), "same length"),
        (lambda: es.paired_bootstrap_test("f1", [0, 1], [0, 1]), "either"),
        (lambda: es.paired_bootstrap_test("f1", [0, 1], [0, 1], [0, 1], alternative="up"), "alternative"),
    ],
)
def test_input_errors(call, match: str) -> None:
    with pytest.raises((es.InputValidationError, es.StatisticalTestError), match=match):
        call()


def test_statistical_errors() -> None:
    with pytest.raises(es.StatisticalTestError):
        es.cohens_d([1.0], [2.0])
    with pytest.raises(es.StatisticalTestError):
        es.cohens_d([1.0, 1.0], [2.0, 2.0])
    with pytest.raises(es.StatisticalTestError):
        es.cohens_d([1.0, 2.0], [1.0, 2.0], paired=True)
    with pytest.raises(es.StatisticalTestError):
        es.hedges_g([1.0], [2.0])
    with pytest.raises(es.MetricInputError):
        es.roc_auc_ci([1, 1, 1], [0.2, 0.4, 0.9])
