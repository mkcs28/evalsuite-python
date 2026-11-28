"""1.3: input validation, edge cases and exports of the new modules."""

import json

import numpy as np
import pytest

import evalsuite as es

E = es.InputValidationError


def test_agreement_edges():
    with pytest.raises(E, match="numeric"):
        es.icc([["a", "b"], ["c", "d"]])
    with pytest.raises(E, match="2-D"):
        es.icc([1, 2, 3])
    with pytest.raises(E, match="both subjects"):
        es.icc([1, 2], subjects=[0, 1])
    with pytest.raises(E, match="same length"):
        es.icc([1, 2, 3], subjects=[0, 1], raters=["a", "b", "c"])
    with pytest.raises(E, match="at least 2"):
        es.icc([[1.0, 2.0]])
    with pytest.raises(E, match="level"):
        es.icc(np.eye(3), level=1.5)
    r = es.icc(np.array([[1.0, 2.0], [3.0, 3.5], [5.0, 6.5]]))
    assert repr(r).startswith("Intraclass") and r.to_dict()["n_raters"] == 2
    with pytest.raises(E, match="numeric"):
        es.bland_altman(["a", "b", "c"], [1, 2, 3])
    with pytest.raises(E, match="finite"):
        es.bland_altman([1, 2, np.nan], [1, 2, 3])
    with pytest.raises(E, match="at least 3"):
        es.bland_altman([1, 2], [1, 2])
    ba = es.bland_altman([1.0, 2.0, 3.0, 4.0], [1.1, 2.2, 2.9, 4.1])
    assert repr(ba).startswith("Bland") and "bias" in ba.to_dict()
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    ax = ba.plot()
    assert ax.get_title() == "Bland–Altman"
    const = es.bland_altman([1.0, 1.0, 1.0], [0.5, 0.5, 0.5])
    assert np.isnan(const.proportional_bias_slope)
    with pytest.raises(E, match="constant and equal"):
        es.lins_ccc([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])


def test_bayes_edges():
    y = np.array([0, 1] * 50)
    with pytest.raises(E, match="n_samples"):
        es.bayes_compare(y, y, y, n_samples=10)
    with pytest.raises(E, match="level"):
        es.bayes_compare(y, y, y, level=2)
    with pytest.raises(E, match="low ≤ high"):
        es.bayes_compare(y, y, y, rope=(0.1, -0.1))
    with pytest.raises(E, match="At least 2"):
        es.bayes_compare([1], [1], [1])
    r = es.bayes_compare(y, y, 1 - y, rope=(-0.05, 0.05), n_samples=100, random_state=0)
    assert r.decision() == "A better" and r.rope == (-0.05, 0.05)
    worse = es.bayes_compare(y, 1 - y, y, n_samples=100, random_state=0)
    assert worse.decision() == "B better"

    def no_weights(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    with pytest.raises(E, match="sample_weight"):
        es.bayes_compare(y, y, 1 - y, metric=no_weights, n_samples=100)
    with pytest.raises(E, match="per fold"):
        es.bayes_correlated_ttest([1, 2], [1])
    with pytest.raises(E, match="finite"):
        es.bayes_correlated_ttest([1.0], [1.0])
    with pytest.raises(E, match="level"):
        es.bayes_correlated_ttest([1, 2], [1, 2], level=0)
    with pytest.raises(E, match="test_fraction"):
        es.bayes_correlated_ttest([1, 2], [1, 2], test_fraction=1.0)
    same = es.bayes_correlated_ttest([0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
    assert same.p_rope == 1.0
    lower = es.bayes_correlated_ttest([0.1, 0.12, 0.11], [0.2, 0.21, 0.19], higher_is_better=False, rope=0.0)
    assert lower.p_a_better > 0.95
    assert json.loads(lower.to_json())["decision"] == "A better"


def test_drift_edges():
    rng = np.random.default_rng(0)
    with pytest.raises(E, match="feature names"):
        es.drift_report(rng.normal(size=(10, 2)), rng.normal(size=(10, 2)), features=["a"])
    with pytest.raises(E, match="1-D / 2-D"):
        es.drift_report(rng.normal(size=(2, 2, 2)), rng.normal(size=(2, 2, 2)))
    with pytest.raises(E, match="one-dimensional"):
        es.drift_report({"a": np.ones((3, 2))}, {"a": np.ones((3, 2))})
    with pytest.raises(E, match="alpha"):
        es.drift_report({"a": [1, 2]}, {"a": [1, 2]}, alpha=1)
    with pytest.raises(E, match="too few"):
        es.drift_report({"a": [1.0, np.nan]}, {"a": [1.0, 2.0]})
    codes = {"grade": rng.integers(1, 4, 500), "x": rng.normal(size=500)}
    later = {"grade": rng.integers(1, 5, 500), "x": rng.normal(size=500)}
    rep = es.drift_report(codes, later, categorical=["grade"])
    assert rep["grade"]["type"] == "categorical" and "grade" in rep.drifted()
    obj = {"c": np.array(["a", None, "b"] * 50, dtype=object)}
    assert es.drift_report(obj, obj)["c"]["missing_reference"] == pytest.approx(1 / 3)
    single = es.drift_report({"c": ["a"] * 20}, {"c": ["a"] * 20})
    assert single["c"]["p_value"] == 1.0
    with pytest.raises(KeyError):
        rep["nope"]
    assert json.loads(rep.to_json())["settings"]["correction"] == "holm"


def test_forecasting_edges():
    with pytest.raises(E, match="numeric"):
        es.mase(["a"], [1], y_train=[1, 2, 3])
    with pytest.raises(E, match="non-empty"):
        es.mase([], [], y_train=[1, 2])
    with pytest.raises(E, match="same length"):
        es.mase([1, 2], [1], y_train=[1, 2, 3])
    with pytest.raises(E, match="m \\(seasonal"):
        es.mase([1], [1], y_train=[1, 2], m=0)
    with pytest.raises(E, match="more than m"):
        es.mase([1], [1], y_train=[1, 2], m=2)
    with pytest.raises(E, match="strictly between"):
        es.pinball_loss([1.0], [1.0], quantiles=[1.0])
    with pytest.raises(E, match="shape"):
        es.pinball_loss([1.0, 2.0], [[1.0, 2.0]], quantiles=[0.5, 0.9])
    assert es.pinball_loss([1.0, 3.0], [2.0, 2.0], quantiles=0.5).value == pytest.approx(0.5)
    with pytest.raises(E, match="level"):
        es.interval_score([1], [0], [2], level=1)
    with pytest.raises(E, match="same length"):
        es.interval_score([1, 2], [0], [2])
    with pytest.raises(E, match="at most"):
        es.interval_score([1], [2], [0])
    with pytest.raises(E, match="rows"):
        es.crps([1.0, 2.0], ensemble=[[1.0]])
    assert es.crps([1.0], ensemble=[2.0]).value == pytest.approx(1.0)  # one member: absolute error
    with pytest.raises(E, match="sd ≥ 0"):
        es.crps([1.0], mean=[1.0], sd=[-1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(E, match="h must"):
        es.diebold_mariano(y, y + 1, y - 1, h=0)
    with pytest.raises(E, match="'squared' or 'absolute'"):
        es.diebold_mariano(y, y + 1, y - 1, loss="huber")
    with pytest.raises(E, match="more observations"):
        es.diebold_mariano(y[:2], y[:2], y[:2])
    with pytest.raises(E, match="not positive"):
        es.diebold_mariano(y, y + 1, y - 1)
    a, b = y + np.array([0.1, -0.2, 0.1, 0.3, -0.1]), y + np.array([1.0, -0.5, 0.8, -1.2, 0.9])
    assert (
        es.diebold_mariano(a, a, b, loss="absolute", harvey=False, alternative="greater").params["p_value"] > 0.5
    )
    with pytest.raises(E, match="alternative"):
        es.diebold_mariano(y, a, b, alternative="both")
    with pytest.raises(E, match="non-empty mapping"):
        es.forecast_report(y, {})
    with pytest.raises(E, match="baseline"):
        es.forecast_report(y, {"a": a}, baseline="z")
    rep = es.forecast_report(y, {"a": a, "b": b, "same": a})
    assert rep.best() == "a" and "MASE" not in rep.to_markdown()
    assert "DM vs baseline" in rep.summary() and repr(rep)


def test_multilabel_and_ranking_edges():
    Y = np.array([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(E, match="matrix"):
        es.multilabel_report([1, 0], [1, 0])
    with pytest.raises(E, match="Give y_pred"):
        es.multilabel_report(Y)
    with pytest.raises(E, match="finite"):
        es.multilabel_report(Y, y_score=[[np.nan, 0], [0, 1], [1, 1]])
    with pytest.raises(E, match="same shape"):
        es.multilabel_report(Y, [[1, 0]])
    with pytest.raises(E, match="At least 2"):
        es.multilabel_report([[1, 0]], [[1, 0]])
    with pytest.raises(E, match="label names"):
        es.multilabel_report(Y, Y, labels=["a"])
    rep = es.multilabel_report(Y, Y, n_resamples=50, random_state=0)
    assert rep["subset_accuracy"] == 1.0 and repr(rep).startswith("Multi-label")
    qrels = {"q1": {"d1": 1}}
    with pytest.raises(E, match="k must"):
        es.ranking_report(qrels, {"a": {"q1": ["d1"]}}, k=0)
    with pytest.raises(E, match="runs must"):
        es.ranking_report(qrels, {})
    with pytest.raises(E, match="has a relevant"):
        es.ranking_report({"q1": {"d1": 0}}, {"a": {"q1": ["d1"]}})
    with pytest.raises(E, match="baseline"):
        es.ranking_report(qrels, {"a": {"q1": ["d1"]}}, baseline="z")
    one = es.ranking_report(qrels, {"a": {"q1": ["d1"]}}, n_resamples=20)
    assert one.systems[0]["ndcg"]["value"] == 1.0 and "test" not in one.systems[0] and repr(one)


def test_recalibration_edges():
    p = np.linspace(0.05, 0.95, 40)
    y = (p > 0.5).astype(int)
    y[::7] = 1 - y[::7]
    with pytest.raises(E, match="equal length"):
        es.recalibrate(y, p[:-1])
    with pytest.raises(E, match="At least 2"):
        es.recalibrate([1], [0.5])
    with pytest.raises(E, match="probabilities in"):
        es.recalibrate(y, p * 2)
    with pytest.raises(E, match="two classes"):
        es.recalibrate(np.arange(40) % 3, p)
    with pytest.raises(E, match="sample_weight"):
        es.recalibrate(y, p, sample_weight=-np.ones(40))
    with pytest.raises(E, match="n_bins"):
        es.recalibrate(y, p, method="histogram", n_bins=0)
    cal = es.recalibrate(y, p, method="platt", sample_weight=np.ones(40))
    with pytest.raises(E, match="within"):
        cal([1.5])
    with pytest.raises(E, match="1-D"):
        cal([[0.5, 0.5]])
    iso = es.recalibrate([0, 1], [0.3, 0.3], method="isotonic")
    assert iso([0.1, 0.9]).tolist() == [0.5, 0.5]
    hist = es.recalibrate(y, np.full(40, 0.4), method="histogram")
    assert np.allclose(hist([0.1, 0.9]), y.mean())
    with pytest.raises(E, match="n × k"):
        es.recalibrate([0, 1], np.ones((3, 2)) / 2, method="temperature")
    with pytest.raises(E, match="classes"):
        es.recalibrate([0, 1, 2], np.ones((3, 2)) / 2, method="temperature")
    temp = es.recalibrate([0, 1, 1], np.array([[0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]), method="temperature")
    with pytest.raises(E, match="probability columns"):
        temp(np.ones((2, 3)) / 3)
    strings = es.recalibrate(
        np.array(["no", "yes", "yes"]), np.array([[0.6, 0.4], [0.3, 0.7], [0.2, 0.8]]), method="temperature"
    )
    assert strings.params["temperature"] > 0
