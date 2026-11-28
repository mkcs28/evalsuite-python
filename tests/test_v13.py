"""1.3: calibration suite, multi-label and ranking, forecasting, drift, agreement and Bayesian comparison.

Reference values were computed with pingouin 0.7 (ICC), dieboldmariano (Diebold–Mariano with the HLN
correction), baycomp (correlated Bayesian t-test), properscoring (CRPS), scikit-learn (multi-label, isotonic,
Platt) and pytrec_eval (ranking); the libraries themselves are not needed to run these tests.
"""

import json

import numpy as np
import pytest

import evalsuite as es

# --------------------------------------------------------------------------------------------- agreement

SHROUT_FLEISS = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], float
)


def test_icc_matches_shrout_fleiss_and_pingouin():
    r = es.icc(SHROUT_FLEISS)
    expected = {
        "ICC1": (0.1657417684054755, 1.7946784922394683, 0.16476880834463953),
        "ICC2": (0.28976377952755916, 11.027247956403299, 0.000134566516484335),
        "ICC3": (0.7148407148407154, 11.027247956403299, 0.000134566516484335),
        "ICC1k": (0.44279713367926876, 1.7946784922394683, 0.16476880834463953),
        "ICC2k": (0.6200505475989893, 11.027247956403299, 0.000134566516484335),
        "ICC3k": (0.9093155423770697, 11.027247956403299, 0.000134566516484335),
    }
    for name, (v, f, p) in expected.items():
        form = r[name]
        assert form.value == pytest.approx(v, abs=1e-12)
        assert form.f == pytest.approx(f, rel=1e-10)
        assert form.p_value == pytest.approx(p, rel=1e-8)
        assert form.low < form.value < form.high
    # published 95% CIs (Shrout & Fleiss 1979 data; psych::ICC): ICC3 0.34–0.95, ICC2 0.02–0.76
    assert (round(r["ICC3"].low, 2), round(r["ICC3"].high, 2)) == (0.34, 0.95)
    assert (round(r["ICC2"].low, 2), round(r["ICC2"].high, 2)) == (0.02, 0.76)
    assert "ICC2" in r.summary() and "| ICC3 |" in r.to_markdown()
    json.loads(r.to_json())
    with pytest.raises(KeyError):
        r["ICC9"]


def test_icc_long_format_and_missing():
    subj = np.repeat(np.arange(6), 4)
    rater = np.tile(list("abcd"), 6)
    long = es.icc(SHROUT_FLEISS.ravel(), subjects=subj, raters=rater)
    assert long["ICC2"].value == pytest.approx(es.icc(SHROUT_FLEISS)["ICC2"].value)
    x = SHROUT_FLEISS.copy()
    x[0, 0] = np.nan
    assert es.icc(x).n_subjects == 5
    with pytest.raises(es.InputValidationError, match="missing"):
        es.icc(x, missing="raise")
    with pytest.raises(es.InputValidationError, match="more than one rating"):
        es.icc([1, 2, 3], subjects=[0, 0, 1], raters=["a", "a", "a"])
    with pytest.raises(es.InputValidationError, match="identical"):
        es.icc(np.ones((4, 3)))


def test_bland_altman_and_ccc():
    rng = np.random.default_rng(0)
    truth = rng.normal(100, 15, 200)
    a = truth + rng.normal(0, 3, 200)
    b = truth + 2 + rng.normal(0, 3, 200)
    ba = es.bland_altman(a, b)
    d = a - b
    assert ba.bias == pytest.approx(d.mean())
    assert ba.upper_loa == pytest.approx(d.mean() + 1.959963984540054 * d.std(ddof=1))
    assert ba.bias_low < ba.bias < ba.bias_high < 0
    assert ba.lower_loa_ci[0] < ba.lower_loa < ba.lower_loa_ci[1]
    assert 0.9 < ba.within_loa <= 1
    assert "Bias" in ba.summary() and "| Bias |" in ba.to_markdown()
    json.loads(ba.to_json())
    ccc = es.lins_ccc(a, b)
    # Lin's CCC by its definition
    sx, sy = a.var(), b.var()
    sxy = np.mean((a - a.mean()) * (b - b.mean()))
    assert ccc.value == pytest.approx(2 * sxy / (sx + sy + (a.mean() - b.mean()) ** 2))
    assert ccc.params["ci_low"] < ccc.value < ccc.params["ci_high"]
    assert es.lins_ccc(a, a).value == pytest.approx(1.0)
    with pytest.raises(es.InputValidationError, match="same subjects"):
        es.bland_altman([1, 2, 3], [1, 2])


# ----------------------------------------------------------------------------------------- Bayesian


def test_correlated_bayesian_ttest_matches_baycomp():
    sa = [0.81, 0.83, 0.80, 0.84, 0.82, 0.85, 0.80, 0.83, 0.81, 0.84]
    sb = [0.80, 0.82, 0.80, 0.82, 0.81, 0.83, 0.79, 0.82, 0.80, 0.83]
    r = es.bayes_correlated_ttest(sa, sb, rope=0.01)
    assert r.p_a_better == pytest.approx(0.6448432986886752, abs=1e-12)
    assert r.p_rope == pytest.approx(0.3551461891611398, abs=1e-12)
    assert r.p_b_better == pytest.approx(1.051215018499363e-05, rel=1e-9)
    assert r.decision() == "undecided" and r.p_rope_or_better > 0.99
    lower = es.bayes_correlated_ttest(sb, sa, rope=0.01, higher_is_better=False)
    assert lower.p_a_better == pytest.approx(r.p_a_better)


def test_bayes_compare_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1500)
    a = np.where(rng.random(1500) < 0.9, y, 1 - y)
    b = np.where(rng.random(1500) < 0.8, y, 1 - y)
    r = es.bayes_compare(y, a, b, n_samples=800, random_state=0)
    assert r.p_a_better > 0.99 and r.decision() == "A better"
    assert r.low < r.mean < r.high
    assert r.p_a_better + r.p_rope + r.p_b_better == pytest.approx(1.0)
    same = es.bayes_compare(y, a, a, n_samples=200, random_state=0)
    assert same.p_rope == 1.0 and same.decision() == "equivalent"
    pa = np.clip(0.6 * y + 0.4 * rng.random(1500), 0, 1)
    brier = es.bayes_compare(y, pa, rng.random(1500), metric="brier_score", n_samples=200, random_state=0)
    assert brier.higher_is_better is False and brier.p_a_better > 0.99
    assert "P(A better)" in r.summary() and "| P(B better) |" in r.to_markdown()
    json.loads(r.to_json())
    with pytest.raises(es.InputValidationError):
        es.bayes_compare(y, a, b[:10])


# ---------------------------------------------------------------------------------------------- drift


def test_drift_report_flags_real_shifts_only():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 4000
    ref = pd.DataFrame(
        {
            "age": rng.normal(50, 10, n),
            "bmi": rng.normal(27, 4, n),
            "sex": rng.choice(["F", "M"], n),
            "site": rng.choice(["a", "b", "c"], n),
            "lab": rng.normal(1, 1, n),
        }
    )
    cur = ref.copy()
    cur["age"] = rng.normal(54, 10, n)
    cur["site"] = rng.choice(["a", "b", "c", "d"], n, p=[0.4, 0.3, 0.2, 0.1])
    cur.loc[:400, "lab"] = np.nan
    rep = es.drift_report(ref, cur)
    assert set(rep.drifted()) == {"age", "site", "lab"}
    assert rep["site"]["new_categories"] == ["d"] and rep["lab"]["missing_shift"]
    assert rep.share_drifted == pytest.approx(0.6)
    assert "DRIFT" in rep.summary() and "| age |" in rep.to_markdown() and "<table>" in rep.to_html()
    json.loads(rep.to_json())
    assert len(rep.to_dataframe()) == 5
    arr = es.drift_report(rng.normal(size=(300, 2)), rng.normal(size=(300, 2)), features=["u", "v"])
    assert arr.drifted() == []
    with pytest.raises(es.InputValidationError, match="lacks"):
        es.drift_report({"a": [1, 2, 3]}, {"b": [1, 2, 3]})


# ---------------------------------------------------------------------------------------- forecasting


def test_forecasting_metrics_against_references():
    rng = np.random.default_rng(42)
    y = rng.normal(size=12)
    a = y + rng.normal(size=12)
    b = y + rng.normal(0, 1.5, 12)
    dm = es.diebold_mariano(y, a, b, h=2)
    assert dm.value == pytest.approx(-1.47431633661036, rel=1e-9)
    assert dm.params["p_value"] == pytest.approx(0.16843177684422095, rel=1e-9)
    assert es.diebold_mariano(y, a, b, alternative="less").params["p_value"] < 0.5
    ens = np.array([[0.0, 1, 2], [1, 1.5, 4]])
    yy = np.array([1.2, 3.0])
    assert es.crps(yy, ensemble=ens).value == pytest.approx(0.5611111111111111)
    assert es.crps(yy, mean=[1, 2], sd=[0.5, 1.5]).value == pytest.approx(0.37770930566257577)
    assert es.crps(yy, mean=yy, sd=0).value == 0
    train = np.array([1.0, 3, 2, 5, 4, 6])
    assert es.mase([7, 8], [6, 9], y_train=train).value == pytest.approx(1 / 1.8)  # naive MAE = 1.8
    assert es.rmsse([7, 8], [6, 9], y_train=train).value == pytest.approx(np.sqrt(1 / 3.8))
    pl = es.pinball_loss([1.0, 2.0], [[0.5, 1.5], [2.5, 2.0]], quantiles=[0.1, 0.9])
    assert pl.params["per_quantile"][0.1] == pytest.approx((0.1 * 0.5 + 0.9 * 0.5) / 2)
    isc = es.interval_score([1.0, 5.0], [0.0, 0.0], [2.0, 4.0], level=0.8)
    assert isc.value == pytest.approx((2 + 4 + (2 / 0.2) * 1) / 2)
    assert es.msis([1.0, 5.0], [0.0, 0.0], [2.0, 4.0], y_train=train, level=0.8).value == pytest.approx(
        isc.value / 1.8
    )
    cov = es.interval_coverage([1.0, 5.0, 2.0, 3.0], [0, 0, 0, 0], [2, 4, 3, 3], level=0.9)
    assert cov.value == 0.75 and 0 <= cov.params["p_value"] <= 1
    with pytest.raises(es.InputValidationError, match="constant"):
        es.mase([1, 2], [1, 2], y_train=[3, 3, 3])
    with pytest.raises(es.InputValidationError):
        es.crps([1.0])


def test_forecast_report():
    rng = np.random.default_rng(3)
    train = np.cumsum(rng.normal(size=100))
    y = train[-1] + np.cumsum(rng.normal(size=30))
    good, bad = y + rng.normal(0, 0.3, 30), y + rng.normal(0, 2, 30)
    rep = es.forecast_report(y, {"naive": np.full(30, train[-1]), "good": good, "bad": bad}, y_train=train)
    assert rep.best() == "good"
    assert rep.rows[1]["dm_p_adjusted"] < 0.05
    assert "MASE" in rep.summary() and "| good |" in rep.to_markdown()
    json.loads(rep.to_json())


# ------------------------------------------------------------------------------------------- calibration


def test_recalibration_methods():
    rng = np.random.default_rng(0)
    p = rng.random(3000)
    y = (rng.random(3000) < p**2).astype(int)  # over-confident model
    p_val, y_val, p_test, y_test = p[:1500], y[:1500], p[1500:], y[1500:]
    raw_ece = es.expected_calibration_error(y_test, p_test).value
    for method in ("platt", "isotonic", "temperature", "beta", "histogram"):
        cal = es.recalibrate(y_val, p_val, method=method)
        q = cal(p_test)
        assert q.shape == p_test.shape and np.all((q >= 0) & (q <= 1))
        if method != "temperature":  # one symmetric parameter cannot fix this kind of miscalibration
            assert es.expected_calibration_error(y_test, q).value < raw_ece
        assert method in repr(cal)
    # isotonic equals the pool-adjacent-violators fit (checked against scikit-learn)
    iso = es.recalibrate([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], method="isotonic")
    np.testing.assert_allclose(iso([0.1, 0.2, 0.3, 0.4, 0.0, 1.0]), [0, 0.5, 0.5, 1, 0, 1])
    probs = rng.dirichlet(np.ones(3) * 0.3, 600)
    labels = np.array([rng.choice(3, p=pp**0.5 / (pp**0.5).sum()) for pp in probs])
    temp = es.recalibrate(labels, probs, method="temperature")
    out = temp(probs)
    assert temp.params["temperature"] > 1 and np.allclose(out.sum(axis=1), 1)
    with pytest.raises(es.InputValidationError, match="method"):
        es.recalibrate(y_val, p_val, method="nope")
    with pytest.raises(es.InputValidationError, match="both outcomes"):
        es.recalibrate([1, 1, 1], [0.2, 0.5, 0.9])


def test_calibration_report_intervals():
    rng = np.random.default_rng(1)
    p = rng.random(800)
    y = (rng.random(800) < p).astype(int)
    rep = es.calibration_report(y, p, n_resamples=200, random_state=0)
    for k in ("brier_score", "calibration_slope", "calibration_intercept"):
        lo, hi = rep.intervals[k]
        assert lo <= rep[k] <= hi
    assert rep.intervals["expected_calibration_error"][0] < rep.intervals["expected_calibration_error"][1]
    assert "bootstrap CIs" in rep.summary() and " to " in rep.to_markdown()
    assert json.loads(rep.to_json())["intervals"]
    assert not es.calibration_report(y, p).intervals


# ----------------------------------------------------------------------------------- multi-label / ranking


def test_multilabel_report_matches_sklearn_definitions():
    Y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
    S = np.array([[0.9, 0.2, 0.6], [0.1, 0.8, 0.4], [0.7, 0.3, 0.2], [0.2, 0.1, 0.9], [0.6, 0.5, 0.1]])
    rep = es.multilabel_report(Y, y_score=S, n_resamples=100, random_state=0, labels=["a", "b", "c"])
    # values from scikit-learn 1.6
    assert rep["subset_accuracy"] == pytest.approx(0.6)
    assert rep["hamming_loss"] == pytest.approx(2 / 15)
    assert rep["micro_f1"] == pytest.approx(0.8571428571428571)
    assert rep["label_ranking_average_precision"] == pytest.approx(1.0)
    assert rep["coverage_error"] == pytest.approx(1.4)
    assert rep["ranking_loss"] == pytest.approx(0.0)
    assert rep.labels[1]["support"] == 2
    assert "subset_accuracy" in rep.summary() and "| b |" in rep.to_markdown()
    json.loads(rep.to_json())
    with pytest.raises(es.InputValidationError, match="only 0 / 1"):
        es.multilabel_report([[2, 0]], [[1, 0]])


def test_ranking_report_trec_conventions():
    qrels = {"q1": {"d1": 2, "d2": 1, "d3": 0}, "q2": {"d4": 1, "d5": 1}, "q3": {"d9": 0}}
    run_a = {"q1": ["d1", "d3", "d2"], "q2": {"d5": 0.9, "d6": 0.5, "d4": 0.1}}
    run_b = {"q1": ["d3", "d2", "d1"], "q2": ["d6", "d7", "d8"]}
    rep = es.ranking_report(
        qrels, {"a": run_a, "b": run_b}, k=3, n_resamples=200, n_permutations=500, random_state=0
    )
    pq = rep.per_query["a"]
    assert pq["q1"]["ap"] == pytest.approx((1 + 2 / 3) / 2)
    assert pq["q2"]["rr"] == 1.0 and pq["q2"]["recall"] == 1.0
    idcg = 2 + 1 / np.log2(3)
    assert pq["q1"]["ndcg"] == pytest.approx((2 + 1 / np.log2(4)) / idcg)
    assert rep.settings["n_queries"] == 2  # q3 has no relevant document
    assert rep.best() == "a"
    assert rep.systems[1]["test"]["difference"] < 0
    assert "baseline" in rep.summary() and "| a |" in rep.to_markdown()
    json.loads(rep.to_json())
    with pytest.raises(es.InputValidationError, match="primary"):
        es.ranking_report(qrels, {"a": run_a}, primary="nope")


# ------------------------------------------------------------------------------------ speed and inputs


def test_bayes_curve_fast_path_equals_weighted_loop(monkeypatch):
    from evalsuite import bayes

    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 200)
    pa, pb = np.round(rng.random(200), 2), np.round(rng.random(200), 2)
    for metric in ("roc_auc", "average_precision"):
        fast = es.bayes_compare(y, pa, pb, metric=metric, n_samples=200, random_state=1)
        with monkeypatch.context() as m:
            m.setattr(bayes, "_curve_fast_path", lambda *a: None)
            slow = es.bayes_compare(y, pa, pb, metric=metric, n_samples=200, random_state=1)
        np.testing.assert_allclose(fast.samples, slow.samples, atol=1e-12)


def test_multilabel_bootstrap_resamples_examples():
    rng = np.random.default_rng(2)
    Y = (rng.random((60, 4)) < 0.3).astype(int)
    S = np.clip(Y * 0.4 + rng.random((60, 4)), 0, 1)
    rep = es.multilabel_report(Y, y_score=S, n_resamples=300, random_state=0)
    for name, m in rep.metrics.items():
        assert m["low"] <= m["high"], name
    assert rep.metrics["micro_f1"]["low"] < rep["micro_f1"] < rep.metrics["micro_f1"]["high"]


@pytest.mark.parametrize("lib", ["polars", "pyarrow"])
def test_dataframe_libraries_as_inputs(lib):
    mod = pytest.importorskip(lib)
    if lib == "polars":
        series, table = mod.Series, lambda d: mod.DataFrame(d)
    else:
        series, table = mod.array, mod.table
    assert es.accuracy(series([0, 1, 1, 0]), series([0, 1, 0, 0])).value == 0.75
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": ["x", "y", "x", "y", "x"]}
    assert es.drift_report(table(data), table(data)).drifted() == []
