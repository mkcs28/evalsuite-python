"""v0.2.0: clinical, calibration and hypothesis tests, checked against hand calculations, SciPy and
statsmodels."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

import evalsuite as es

sm = pytest.importorskip("statsmodels.api")
from statsmodels.stats.contingency_tables import Table2x2  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

# 2x2: TP=3, FP=1, FN=1, TN=5
Y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
P = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]


def _risk(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.4 + 1.3 * x)))).astype(int)
    p = 1 / (1 + np.exp(-(0.1 + 2.0 * x)))  # deliberately miscalibrated
    return y, p


# ---- clinical ------------------------------------------------------------------------------------
def test_diagnostic_metrics_known_answers() -> None:
    assert float(es.sensitivity(Y, P)) == pytest.approx(3 / 4)
    assert float(es.ppv(Y, P)) == pytest.approx(3 / 4)
    sens, spec = 3 / 4, 5 / 6
    assert float(es.lr_positive(Y, P)) == pytest.approx(sens / (1 - spec))
    assert float(es.lr_negative(Y, P)) == pytest.approx((1 - sens) / spec)
    assert float(es.diagnostic_odds_ratio(Y, P)) == pytest.approx(3 * 5 / (1 * 1))
    assert float(es.youden_j(Y, P)) == pytest.approx(sens + spec - 1)
    assert float(es.sensitivity(Y, P)) == float(es.recall(Y, P))
    assert float(es.ppv(Y, P)) == float(es.precision(Y, P))


def test_ratios_are_inf_or_nan_with_warning_never_zero() -> None:
    perfect_spec = [1, 1, 0, 0], [1, 0, 0, 0]
    with pytest.warns(es.UndefinedMetricWarning):
        assert math.isinf(float(es.lr_positive(*perfect_spec)))
    with pytest.warns(es.UndefinedMetricWarning):
        assert math.isinf(float(es.diagnostic_odds_ratio(*perfect_spec)))
    corrected = es.diagnostic_odds_ratio(*perfect_spec, correction=0.5)
    assert float(corrected) == pytest.approx(1.5 * 2.5 / (0.5 * 1.5))
    assert corrected.params["correction_applied"]


def test_diagnostic_metrics_need_binary_and_both_classes() -> None:
    with pytest.raises(es.UnsupportedTaskError):
        es.lr_positive([0, 1, 2], [0, 1, 2])
    with pytest.raises(es.InputValidationError, match="both people"):
        es.youden_j([1, 1, 1], [1, 0, 1])


def test_pos_label_and_weights() -> None:
    y = ["d", "d", "h", "h", "h"]
    p = ["d", "h", "h", "h", "d"]
    assert float(es.sensitivity(y, p, pos_label="d")) == pytest.approx(0.5)
    w = [2, 1, 1, 1, 1]
    assert float(es.sensitivity(y, p, pos_label="d", sample_weight=w)) == pytest.approx(2 / 3)


def test_diagnostic_report_intervals_match_references() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    p = np.where(rng.random(400) < 0.8, y, 1 - y)
    rep = es.diagnostic_report(y, p)
    tp, fp, fn, tn = (int(rep.counts[k]) for k in ("tp", "fp", "fn", "tn"))
    lo, hi = proportion_confint(tp, tp + fn, method="wilson")
    assert (rep["sensitivity"].low, rep["sensitivity"].high) == pytest.approx((lo, hi))
    lo, hi = proportion_confint(tn, tn + fp, method="wilson")
    assert (rep["specificity"].low, rep["specificity"].high) == pytest.approx((lo, hi))
    t = Table2x2(np.array([[tp, fn], [fp, tn]]))
    assert rep["diagnostic_odds_ratio"].estimate == pytest.approx(t.oddsratio)
    assert (rep["diagnostic_odds_ratio"].low, rep["diagnostic_odds_ratio"].high) == pytest.approx(
        t.oddsratio_confint()
    )
    # Simel et al. 1991 log interval, written out independently
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    se = math.sqrt(1 / tp - 1 / (tp + fn) + 1 / fp - 1 / (fp + tn))
    lr = sens / (1 - spec)
    z = stats.norm.ppf(0.975)
    assert (rep["lr_positive"].low, rep["lr_positive"].high) == pytest.approx(
        (lr * math.exp(-z * se), lr * math.exp(z * se))
    )
    assert rep["accuracy"].estimate == pytest.approx((tp + tn) / 400)
    assert "Diagnostic odds ratio" in rep.summary() and rep.to_markdown().startswith("| Measure")
    assert rep.to_dict()["measures"]["youden_j"]["estimate"] == pytest.approx(sens + spec - 1)
    assert rep.to_html().startswith("<!doctype html>") and "\\toprule" in rep.to_latex()


def test_diagnostic_report_zero_cell_uses_haldane() -> None:
    rep = es.diagnostic_report([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0])
    assert "+0.5" in rep["diagnostic_odds_ratio"].method
    assert math.isfinite(rep["diagnostic_odds_ratio"].high)
    assert math.isnan(rep["lr_positive"].low)  # infinite LR+: no interval, not a fake one


def test_decision_curve_matches_formula() -> None:
    y, p = _risk()
    thresholds = np.array([0.1, 0.3, 0.5])
    dc = es.decision_curve(y, {"m": p, "half": p / 2}, thresholds=thresholds)
    n, prev = len(y), y.mean()
    for i, t in enumerate(thresholds):
        tp = np.sum((p >= t) & (y == 1))
        fp = np.sum((p >= t) & (y == 0))
        assert dc.net_benefit["m"][i] == pytest.approx(tp / n - fp / n * t / (1 - t))
        assert dc.treat_all[i] == pytest.approx(prev - (1 - prev) * t / (1 - t))
    assert float(es.net_benefit(y, p, threshold=0.3)) == pytest.approx(dc.net_benefit["m"][1])
    assert dc.prevalence == pytest.approx(prev)
    assert list(dc.to_dataframe().columns)[:2] == ["threshold", "net_benefit[m]"]
    assert dc.to_csv().startswith("threshold,m,half,treat_all,treat_none")
    with pytest.raises(es.InputValidationError):
        es.decision_curve(y, p, thresholds=[0.0, 0.5])


# ---- calibration ---------------------------------------------------------------------------------
def test_calibration_slope_and_intercept_match_statsmodels_glm() -> None:
    y, p = _risk()
    lp = np.log(p / (1 - p))
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    assert float(es.calibration_slope(y, p)) == pytest.approx(fit.params[1], rel=1e-8)
    assert es.calibration_slope(y, p).params["intercept"] == pytest.approx(fit.params[0], rel=1e-8)
    off = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    assert float(es.calibration_intercept(y, p)) == pytest.approx(off.params[0], rel=1e-8)


def test_well_calibrated_predictions_have_slope_near_one() -> None:
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert float(es.calibration_slope(y, p)) == pytest.approx(1, abs=0.06)
    assert float(es.calibration_intercept(y, p)) == pytest.approx(0, abs=0.04)


def test_maximum_calibration_error() -> None:
    y, p = _risk()
    pt, pp, _ = es.calibration_curve(y, p, n_bins=10)
    assert float(es.maximum_calibration_error(y, p)) == pytest.approx(np.max(np.abs(pt - pp)))
    assert float(es.maximum_calibration_error(y, p)) >= float(es.expected_calibration_error(y, p))


def test_hosmer_lemeshow_matches_manual_formula() -> None:
    y, p = _risk()
    res = es.hosmer_lemeshow(y, p, n_groups=10)
    order = np.argsort(p, kind="mergesort")
    stat = 0.0
    for g in np.array_split(order, 10):
        o, e, n = y[g].sum(), p[g].sum(), len(g)
        stat += (o - e) ** 2 / e + ((n - o) - (n - e)) ** 2 / (n - e)
    assert res.statistic == pytest.approx(stat)
    assert res.p_value == pytest.approx(stats.chi2.sf(stat, 8))
    assert res.params["df"] == 8 and res.p_value < 0.05  # the model is miscalibrated
    rng = np.random.default_rng(5)
    q = rng.uniform(0.1, 0.9, 3000)
    assert es.hosmer_lemeshow((rng.random(3000) < q).astype(int), q).p_value > 0.01


# ---- hypothesis tests ----------------------------------------------------------------------------
@pytest.fixture
def ab() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    return rng.normal(0, 1, 25), rng.normal(0.6, 1.4, 25)


def test_t_tests_match_scipy_and_statsmodels(ab: tuple[np.ndarray, np.ndarray]) -> None:
    from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

    a, b = ab
    for equal_var, usevar in ((False, "unequal"), (True, "pooled")):
        r = es.t_test(a, b, equal_var=equal_var)
        ref = stats.ttest_ind(a, b, equal_var=equal_var)
        assert (r.statistic, r.p_value) == pytest.approx((ref.statistic, ref.pvalue))
        cm = CompareMeans(DescrStatsW(a), DescrStatsW(b))
        assert (r.ci.low, r.ci.high) == pytest.approx(cm.tconfint_diff(usevar=usevar))
    r = es.paired_t_test(a, b)
    ref = stats.ttest_rel(a, b)
    assert (r.statistic, r.p_value) == pytest.approx((ref.statistic, ref.pvalue))
    assert (r.ci.low, r.ci.high) == pytest.approx(DescrStatsW(a - b).tconfint_mean())
    assert r.params["cohens_dz"] == pytest.approx(es.cohens_d(a, b, paired=True))
    one = es.t_test(a, b, alternative="less")
    assert one.p_value == pytest.approx(stats.ttest_ind(a, b, equal_var=False, alternative="less").pvalue)
    assert one.ci.low == -math.inf


def test_rank_tests_match_scipy(ab: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = ab
    r = es.mann_whitney_test(a, b)
    ref = stats.mannwhitneyu(a, b)
    assert (r.statistic, r.p_value) == pytest.approx((ref.statistic, ref.pvalue))
    assert r.estimate == pytest.approx(es.cliffs_delta(a, b))
    w = es.wilcoxon_test(a, b)
    ref = stats.wilcoxon(a, b)
    assert (w.statistic, w.p_value) == pytest.approx((ref.statistic, ref.pvalue))
    c = a + 1
    k = es.kruskal_wallis_test(a, b, c)
    ref = stats.kruskal(a, b, c)
    assert (k.statistic, k.p_value) == pytest.approx((ref.statistic, ref.pvalue))
    f = es.friedman_test(a, b, c)
    ref = stats.friedmanchisquare(a, b, c)
    assert (f.statistic, f.p_value) == pytest.approx(tuple(ref))
    assert f.estimate == pytest.approx(ref.statistic / (25 * 2))
    s = es.shapiro_wilk_test(a)
    assert (s.statistic, s.p_value) == pytest.approx(tuple(stats.shapiro(a)))


def test_contingency_tests() -> None:
    t = [[12, 5, 9], [7, 15, 4]]
    r = es.chi_square_test(t)
    chi2, p, dof, _ = stats.chi2_contingency(t)
    assert (r.statistic, r.p_value, r.params["df"]) == pytest.approx((chi2, p, dof))
    assert r.estimate == pytest.approx(stats.contingency.association(t, method="cramer"))
    assert es.cramers_v(t) == pytest.approx(stats.contingency.association(t, method="cramer"))
    assert 0 <= es.cramers_v(t, bias_correction=True) <= es.cramers_v(t)
    two = [[8, 2], [1, 5]]
    f = es.fisher_exact_test(two)
    ref = stats.fisher_exact(two)
    assert (f.statistic, f.p_value) == pytest.approx(tuple(ref))
    with pytest.raises(es.InputValidationError):
        es.fisher_exact_test(t)


def test_hochberg_matches_statsmodels() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.002, 0.049])
    assert es.adjust_pvalues(p, method="hochberg") == pytest.approx(multipletests(p, method="simes-hochberg")[1])
    assert np.all(es.adjust_pvalues(p, method="hochberg") <= es.adjust_pvalues(p, method="holm") + 1e-15)


def test_test_errors_are_explicit() -> None:
    with pytest.raises(es.StatisticalTestError):
        es.t_test([1, 1, 1], [2, 2, 2])
    with pytest.raises(es.StatisticalTestError):
        es.wilcoxon_test([1, 2, 3], [1, 2, 3])
    with pytest.raises(es.InputValidationError):
        es.friedman_test([1, 2], [1, 2, 3], [1, 2])
    with pytest.raises(es.InputValidationError):
        es.t_test([1, 2, 3], [1, 2, 4], alternative="bigger")  # type: ignore[arg-type]


def test_new_metrics_registered_and_documented() -> None:
    ids = set(es.list_metrics())
    for m in (
        "clinical.sensitivity",
        "clinical.ppv",
        "clinical.lr_positive",
        "clinical.lr_negative",
        "clinical.diagnostic_odds_ratio",
        "clinical.youden_j",
        "clinical.net_benefit",
        "calibration.maximum_calibration_error",
        "calibration.calibration_slope",
        "calibration.calibration_intercept",
    ):
        assert m in ids
        info = es.metric_info(m)
        assert info.formula and info.references and info.definition
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = es.bootstrap_ci("youden_j", Y * 10, P * 10, random_state=0)
    assert ci.low <= float(es.youden_j(Y, P)) <= ci.high
