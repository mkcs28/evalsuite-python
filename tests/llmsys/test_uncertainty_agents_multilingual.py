"""v0.5.0 uncertainty, agent and multilingual metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

import evalsuite as es


def _data(seed: int = 0, n: int = 300):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0, 1, n)
    correct = rng.random(n) < conf * 0.8 + 0.1
    return correct, conf


def test_ace_equal_mass_bins() -> None:
    c, p = _data()
    order = np.argsort(p, kind="mergesort")
    gaps = [abs(c[i].mean() - p[i].mean()) for i in np.array_split(order, 10)]
    assert float(es.adaptive_calibration_error(c, p, n_bins=10)) == pytest.approx(np.mean(gaps))
    assert es.adaptive_calibration_error(c[:3], p[:3], n_bins=10).params["n_bins"] == 3
    with pytest.raises(es.InputValidationError):
        es.adaptive_calibration_error(c, p, n_bins=0)
    with pytest.raises(es.InputValidationError):
        es.adaptive_calibration_error(c, p * 2)


def test_selective_prediction_hand_example() -> None:
    correct = [True, True, False, True, False]
    conf = [0.9, 0.8, 0.7, 0.6, 0.5]
    # risks at k = 1..5: 0, 0, 1/3, 1/4, 2/5
    a = es.aurc(correct, conf)
    assert float(a) == pytest.approx(np.mean([0, 0, 1 / 3, 1 / 4, 2 / 5]))
    # optimal ordering puts both errors last: 0, 0, 0, 1/4, 2/5
    assert a.params["optimal_aurc"] == pytest.approx(np.mean([0, 0, 0, 1 / 4, 2 / 5]))
    assert float(es.selective_risk(correct, conf, coverage=0.6)) == pytest.approx(1 / 3)
    assert float(es.risk_at_coverage(correct, conf, coverage=0.4)) == 0.0
    cr = es.coverage_at_risk(correct, conf, risk=0.25)
    assert float(cr) == 0.8 and cr.params["threshold"] == 0.6
    assert float(es.coverage_at_risk([False, False], [0.9, 0.8], risk=0.1)) == 0.0
    cov, risk, _thr = es.risk_coverage_curve([True, False, True], [0.5, 0.5, 0.9])
    assert cov.tolist() == pytest.approx([1 / 3, 1.0]) and risk.tolist() == pytest.approx([0.0, 1 / 3])
    for bad in (
        lambda: es.selective_risk(correct, conf, coverage=0),
        lambda: es.risk_at_coverage(correct, conf, coverage=2),
        lambda: es.coverage_at_risk(correct, conf, risk=1),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_confidence_auroc_matches_sklearn() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    from scipy import stats

    c, p = _data(1)
    r = es.confidence_accuracy_correlation(c, p)
    assert float(r) == pytest.approx(skm.roc_auc_score(c, p), abs=1e-12)
    assert r.params["spearman"] == pytest.approx(stats.spearmanr(c, p)[0])
    assert math.isnan(es.confidence_accuracy_correlation([True, False], [0.5, 0.5]).params["spearman"])
    with pytest.raises(es.InputValidationError):
        es.confidence_accuracy_correlation([True, True], [0.2, 0.3])


def test_existing_calibration_functions_accept_llm_confidence() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    c, p = _data(2)
    assert float(es.brier_score(c.astype(int), p)) == pytest.approx(skm.brier_score_loss(c, p))
    assert float(es.log_loss(c.astype(int), np.column_stack([1 - p, p]))) == pytest.approx(skm.log_loss(c, p))


TRAJ = [
    [
        {"name": "search", "arguments": {"q": "a"}, "ok": False},
        {"name": "search", "arguments": '{"q": "b"}', "ok": True},
        {"name": "search", "arguments": {"q": "b"}, "ok": True},
    ],
    [
        {"name": "calc", "arguments": {"x": "1"}, "ok": True},
        {"name": "nope", "arguments": {}},
        {"name": "calc", "arguments": "{bad", "ok": False},
    ],
    [{"name": "calc", "arguments": {"x": 2}}] * 3,
]
TOOLS = {
    "search": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    "calc": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]},
}


def test_agent_trajectory_metrics() -> None:
    inv = es.invalid_tool_call_rate(TRAJ, TOOLS)
    # invalid: calc x="1" (schema), nope (unknown), "{bad" (unparseable) -> 3 of 9
    assert float(inv) == pytest.approx(3 / 9)
    assert inv.params["by_reason"] == {
        "unknown_tool": pytest.approx(1 / 9),
        "unparseable_arguments": pytest.approx(1 / 9),
        "schema": pytest.approx(1 / 9),
    }
    assert inv.params["execution_failure_rate"] == pytest.approx(2 / 5)
    rec = es.tool_failure_recovery_rate(TRAJ)
    assert float(rec) == 0.5  # search recovered, calc did not
    un = es.unnecessary_tool_call_rate(TRAJ)
    assert float(un) == pytest.approx(3 / 9) and un.params["loop_rate"] == pytest.approx(1 / 3)
    for bad in (
        lambda: es.invalid_tool_call_rate(TRAJ, {}),
        lambda: es.invalid_tool_call_rate([[]], TOOLS),
        lambda: es.tool_failure_recovery_rate([[{"name": "a", "ok": True}]]),
        lambda: es.unnecessary_tool_call_rate([[]]),
        lambda: es.unnecessary_tool_call_rate(TRAJ, loop_length=1),
        lambda: es.invalid_tool_call_rate([[{"arguments": {}}]], TOOLS),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_agent_task_metrics() -> None:
    t = es.task_completion_rate([True, False, True, True], progress=[1, 0.5, 1, 1])
    assert float(t) == 0.75 and t.params["progress_rate"] == pytest.approx(0.875)
    e = es.tool_use_efficiency([4, 2, 0, 0], [2, 2, 0, 1])
    assert float(e) == pytest.approx((0.5 + 1 + 1 + 0) / 4) and e.params["mean_excess_calls"] == 0.5
    s = es.steps_per_task([3, 5, 10], tool_calls=[1, 2, 4], completed=[True, True, False])
    assert float(s) == 6 and s.params["mean_steps_completed"] == 4 and s.params["steps"]["p50"] == 5
    assert math.isnan(es.steps_per_task([1], completed=[False]).params["mean_steps_completed"])
    pl = es.plan_adherence([["a", "b", "c"], ["x"], []], [["a", "c", "d"], ["x"], []])
    assert pl.params["precision"] == pytest.approx((2 / 3 + 1 + 1) / 3)
    assert float(pl) == pytest.approx((2 / 3 + 1 + 1) / 3) and pl.params["exact_match"] == pytest.approx(2 / 3)
    st = es.state_tracking_accuracy([{"a": 1, "b": 2}, {"a": 1}], [{"a": 1, "b": 2}, {"a": 2, "c": 3}])
    assert float(st) == 0.5 and st.params["slot_accuracy"] == pytest.approx(2 / 4)
    with pytest.raises(es.InputValidationError):
        es.state_tracking_accuracy([1], [{}])
    h = es.human_intervention_rate([0, 2, 0, 1])
    assert float(h) == 0.5 and h.params["mean_per_task"] == 0.75
    c = es.agent_cost_per_task([0.1, 0.3, 0.2], durations=[10, 20, 30], completed=[True, False, True])
    assert float(c) == pytest.approx(0.2) and c.params["cost_per_success"] == pytest.approx(0.3)
    assert es.agent_cost_per_task([1.0], completed=[False]).params["cost_per_success"] == math.inf


def test_language_metrics() -> None:
    skm = pytest.importorskip("sklearn.metrics")
    t = ["en", "fr", "fr", "de", "en"]
    p = ["en", "fr", "en", "de", "en"]
    r = es.language_id_accuracy(t, p)
    assert float(r) == 0.8 and r.params["macro_f1"] == pytest.approx(skm.f1_score(t, p, average="macro"))
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 16))
    tgt = src + rng.normal(0, 0.1, src.shape)
    b = es.bitext_mining_accuracy(src, tgt)
    assert float(b) == 1.0 and b.params["mean_pair_cosine"] > 0.9
    assert float(es.bitext_mining_accuracy(src, tgt, scoring="margin")) == 1.0
    noisy = es.bitext_mining_accuracy(src, rng.normal(size=(20, 16)))
    assert float(noisy) < 0.5
    for bad in (
        lambda: es.bitext_mining_accuracy(src, tgt[:5]),
        lambda: es.bitext_mining_accuracy(src, tgt, scoring="x"),
        lambda: es.bitext_mining_accuracy([1, 2], [1, 2]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    par = es.language_parity({"en": [1, 1, 1, 0], "sw": [1, 0, 0, 0], "fr": 0.5})
    assert float(par) == pytest.approx(0.25 / 0.75) and par.params["worst_language"] == "sw"
    with pytest.raises(es.InputValidationError):
        es.language_parity({"en": [1]})
    with pytest.raises(es.InputValidationError):
        es.language_parity({"en": 0, "fr": 0})
    cs = es.code_switching_robustness([True, True, False], [True, False, False])
    assert float(cs) == pytest.approx(1 / 3) and cs.params["flip_rate"] == 0.5


def test_direct_assessment_z_scores() -> None:
    scores = [60, 80, 100, 10, 20, 30]
    raters = ["a", "a", "a", "b", "b", "b"]
    r = es.direct_assessment(scores, raters, systems=["s1", "s2", "s1", "s1", "s2", "s1"])
    assert float(r) == pytest.approx(0.0, abs=1e-12)
    assert r.params["by_system_z"]["s2"] == pytest.approx(0.0)
    assert r.params["by_system_z"]["s1"] == pytest.approx(0.0)
    assert float(es.direct_assessment([50, 70], ["a", "b"])) == 0.0


def test_culture_and_consistency() -> None:
    c = es.cultural_appropriateness([5, 3, 1, 5], regions=["IN", "IN", "BR", "BR"])
    assert float(c) == pytest.approx((1 + 0.5 + 0 + 1) / 4) and c.params["region_gap"] == pytest.approx(0.25)
    with pytest.raises(es.InputValidationError):
        es.cultural_appropriateness([3], scale=(5, 1))
    lc = es.language_consistency(["hi", "hi", "en"], ["hi", "en", "en"])
    assert float(lc) == pytest.approx(2 / 3) and lc.params["by_language"] == {"en": 1.0, "hi": 0.5}
    x = es.cross_lingual_consistency([{"en": "Paris", "fr": "paris"}, {"en": "1", "fr": "2", "de": "1"}])
    assert float(x) == 0.5 and x.params["pairwise_agreement"] == pytest.approx((1 + 1 / 3) / 2)
    assert float(es.cross_lingual_consistency([{"en": 1, "fr": 1.0}], normalize=float)) == 1.0
    with pytest.raises(es.InputValidationError):
        es.cross_lingual_consistency([{"en": "x"}])
