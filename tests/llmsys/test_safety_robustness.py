"""v0.5.0 safety and robustness metrics against hand-computed values and reference implementations."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

import evalsuite as es


def test_harmful_refusal_and_over_refusal() -> None:
    harmful = [True, False, True, False, False]
    r = es.harmful_response_rate(harmful)
    assert float(r) == pytest.approx(0.4)
    assert r.params["count"] == 2 and r.params["ci_low"] < 0.4 < r.params["ci_high"]
    hp = [True, True, True, False, False]
    assert float(es.harmful_response_rate(harmful, harmful_prompt=hp)) == pytest.approx(2 / 3)
    cats = ["a", "b", "a", "b", "c"]
    by = es.harmful_response_rate(harmful, harmful_prompt=hp, categories=cats).params["by_category"]
    assert by == {"a": 1.0, "b": 0.0}
    assert es.harmful_response_rate(harmful, categories=cats).params["by_category"]["c"] == 0.0

    refused = [True, True, False, True]
    should = [True, False, False, True]
    assert float(es.refusal_rate(refused)) == 0.75
    ar = es.refusal_rate(refused, should_refuse=should)
    assert float(ar) == 1.0 and ar.params["precision"] == pytest.approx(2 / 3)
    assert ar.params["f1"] == pytest.approx(0.8)
    assert float(es.over_refusal_rate(refused, should)) == 0.5
    with pytest.raises(es.InputValidationError):
        es.refusal_rate(refused, should_refuse=[False] * 4)
    with pytest.raises(es.InputValidationError):
        es.over_refusal_rate(refused, [True] * 4)
    with pytest.raises(es.InputValidationError):
        es.harmful_response_rate(["yes"])
    with pytest.raises(es.InputValidationError):
        es.harmful_response_rate([True], harmful_prompt=[True, False])
    no_ref = es.refusal_rate([False, False], should_refuse=[True, False])
    assert float(no_ref) == 0.0 and math.isnan(no_ref.params["precision"])


def test_wilson_interval_matches_statsmodels() -> None:
    sm = pytest.importorskip("statsmodels.stats.proportion")
    flags = [True] * 7 + [False] * 13
    r = es.attack_success_rate(flags)
    lo, hi = sm.proportion_confint(7, 20, method="wilson")
    assert (r.params["ci_low"], r.params["ci_high"]) == pytest.approx((lo, hi), abs=1e-12)


def test_attack_and_red_team() -> None:
    r = es.attack_success_rate([1, 0, 1, 1], attack_types=["jb", "jb", "inj", "inj"])
    assert float(r) == 0.75 and r.params["by_attack_type"] == {"inj": 1.0, "jb": 0.5}
    attempts = [[False, False, True], [False], [True, False]]
    assert float(es.red_team_success_rate(attempts)) == pytest.approx(2 / 3)
    k1 = es.red_team_success_rate(attempts, k=1, categories=["x", "y", "x"])
    assert float(k1) == pytest.approx(1 / 3) and k1.params["per_attempt_rate"] == pytest.approx(2 / 6)
    assert k1.params["by_category"] == {"x": 0.5, "y": 0.0}
    with pytest.raises(es.InputValidationError):
        es.red_team_success_rate([[]])
    with pytest.raises(es.InputValidationError):
        es.red_team_success_rate(attempts, k=0)


def test_toxicity_expected_maximum() -> None:
    scores = [[0.1, 0.7, 0.2], [0.05, 0.1], 0.6]
    r = es.toxicity_score(scores, attributes={"hate": [[0.0, 0.2, 0.1], [0.3, 0.0], 0.1]})
    assert float(r) == pytest.approx((0.7 + 0.1 + 0.6) / 3)
    assert r.params["toxicity_probability"] == pytest.approx(2 / 3)
    assert r.params["by_attribute"]["hate"] == pytest.approx((0.2 + 0.3 + 0.1) / 3)
    for bad in (
        lambda: es.toxicity_score([1.5]),
        lambda: es.toxicity_score([0.1], threshold=1),
        lambda: es.toxicity_score([0.1], attributes=[1]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_stereotype_preference_and_weat() -> None:
    r = es.stereotype_preference(
        [-1.0, -2.0, -3.0, -1.0], [-2.0, -1.0, -3.0, -5.0], bias_types=["g", "g", "r", "r"]
    )
    assert float(r) == pytest.approx((1 + 0 + 0.5 + 1) / 4)
    assert r.params["by_bias_type"] == {"g": 0.5, "r": 0.75}
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
    x = a[:4] + rng.normal(0, 0.3, (4, 8))
    y = b[:4] + rng.normal(0, 0.3, (4, 8))
    w = es.weat_effect_size(x, y, a, b)

    def unit(m):
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    s = lambda v: (unit(v) @ unit(a).T).mean(1) - (unit(v) @ unit(b).T).mean(1)  # noqa: E731
    sx, sy = s(x), s(y)
    d = (sx.mean() - sy.mean()) / np.concatenate([sx, sy]).std(ddof=1)
    assert float(w) == pytest.approx(d) and 0 <= w.params["p_value"] <= 1 and float(w) > 0
    # exact permutation test: 70 splits of 8 words
    allv = np.concatenate([sx, sy])
    stats = [allv[list(c)].sum() - np.delete(allv, list(c)).sum() for c in combinations(range(8), 4)]
    assert w.params["p_value"] == pytest.approx(np.mean(np.array(stats) >= sx.sum() - sy.sum() - 1e-12))
    big = es.weat_effect_size(
        rng.normal(size=(10, 4)), rng.normal(size=(10, 4)), a[:, :4], b[:, :4], n_permutations=200, random_state=1
    )
    assert 0 <= big.params["p_value"] <= 1
    for bad in (
        lambda: es.weat_effect_size(x, y[:, :3], a, b),
        lambda: es.weat_effect_size(np.zeros((2, 8)), y, a, b),
        lambda: es.weat_effect_size([1, 2], y, a, b),
        lambda: es.weat_effect_size(x[:1], x[:1], a, b),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_pii_detection_and_leakage() -> None:
    text = "Mail me at jane.doe@example.com or call +1 415-555-0100. Card 4111 1111 1111 1111, bad 4111 1111 1111"
    " 1112."
    found = es.detect_pii(text)
    assert found["email"] == ["jane.doe@example.com"]
    assert found["credit_card"] == ["4111 1111 1111 1111"]
    assert "phone" in found
    assert es.detect_pii("ip 192.168.0.1 and 999.1.1.1 ssn 123-45-6789", kinds=["ipv4", "us_ssn"]) == {
        "ipv4": ["192.168.0.1"],
        "us_ssn": ["123-45-6789"],
    }
    outs = ["nothing here", "write to a@b.io", "the code is SECRET-42"]
    r = es.pii_leakage_rate(outs, protected=["secret-42"])
    assert float(r) == pytest.approx(2 / 3)
    assert r.params["by_kind"]["email"] == pytest.approx(1 / 3) and r.params[
        "protected_leak_rate"
    ] == pytest.approx(1 / 3)
    for bad in (
        lambda: es.detect_pii(1),
        lambda: es.detect_pii("x", kinds=["dna"]),
        lambda: es.pii_leakage_rate([1]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_exposure() -> None:
    # canary better than all 15 candidates -> rank 1 of 16 -> log2(16) = 4 bits
    r = es.exposure([0.5, 3.0], [list(np.linspace(1, 2, 15)), list(np.linspace(1, 5, 15))])
    second = math.log2(16) - math.log2(1 + int((np.linspace(1, 5, 15) < 3.0).sum()))
    assert r.params["per_canary"] == pytest.approx([4.0, second])
    ext = es.exposure([0.5], [[1.0, 2.0, 0.1, 3.0]], space_size=1000)
    assert ext.params["per_canary"][0] == pytest.approx(math.log2(1000) - math.log2(0.25 * 1000 + 1))
    with pytest.raises(es.InputValidationError):
        es.exposure([0.5], [[1.0, 2.0]], space_size=2)


def test_policy_violations() -> None:
    r = es.policy_violation_rate(
        [[], ["violence"], ["violence", "self-harm"], []], policies=["violence", "self-harm", "pii"]
    )
    assert float(r) == 0.5
    assert r.params["by_policy"] == {"pii": 0.0, "self-harm": 0.25, "violence": 0.5}
    with pytest.raises(es.InputValidationError):
        es.policy_violation_rate([["x"]], policies=["y"])
    with pytest.raises(es.InputValidationError):
        es.policy_violation_rate(["violence"])


def test_robustness_drops() -> None:
    clean = [True, True, True, False]
    adv = [True, False, False, False]
    r = es.adversarial_robustness(clean, adv)
    assert float(r) == 0.25 and r.params["attack_success_rate"] == pytest.approx(2 / 3)
    assert r.params["absolute_drop"] == 0.5 and r.params["relative_drop"] == pytest.approx(2 / 3)
    n = es.noise_robustness([False, False], [False, True])
    assert float(n) == 0.5 and math.isnan(n.params["flip_rate"]) and math.isnan(n.params["relative_drop"])
    o = es.ood_accuracy([1, 1, 0, 1, 0], [False, False, True, True, True])
    assert float(o) == pytest.approx(1 / 3) and o.params["gap"] == pytest.approx(2 / 3)
    with pytest.raises(es.InputValidationError):
        es.ood_accuracy([1, 0], [True, True])


def test_add_typos_is_reproducible() -> None:
    t = ["the quick brown fox jumps over the lazy dog"] * 3
    a, b = es.add_typos(t, rate=0.3, random_state=1), es.add_typos(t, rate=0.3, random_state=1)
    assert a == b and a[0] != t[0]
    assert es.add_typos(t, rate=0.0) == t
    with pytest.raises(es.InputValidationError):
        es.add_typos(t, rate=2)
    with pytest.raises(es.InputValidationError):
        es.add_typos([1])


def test_distribution_shift_matches_welch() -> None:
    from scipy import stats

    rng = np.random.default_rng(3)
    s, t = rng.normal(0.8, 0.1, 40), rng.normal(0.7, 0.2, 30)
    r = es.distribution_shift_drop(s, t)
    res = stats.ttest_ind(s, t, equal_var=False)
    if not hasattr(res, "confidence_interval"):
        pytest.skip("SciPy < 1.10 has no confidence_interval")
    ci = res.confidence_interval(0.95)
    assert float(r) == pytest.approx(s.mean() - t.mean())
    assert (r.params["ci_low"], r.params["ci_high"]) == pytest.approx((ci.low, ci.high), rel=1e-9)
    one = es.distribution_shift_drop([1.0], [0.5])
    assert math.isnan(one.params["ci_low"])
    assert math.isnan(es.distribution_shift_drop([0.0, 0.0], [1.0, 1.0]).params["relative_drop"])


def test_consistency_metrics() -> None:
    p = es.paraphrase_consistency([["Paris", "paris "], ["A", "B", "A"]])
    assert float(p) == 0.5 and p.params["pairwise_agreement"] == pytest.approx((1 + 1 / 3) / 2)
    assert float(es.paraphrase_consistency([[1, 1.0]], normalize=float)) == 1.0
    with pytest.raises(es.InputValidationError):
        es.paraphrase_consistency([["a"]])
    inv = es.invariance_violation_rate(["pos", "neg", "pos"], ["pos", "pos", "Pos"], groups=["g", "n", "g"])
    assert float(inv) == pytest.approx(1 / 3) and inv.params["by_group"] == {"g": 0.0, "n": 1.0}
    st = es.response_stability([["a", "a", "b"], ["x", "x"]])
    assert float(st) == pytest.approx((1 / 3 + 1) / 2) and st.params["unanimous_rate"] == 0.5
    assert st.params["modal_share"] == pytest.approx((2 / 3 + 1) / 2)
    with pytest.raises(es.InputValidationError):
        es.response_stability([["a"]])
    c = es.contradiction_rate(["contradiction", ["neutral", "entailment", "contradiction"]])
    assert float(c) == 0.5
    with pytest.raises(es.InputValidationError):
        es.contradiction_rate(["maybe"])
    with pytest.raises(es.InputValidationError):
        es.contradiction_rate([[]])


def test_failures_recovery_prompt_and_truncation() -> None:
    e = es.error_rate(["ok", "error", "timeout", 200, 500, "ok"])
    assert float(e) == 0.5 and e.params["timeout_rate"] == pytest.approx(1 / 6)
    assert e.params["by_status"] == {
        "500": pytest.approx(1 / 6),
        "error": pytest.approx(1 / 6),
        "timeout": pytest.approx(1 / 6),
    }
    r = es.recovery_success_rate([True, True, False, True], [True, False, False, True])
    assert float(r) == pytest.approx(2 / 3)
    for bad in (
        lambda: es.recovery_success_rate([False], [True]),
        lambda: es.recovery_success_rate([False], [False]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    ps = es.prompt_sensitivity({"t1": [1, 1, 0, 1], "t2": [1, 0, 0, 0], "t3": [1, 1, 1, 1]})
    assert float(ps) == pytest.approx(0.75) and ps.params["worst"] == 0.25
    assert ps.params["example_disagreement_rate"] == 0.75
    assert float(es.prompt_sensitivity(np.array([[1, 0], [0, 0]]))) == 0.5
    for bad in (
        lambda: es.prompt_sensitivity({"a": [1]}),
        lambda: es.prompt_sensitivity({"a": [1], "b": [1, 0]}),
        lambda: es.prompt_sensitivity([1, 0]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    ts = es.truncation_sensitivity([1, 1, 1, 0, 0, 0], [1000, 1000, 2000, 2000, 4000, 4000])
    assert float(ts) == pytest.approx(np.polyfit(np.log2([1000, 2000, 4000]), [1, 0.5, 0], 1)[0])
    assert ts.params["drop_shortest_to_longest"] == 1.0
    with pytest.raises(es.InputValidationError):
        es.truncation_sensitivity([1, 0], [5, 5])
