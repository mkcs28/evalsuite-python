"""1.2: planning, survival, clustered intervals, fairness, judge / rubric / RAG reports, HTML report, benchmarks."""

import json
import math
import sys

import numpy as np
import pytest

import evalsuite as es
from evalsuite.core.exceptions import InputValidationError, OptionalDependencyError

# ------------------------------------------------------------------------------------------------ planning


def test_sample_size_textbook_values():
    assert es.sample_size_proportion(0.5, 0.05).n == 385
    assert es.sample_size_two_proportions(0.7, 0.8, paired=False).n == 294
    assert es.sample_size_mean_difference(0.5, 1.0).n == 34
    assert es.sample_size_mean_difference(0.5, 1.0, paired=False).n == 64
    s = es.sample_size_auc(0.8, prevalence=0.3, half_width=0.05)
    assert s.details["positives"] + s.details["negatives"] == s.n


def test_power_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.power")
    assert es.power_mean_difference(34, 0.5, 1.0) == pytest.approx(sm.TTestPower().power(0.5, 34, 0.05), rel=1e-7)
    assert es.power_mean_difference(64, 0.5, 1.0, paired=False) == pytest.approx(
        sm.TTestIndPower().power(0.5, 64, 0.05), rel=1e-7
    )
    prop = pytest.importorskip("statsmodels.stats.proportion")
    ours = es.power_two_proportions(294, 0.7, 0.8, paired=False)
    theirs = prop.power_proportions_2indep(0.1, 0.7, 294, alpha=0.05, return_results=False)
    assert ours == pytest.approx(theirs, abs=1e-5)


def test_planning_round_trip_and_errors():
    s = es.sample_size_two_proportions(0.75, 0.8, discordance=0.15)
    assert es.power_two_proportions(s.n, 0.75, 0.8, discordance=0.15) >= 0.8
    assert es.power_two_proportions(s.n - 5, 0.75, 0.8, discordance=0.15) < 0.8
    assert int(s) == s.n and "McNemar" in s.summary()
    with pytest.raises(InputValidationError):
        es.sample_size_two_proportions(0.8, 0.8)
    with pytest.raises(InputValidationError):
        es.sample_size_two_proportions(0.7, 0.8, discordance=0.05)
    with pytest.raises(InputValidationError):
        es.sample_size_auc(0.8, prevalence=0.3)


# ------------------------------------------------------------------------------------------------ survival


def _surv_data(seed=0, n=150):
    rng = np.random.default_rng(seed)
    t = np.round(rng.exponential(10, n), 1) + 0.1
    e = rng.random(n) < 0.7
    r = np.round(rng.normal(size=n) - 0.05 * t, 2)
    return t, e, r


def test_concordance_basics():
    t = np.array([1.0, 2, 3, 4, 5])
    e = np.ones(5, bool)
    assert es.concordance_index(t, e, -t).value == 1.0
    assert es.concordance_index(t, e, t).value == 0.0
    assert es.concordance_index(t, e, np.zeros(5)).value == 0.5
    with pytest.raises(InputValidationError):
        es.concordance_index(t, np.zeros(5, bool), t)


def test_survival_matches_scikit_survival():
    sm = pytest.importorskip("sksurv.metrics")
    from sksurv.util import Surv

    for seed in range(5):
        t, e, r = _surv_data(seed)
        y = Surv.from_arrays(e, t)
        assert es.concordance_index(t, e, r).value == pytest.approx(
            sm.concordance_index_censored(e, t, r)[0], abs=1e-12
        )
        assert es.concordance_index(t, e, r, ipcw=True).value == pytest.approx(
            sm.concordance_index_ipcw(y, y, r)[0], abs=1e-12
        )
        ts = np.quantile(t[e], [0.25, 0.5, 0.75])
        auc = es.time_dependent_auc(t, e, r, ts)
        ref_auc, ref_mean = sm.cumulative_dynamic_auc(y, y, r, ts)
        assert auc.value == pytest.approx(ref_mean, abs=1e-12)
        assert np.allclose(auc.params["auc"], ref_auc, atol=1e-12)
        sp = np.clip(np.exp(-np.outer(np.exp(0.3 * r), ts) / 20), 0, 1)
        assert np.allclose(es.brier_score_survival(t, e, sp, ts).params["brier"], sm.brier_score(y, y, sp, ts)[1])
        assert es.integrated_brier_score(t, e, sp, ts).value == pytest.approx(
            sm.integrated_brier_score(y, y, sp, ts), abs=1e-12
        )


def test_kaplan_meier_and_bootstrap_ci():
    km = es.kaplan_meier([1, 2, 3, 4], [1, 0, 1, 1])
    assert km(0.5) == 1.0 and km(1) == pytest.approx(0.75)
    t, e, r = _surv_data(1)
    c = es.concordance_index(t, e, r, ci="bootstrap", n_resamples=200)
    assert c.params["ci_low"] <= c.value <= c.params["ci_high"]


# ------------------------------------------------------------------------------------------------ clusters


def _clustered(seed=0, g=40, m=8):
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-(1 - rng.normal(0, 1.2, g))))
    corr = (rng.random((g, m)) < p[:, None]).ravel().astype(int)
    return np.ones(g * m, int), corr, np.repeat(np.arange(g), m)


def test_cluster_bootstrap_is_wider_than_iid():
    yt, yp, cl = _clustered()
    iid = es.bootstrap_ci("accuracy", yt, yp, n_resamples=400, random_state=0)
    clu = es.bootstrap_ci("accuracy", yt, yp, clusters=cl, n_resamples=400, random_state=0)
    assert clu.params["clusters"] == 40
    assert (clu.high - clu.low) > 1.3 * (iid.high - iid.low)
    with pytest.raises(InputValidationError):
        es.bootstrap_ci("accuracy", yt, yp, clusters=cl[:-1])


def test_evaluate_with_clusters():
    yt, yp, cl = _clustered()
    yt = yt.copy()
    yt[::5] = 0
    r = es.evaluate(yt, yp, clusters=cl, n_resamples=200, baseline_pred=1 - yp)
    acc = r["accuracy"].params
    assert acc["ci_method"].startswith("cluster bootstrap")
    assert acc["baseline_test"] == "paired cluster bootstrap"
    assert r.metadata["clusters"] == 40


# ------------------------------------------------------------------------------------------------ fairness


def _fair_data(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["F", "M"], n)
    age = rng.choice(["young", "old"], n)
    y = (rng.random(n) < 0.35).astype(int)
    prob = np.clip(0.3 * y + 0.35 + rng.normal(0, 0.2, n) + 0.05 * (sex == "M"), 0, 1)
    return y, (prob >= 0.5).astype(int), prob, sex, age


def test_fairness_gaps_match_fairlearn():
    fl = pytest.importorskip("fairlearn.metrics")
    y, pred, prob, sex, _ = _fair_data()
    r = es.fairness_report(y, pred, prob, groups={"sex": sex}, n_resamples=100)
    g = {x["name"]: x["value"] for x in r.gaps}
    assert g["Demographic parity difference"] == pytest.approx(
        fl.demographic_parity_difference(y, pred, sensitive_features=sex)
    )
    assert g["Equalized odds difference"] == pytest.approx(
        fl.equalized_odds_difference(y, pred, sensitive_features=sex)
    )


def test_fairness_report_structure():
    y, pred, prob, sex, age = _fair_data()
    r = es.fairness_report(y, pred, prob, groups={"sex": sex, "age": age}, n_resamples=100)
    assert len(r.groups) == 4 and all(" & " in g["group"] for g in r.groups)
    assert all(t["p_adjusted"] >= t["p_value"] for t in r.tests)
    assert "Gaps" in r.summary() and "| Gap |" in r.to_markdown()
    json.loads(json.dumps(r.to_dict()))
    sep = es.fairness_report(y, pred, groups={"sex": sex, "age": age}, intersectional=False, n_resamples=50)
    assert len(sep.groups) == 4 and any(g["group"].startswith("age=") for g in sep.groups)
    with pytest.raises(InputValidationError):
        es.fairness_report(y, pred, groups=np.zeros(len(y)))


# ------------------------------------------------------------------------------------------------ judges


def _judgements(seed=0, n=300):
    rng = np.random.default_rng(seed)
    strength = {"m1": 1.0, "m2": 0.5, "m3": 0.0}
    rows = []
    for _ in range(n):
        a, b = rng.choice(list(strength), 2, replace=False)
        la, lb = (int(x) for x in rng.integers(50, 300, 2))
        p = 1 / (1 + np.exp(-(strength[a] - strength[b]))) + 0.2 * np.sign(la - lb)
        v = "A" if rng.random() < min(max(p, 0.02), 0.98) else "B"
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "verdict": v,
                "verdict_swapped": v if rng.random() < 0.85 else "tie",
                "length_a": la,
                "length_b": lb,
                "human": "A" if rng.random() < 0.5 else "B",
            }
        )
    return rows


def test_judge_report_detects_length_bias():
    r = es.judge_report(_judgements(), n_resamples=100)
    assert r.models[0]["model"] == "m1"
    assert r.checks["length"]["flag"] and r.checks["length"]["value"] > 0.6
    assert 0 < r.checks["position"]["value"] < 1
    assert "kappa" in r.checks["human"]
    assert "Length bias" in r.summary() and "| Model |" in r.to_markdown()
    json.loads(r.to_json())
    with pytest.raises(InputValidationError):
        es.judge_report([{"model_a": "x", "model_b": "y", "verdict": "maybe"}])


def test_rubric_report():
    rows = [
        {"id": i, "model": m, "criterion": c, "score": s + (i % 3) * 0.1, "rater": rt, "human": s}
        for i in range(10)
        for m, s in (("a", 4), ("b", 2))
        for c in ("helpful", "correct")
        for rt in ("r1", "r2")
    ]
    r = es.rubric_report(rows)
    means = {(x["model"], x["criterion"]): x["mean"] for x in r.scores}
    assert means[("a", "helpful")] > means[("b", "helpful")]
    assert r.reliability["raters"] == 2 and r.human["n"] == len(rows)


# ------------------------------------------------------------------------------------------------ RAG


def test_rag_report_attribution():
    rows = [
        {
            "id": 1,
            "retrieved": ["d1", "x"],
            "relevant": ["d1"],
            "answer": "Paris",
            "reference": "paris",
            "grounded": True,
        },
        {"id": 2, "retrieved": ["x", "y"], "relevant": ["d2"], "answer": "Rome", "reference": "Oslo"},
        {"id": 3, "retrieved": ["d3"], "relevant": ["d3"], "answer": "Lima", "reference": "Quito"},
        {
            "id": 4,
            "retrieved": ["x"],
            "relevant": ["d4"],
            "answer": "Bern",
            "reference": "bern",
            "grounded": False,
        },
    ]
    r = es.rag_report(rows, k=2)
    assert r.examples["correct"] == (1,)
    assert r.examples["retrieval_failure"] == (2,)
    assert r.examples["generation_failure"] == (3,)
    assert r.examples["unsupported_correct"] == (4,)
    assert r.metrics["hit_rate"]["value"] == pytest.approx(0.5)
    assert r.biggest_problem() in ("retrieval_failure", "generation_failure", "unsupported_correct")
    json.loads(r.to_json())
    assert es.rag_report([{"answer": "a", "reference": "b"}]).examples["wrong_unknown_cause"] == (0,)


# ------------------------------------------------------------------------------------------------ HTML report


def test_html_report(tmp_path):
    y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    r = es.evaluate(y, [0, 1, 0, 0, 1, 1, 1, 1])
    path = tmp_path / "r.html"
    page = es.html_report(
        {"Results": r, "Plan": es.sample_size_proportion(0.8, 0.05), "Notes <b>": "x <script>alert(1)</script>"},
        title="T <t>",
        path=str(path),
        metadata={"Dataset": "demo"},
    )
    assert path.read_text(encoding="utf-8") == page
    assert "<script>" not in page and "&lt;script&gt;" in page and "95% CI" in page
    assert page.count("<section") == 3 and "Dataset: demo" in page
    plt = pytest.importorskip("matplotlib.pyplot")
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    assert "data:image/png;base64" in es.html_report([fig])
    with pytest.raises(InputValidationError):
        es.html_report({})


# ------------------------------------------------------------------------------------------------ benchmarks


def test_benchmark_presets_in_leaderboard():
    from evalsuite.llm_benchmarks import _gsm8k, _mmlu

    refs = [
        _mmlu(i, {"question": f"Q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4, "subject": "law"})
        for i in range(30)
    ]
    runs = {
        "good": [{"id": r["id"], "prediction": "The answer is " + r["reference"]} for r in refs],
        "bad": [{"id": r["id"], "prediction": "B"} for r in refs],
    }
    b = es.leaderboard(runs, benchmark="mmlu", references=refs)
    assert b.settings["scorer"] == "choice" and b.rows[0]["model"] == "good" and b.rows[0]["score"] == 1.0
    he = {
        "a": [{"id": i, "passed": i % 3 != 0} for i in range(30)],
        "b": [{"id": i, "n_samples": 5, "n_correct": i % 6} for i in range(30)],
    }
    assert es.leaderboard(he, benchmark="humaneval").settings["scorer"] == "pass@1"
    g = [_gsm8k(i, {"question": "q", "answer": f"x #### {i * 3}"}) for i in range(10)]
    gb = es.leaderboard(
        {
            "x": [{"id": i, "prediction": f"#### {i * 3}"} for i in range(10)],
            "y": [{"id": i, "prediction": "0"} for i in range(10)],
        },
        benchmark="gsm8k",
        references=g,
    )
    assert gb.rows[0]["model"] == "x"
    with pytest.raises(InputValidationError):
        es.leaderboard(runs, benchmark="nope", references=refs)
    assert "A. one" in es.format_multiple_choice("Q?", ["one", "two"])


def test_load_benchmark_needs_datasets(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(OptionalDependencyError):
        es.load_benchmark("gsm8k")
    assert set(es.BENCHMARKS) >= {"gsm8k", "mmlu", "arc", "truthfulqa", "humaneval"}
    assert math.isfinite(1.0)
