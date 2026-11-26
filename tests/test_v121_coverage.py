"""1.2.1: exercise the report exports, fallbacks and error paths of the 1.2 modules."""

import json

import numpy as np
import pytest

import evalsuite as es


@pytest.fixture
def surv():
    rng = np.random.default_rng(0)
    n = 200
    t = np.round(rng.exponential(5, n), 1) + 0.1
    e = rng.random(n) < 0.7
    risk = -t + rng.normal(0, 2, n)
    return t, e, risk


def test_survival_paths(surv, monkeypatch):
    t, e, risk = surv
    for kw in ({}, {"ipcw": True}, {"ipcw": True, "tau": float(np.quantile(t, 0.8))}):
        r = es.concordance_index(t, e, risk, ci="bootstrap", n_resamples=100, **kw)
        assert r.params["ci_low"] <= r.value <= r.params["ci_high"]
    with pytest.raises(es.InputValidationError, match="ci must be"):
        es.concordance_index(t, e, risk, ci="nope")
    with pytest.raises(es.InputValidationError, match="one value per subject"):
        es.concordance_index(t, e, risk[:-1])
    # pure-Python fallbacks give the same numbers
    from evalsuite import _native

    fast = es.concordance_index(t, e, risk, ci="bootstrap", n_resamples=100)
    monkeypatch.setattr(_native, "core", None)
    slow = es.concordance_index(t, e, risk, ci="bootstrap", n_resamples=100)
    assert slow.value == fast.value and slow.params == fast.params
    km = es.kaplan_meier(t, e)
    assert 0 <= km(float(np.median(t))) <= 1
    times = np.quantile(t, [0.3, 0.6])
    probs = np.clip(np.exp(-np.outer(np.exp(risk / 5), times) / 10), 0, 1)
    auc = es.time_dependent_auc(t, e, risk, times)
    assert np.isfinite(auc.value)
    bs = es.brier_score_survival(t, e, probs, times)
    ibs = es.integrated_brier_score(t, e, probs, times)
    assert np.all(np.isfinite(np.atleast_1d(bs.value))) and np.isfinite(ibs.value)
    with pytest.raises(es.InputValidationError, match="follow-up"):
        es.time_dependent_auc(t, e, risk, [t.max() + 1])


def _fair():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 2, n)
    prob = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    groups = {"sex": rng.choice(["F", "M"], n), "age": rng.choice(["<40", "40+"], n)}
    return y, prob, groups


def test_fairness_exports_and_options():
    y, prob, groups = _fair()
    for kw in ({}, {"intersectional": False}, {"reference": "sex=F & age=<40"}):
        rep = es.fairness_report(y, None, prob, groups=groups, n_resamples=100, **kw)
        assert rep.summary() and rep.to_markdown() and "<table" in rep.to_html()
        json.loads(rep.to_json())
        assert "groups" in rep.to_dict()
    with pytest.raises(es.InputValidationError, match="not one of the groups"):
        es.fairness_report(y, None, prob, groups=groups, reference="zzz", n_resamples=100)
    with pytest.raises(es.InputValidationError):
        es.fairness_report(y, None, None, groups=groups)
    with pytest.raises(es.InputValidationError, match="at least one attribute"):
        es.fairness_report(y, (prob > 0.5).astype(int), groups={})
    two_d = np.column_stack([groups["sex"], groups["age"]])
    assert len(es.fairness_report(y, (prob > 0.5).astype(int), groups=two_d, n_resamples=100).groups) == 4


def _judge_rows():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(120):
        a, b = ("x", "y") if i % 2 else ("y", "z")
        rows.append(
            {
                "id": i,
                "model_a": a,
                "model_b": b,
                "verdict": str(rng.choice(["A", "B", "tie"])),
                "length_a": int(rng.integers(10, 400)),
                "length_b": int(rng.integers(10, 400)),
                "judge": "x",
                "human": str(rng.choice(["A", "B", "tie"])),
                "position": int(i % 2),
            }
        )
    return rows


def test_judge_and_rubric_exports():
    rep = es.judge_report(_judge_rows(), n_resamples=100, random_state=0)
    assert rep.summary() and rep.to_markdown() and "<table" in es.html_report({"Judge": rep})
    json.loads(json.dumps(rep.to_dict()))
    rows = [
        {
            "id": i,
            "model": m,
            "criterion": c,
            "score": float((i * 7 + len(m)) % 5 + 1),
            "rater": r,
            "human": float(i % 5 + 1),
        }
        for i in range(30)
        for m in ("a", "b")
        for c in ("helpful", "correct")
        for r in ("r1", "r2")
    ]
    for scale in ("interval", "ordinal"):
        rub = es.rubric_report(rows, scale=scale)
        assert rub.summary() and rub.to_markdown() and repr(rub)
        json.loads(json.dumps(rub.to_dict()))
    with pytest.raises(es.InputValidationError):
        es.rubric_report([{"id": 1, "model": "a", "criterion": "c"}])


def test_cluster_fallbacks_without_fast_paths():
    rng = np.random.default_rng(4)
    n = 300
    y = rng.integers(0, 3, n)  # multiclass: no curve or mean-loss fast path
    pred = rng.integers(0, 3, n)
    clusters = rng.integers(0, 30, n)
    for method in ("bca", "percentile", "basic"):
        ci = es.bootstrap_ci(es.f1, y, pred, average="macro", clusters=clusters, method=method, n_resamples=200)
        assert ci.low <= ci.high
    yr = rng.normal(size=n)
    ci = es.bootstrap_ci(es.r2, yr, yr + rng.normal(0, 0.5, n), clusters=clusters, n_resamples=200)
    assert ci.low <= ci.estimate <= ci.high
    res = es.evaluate(y, pred, clusters=clusters, baseline_pred=rng.integers(0, 3, n), n_resamples=200)
    assert all("baseline_p_value" in m.params for m in res.metrics.values() if "ci_low" in m.params)


def test_planning_auc_test_and_power():
    plan = es.sample_size_auc(0.8, prevalence=0.3, auc0=0.7)
    assert plan.n > 10 and plan.details["positives"] + plan.details["negatives"] == plan.n
    assert 0.75 < es.power_auc(plan.n, 0.8, 0.7, prevalence=0.3) < 0.9
    with pytest.raises(es.InputValidationError, match="below the expected"):
        es.sample_size_auc(0.7, prevalence=0.3, auc0=0.8)
    with pytest.raises(es.InputValidationError, match="exactly one"):
        es.sample_size_auc(0.8, prevalence=0.3)
    two = es.sample_size_two_proportions(0.7, 0.8, paired=False, two_sided=False)
    assert two.n > 0 and str(two)
    assert 0 < es.power_two_proportions(two.n, 0.7, 0.8, paired=False) < 1


def test_rag_report_continuous_scores_and_exports():
    rows = [
        {
            "question": f"q{i}",
            "answer": "paris is the capital" if i % 3 else "berlin",
            "reference": "paris",
            "retrieved": ["d1", "d2"] if i % 2 else ["d3"],
            "relevant": ["d1"],
            "contexts": ["paris is the capital of france"],
            "grounded": bool(i % 2),
        }
        for i in range(12)
    ]
    rep = es.rag_report(rows, k=2)
    assert rep.biggest_problem() and rep.summary()
    md = rep.to_markdown()
    assert "| Outcome |" in md
    json.loads(rep.to_json())
    with pytest.raises(es.InputValidationError, match="list of document ids"):
        es.rag_report([{**rows[0], "retrieved": 3.5}])


def test_html_report_lists_figures_and_markdown(tmp_path):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    text = "# Heading\nIntro line\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    page = es.html_report(
        [es.evaluate([0, 1, 1, 0], [0, 1, 0, 0]), fig, ax, [text, None]],
        title="T",
        intro="Intro",
        path=str(tmp_path / "r.html"),
        metadata={"data": "x"},
    )
    plt.close(fig)
    assert page.count("<img") == 2 and "<ul>" in page and "<h3>" in page and (tmp_path / "r.html").exists()
    with pytest.raises(es.InputValidationError):
        es.html_report("not sections")


def test_load_benchmark_with_a_stub_datasets_module(monkeypatch):
    import sys
    import types

    data = {
        "openai/gsm8k": [{"question": "1+1?", "answer": "work #### 2"}],
        "cais/mmlu": [{"question": "Q", "choices": ["a", "b"], "answer": 1, "subject": "math"}],
        "allenai/ai2_arc": [
            {"id": "x", "question": "Q", "choices": {"label": ["1", "2"], "text": ["a", "b"]}, "answerKey": "2"}
        ],
        "truthfulqa/truthful_qa": [{"question": "Q", "mc1_targets": {"choices": ["t", "f"], "labels": [1, 0]}}],
        "openai/openai_humaneval": [
            {"task_id": "H/0", "prompt": "def f():", "canonical_solution": "pass", "entry_point": "f", "test": ""}
        ],
    }
    stub = types.ModuleType("datasets")
    stub.load_dataset = lambda path, config, split=None, cache_dir=None: data[path] * 3
    monkeypatch.setitem(sys.modules, "datasets", stub)
    assert es.load_benchmark("gsm8k", limit=2)[0]["reference"] == "2"
    assert es.load_benchmark("mmlu")[0]["category"] == "math"
    assert es.load_benchmark("arc")[0]["reference"] == "B"
    assert es.load_benchmark("truthfulqa")[0]["reference"] == "A"
    assert es.load_benchmark("humaneval")[0]["entry_point"] == "f"
    with pytest.raises(es.InputValidationError, match="At most 26"):
        es.format_multiple_choice("Q", ["x"] * 27)
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(es.OptionalDependencyError):
        es.load_benchmark("gsm8k")
