"""1.2.1: lazy loading of the package, and regression tests for the bugs fixed in the 1.2.1 clean-up."""

import os
import subprocess
import sys

import numpy as np
import pytest

import evalsuite as es

HEAVY = ("numpy", "scipy", "pandas", "matplotlib", "sklearn", "polars", "torch")


def _modules_after(code: str) -> set[str]:
    out = subprocess.run(  # noqa: S603 - fixed interpreter, test-only code
        [sys.executable, "-c", f"import sys; {code}; print(','.join(sorted(sys.modules)))"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "EVALSUITE_NO_UPDATE_CHECK": "1"},
    )
    return set(out.stdout.strip().split(","))


def test_plain_import_loads_no_heavy_library():
    loaded = _modules_after("import evalsuite")
    assert not [m for m in HEAVY if m in loaded]
    assert not [
        m for m in loaded if m.startswith("evalsuite.") and m not in {"evalsuite.version", "evalsuite._update"}
    ]


def test_only_what_is_used_gets_loaded():
    loaded = _modules_after("import evalsuite as es; es.accuracy([1, 0], [1, 1])")
    assert "evalsuite.classification" in loaded
    for unused in ("evalsuite.survival", "evalsuite.fairness", "evalsuite.judging", "evalsuite.plot", "scipy"):
        assert unused not in loaded


def test_every_public_name_resolves_and_is_cached():
    for name in es.__all__:
        assert getattr(es, name) is getattr(es, name)
    assert set(es.__all__) <= set(dir(es))


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'not_a_metric'"):
        es.not_a_metric  # noqa: B018


def test_star_import_and_catalogue_work_lazily():
    ns: dict = {}
    exec("from evalsuite import *", ns)  # noqa: S102
    assert "concordance_index" in ns and "accuracy" in ns
    loaded = _modules_after("import evalsuite as es; print(len(es.list_metrics()), file=sys.stderr)")
    assert "evalsuite.vision" in loaded  # the catalogue imports every metric package it describes
    assert len(es.list_metrics()) > 150


# --- bug clean ---------------------------------------------------------------------------------------------------


def test_rubric_rejects_non_numeric_and_nan_scores():
    rows = [{"id": i, "model": "m", "criterion": "c", "score": 3} for i in range(3)]
    with pytest.raises(es.InputValidationError, match="must be a number"):
        es.rubric_report([*rows, {"id": 9, "model": "m", "criterion": "c", "score": "good"}])
    with pytest.raises(es.InputValidationError, match="must be finite"):
        es.rubric_report([*rows, {"id": 9, "model": "m", "criterion": "c", "score": float("nan")}])


def test_fairness_rejects_missing_groups():
    with pytest.raises(es.InputValidationError, match="missing value"):
        es.fairness_report([1, 0] * 20, [1, 0] * 20, groups=[None, "b"] * 20)


def test_proportion_half_width_must_be_below_one_half():
    with pytest.raises(es.InputValidationError, match=r"below 0\.5"):
        es.sample_size_proportion(0.5, 0.6)


def test_singular_counts_read_naturally():
    judge = es.judge_report([{"model_a": "x", "model_b": "y", "verdict": "a"}], n_resamples=100)
    assert "(1 comparison," in judge.summary()
    rag = es.rag_report([{"question": "q", "answer": "x", "reference": "y", "retrieved": [], "relevant": []}])
    assert "(1 question," in rag.summary()


def test_functions_named_like_their_module_stay_functions():
    loaded = _modules_after(
        "import evalsuite.rag_report, evalsuite.html_report, evalsuite as es; "
        "assert callable(es.rag_report) and callable(es.html_report)"
    )
    assert "evalsuite.rag_report" in loaded


# --- speed paths are exact --------------------------------------------------------------------------------------


def test_native_concordance_is_bit_identical_to_python():
    from evalsuite import survival as sv
    from evalsuite._native import core

    if core is None:
        pytest.skip("compiled core not available")
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 200))
        t = np.round(rng.exponential(5, n), int(rng.integers(0, 3)))
        e = rng.random(n) < rng.random()
        r = np.round(rng.random(n), int(rng.integers(1, 4)))
        w = rng.random(n) + 0.1
        tol = [0.0, 1e-8, 0.05][seed % 3]
        try:
            fast = sv._concordance(t, e, r, w, tol)
        except es.InputValidationError:
            with pytest.raises(es.InputValidationError):
                sv._concordance_python(t, e, r, w, tol)
            continue
        assert fast == sv._concordance_python(t, e, r, w, tol)


def test_batched_concordance_bootstrap_matches_one_by_one(monkeypatch):
    from evalsuite import survival as sv

    rng = np.random.default_rng(3)
    t, e, r = np.round(rng.exponential(5, 300), 1), rng.random(300) < 0.6, np.round(rng.random(300), 2)
    fast = es.concordance_index(t, e, r, ci="bootstrap", n_resamples=150, random_state=1).params
    monkeypatch.setattr(sv, "_bootstrap_concordance", lambda *a: None)
    assert es.concordance_index(t, e, r, ci="bootstrap", n_resamples=150, random_state=1).params == fast


def _slow_cluster(call, members, picks):
    out = []
    for pick in picks:
        try:
            out.append(call(np.concatenate([members[k] for k in pick])))
        except es.EvalSuiteError:
            out.append(np.nan)
    return np.array(out)


@pytest.mark.parametrize("metric", ["accuracy", "f1", "mcc", "roc_auc", "average_precision", "log_loss", "brier"])
def test_cluster_resamples_equal_concatenating_clusters(metric):
    from evalsuite.stats._resolve import MetricCall
    from evalsuite.stats.intervals import _cluster_members, cluster_distribution

    fn = {"brier": es.brier_score}.get(metric) or getattr(es, metric)
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n, g = int(rng.integers(30, 300)), int(rng.integers(2, 25))
        y = rng.integers(0, 2, n)
        score = np.round(rng.random(n), 1)  # ties
        weight = None if seed % 2 else rng.random(n) + 0.1
        prob = metric in ("roc_auc", "average_precision", "log_loss", "brier")
        if prob:
            call = MetricCall(fn, y, None, score, weight, {})
        else:
            call = MetricCall(fn, y, (score > 0.5).astype(int), None, None, {})
        members = _cluster_members(rng.integers(0, g, n), n)
        picks = [rng.integers(0, len(members), len(members)) for _ in range(40)]
        fast, _ = cluster_distribution(call, members, picks)
        slow = _slow_cluster(call, members, picks)
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-14)


def test_judge_bootstrap_unchanged_by_matrix_resampling():
    rng = np.random.default_rng(1)
    recs = [
        {"model_a": a, "model_b": b, "verdict": str(rng.choice(["a", "b", "tie"]))}
        for a, b in [("x", "y"), ("y", "z"), ("x", "z")]
        for _ in range(40)
    ]
    from evalsuite.text.judge import bradley_terry

    rep = es.judge_report(recs, n_resamples=200, random_state=0)
    rows = [(r["model_a"], r["model_b"], {"a": "win", "b": "loss", "tie": "tie"}[r["verdict"]]) for r in recs]
    point = dict(zip(bradley_terry(rows).labels, bradley_terry(rows).value))
    for m in rep.models:
        assert m["bt"] == pytest.approx(point[m["model"]], abs=1e-12)
