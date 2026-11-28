"""1.3.5: bug clean and bounded-memory resampling."""

import numpy as np
import pytest

import evalsuite as es

E = es.InputValidationError


def test_ranking_counts_each_document_once_and_rejects_nan_scores():
    rep = es.ranking_report({"q": {"d": 1}}, {"a": {"q": ["d", "d", "d"]}}, k=3, n_resamples=20, n_permutations=20)
    pq = rep.per_query["a"]["q"]
    assert pq["ap"] == 1.0 and pq["recall"] == 1.0 and pq["precision"] == pytest.approx(1 / 3)
    with pytest.raises(E, match="finite"):
        es.ranking_report({"q": {"d": 1}}, {"a": {"q": {"d": np.nan}}})


def test_temperature_scaling_maps_string_labels_to_columns():
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.2, 0.1, 0.7]])
    labels = np.array(["x", "z", "x", "z"])
    with pytest.raises(E, match="classes="):
        es.recalibrate(labels, probs, method="temperature")
    by_name = es.recalibrate(labels, probs, method="temperature", classes=["x", "y", "z"])
    by_index = es.recalibrate(np.array([0, 2, 0, 2]), probs, method="temperature")
    assert by_name.params["temperature"] == pytest.approx(by_index.params["temperature"])
    with pytest.raises(E, match="not among"):
        es.recalibrate(np.array(["x", "w", "x", "z"]), probs, method="temperature", classes=["x", "y", "z"])
    with pytest.raises(E, match="entries"):
        es.recalibrate(labels, probs, method="temperature", classes=["x", "z"])


def test_undefined_multilabel_metrics_are_left_out():
    rng = np.random.default_rng(0)
    rep = es.multilabel_report(np.zeros((6, 3), int), y_score=rng.random((6, 3)), n_resamples=50)
    assert "macro_roc_auc" not in rep.metrics and rep.settings["undefined"] == ["macro_roc_auc", "micro_roc_auc"]
    assert "Undefined" in rep.summary()


def test_degenerate_inputs_give_clear_errors():
    with pytest.raises(E, match="overflows"):
        es.diebold_mariano([1e300, -1e300, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 2.0])
    with pytest.raises(E, match="do not differ"):
        es.icc([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])


def test_every_report_has_the_standard_exports():
    import inspect

    for name in es.__all__:
        obj = getattr(es, name)
        if inspect.isclass(obj) and hasattr(obj, "to_markdown") and name != "MetricResult":
            for method in ("summary", "to_markdown", "to_json", "to_dict", "to_html"):
                assert hasattr(obj, method), f"{name}.{method}"
    rep = es.bayes_correlated_ttest([0.8, 0.82, 0.81], [0.79, 0.8, 0.8])
    assert "<table>" in rep.to_html()


def test_chunked_resampling_equals_drawing_everything_at_once(monkeypatch):
    from evalsuite.stats import intervals

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.where(rng.random(400) < 0.8, y, 1 - y)
    prob = rng.random(400)
    whole = es.bootstrap_ci(es.accuracy, y, p, stratify=False, random_state=3)
    auc = es.bootstrap_ci(es.roc_auc, y, y_prob=prob, stratify=False, random_state=3)
    test = es.paired_bootstrap_test(es.accuracy, y, p, 1 - p, stratify=False, random_state=3)
    monkeypatch.setattr(intervals, "_CHUNK_ELEMENTS", 1000)  # forces 2–3 resamples per chunk
    again = es.bootstrap_ci(es.accuracy, y, p, stratify=False, random_state=3)
    again_auc = es.bootstrap_ci(es.roc_auc, y, y_prob=prob, stratify=False, random_state=3)
    assert (again.low, again.high) == (whole.low, whole.high)
    assert (again_auc.low, again_auc.high) == (auc.low, auc.high)
    assert str(es.paired_bootstrap_test(es.accuracy, y, p, 1 - p, stratify=False, random_state=3)) == str(test)
