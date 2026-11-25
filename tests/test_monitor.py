"""Monitoring mode: sliding-window metrics, drift and alerts."""

import numpy as np
import pytest

import evalsuite as es


def test_window_metrics_and_alerts():
    fired = []
    mon = es.Monitor(window=200, alerts={"accuracy": ("<", 0.8)}, on_alert=fired.append, min_n=50)
    rng = np.random.default_rng(0)
    for _ in range(5):  # good model
        y = rng.integers(0, 2, 50)
        assert mon.update(y, y_pred=np.where(rng.random(50) < 0.95, y, 1 - y)) == []
    for _ in range(6):  # degraded model: the window fills with errors
        y = rng.integers(0, 2, 50)
        alerts = mon.update(y, y_pred=np.where(rng.random(50) < 0.6, y, 1 - y))
    assert alerts and alerts[0].name == "accuracy" and alerts[0].n == 200
    assert fired and "accuracy" in repr(fired[-1])
    snap = mon.snapshot()
    assert snap.n_samples == 200 and "ci_low" in snap["accuracy"].params
    assert len(mon.history) == 11 and mon.n_seen == 550


def test_drift_without_labels():
    rng = np.random.default_rng(1)
    ref = rng.beta(2, 5, 5000)
    mon = es.Monitor(reference_scores=ref, alerts={"psi": (">", 0.25), "ks_p": ("<", 0.01)}, window=500)
    assert mon.observe(rng.beta(2, 5, 500)) == []
    shifted = mon.observe(rng.beta(5, 2, 500))
    assert {a.name for a in shifted} == {"psi", "ks_p"}
    assert mon.drift()["psi"] > 0.25


def test_psi_and_ks():
    rng = np.random.default_rng(2)
    a = rng.normal(size=10000)
    assert es.population_stability_index(a, rng.normal(size=10000)) < 0.02
    assert es.population_stability_index(a, rng.normal(1, 1, 10000)) > 0.25
    stat, p = es.ks_drift(a, a + 0.5)
    assert stat > 0.15 and p < 1e-10
    with pytest.raises(es.InputValidationError):
        es.population_stability_index([1.0], [1.0])


def test_monitor_errors():
    with pytest.raises(es.InputValidationError):
        es.Monitor(alerts={"accuracy": ("==", 0.5)})
    mon = es.Monitor()
    with pytest.raises(es.InputValidationError):
        mon.snapshot()
    mon.update([0, 1], y_pred=[0, 1])
    with pytest.raises(es.InputValidationError):
        mon.update([0, 1], y_prob=[0.2, 0.9])
    with pytest.raises(es.InputValidationError):
        mon.update([0, 1], y_pred=[0])
