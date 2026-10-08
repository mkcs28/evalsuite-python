from __future__ import annotations

import subprocess
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evalsuite as es


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    a = 1 / (1 + np.exp(-(2.5 * (y - 0.5) + rng.normal(0, 1, 300))))
    b = 1 / (1 + np.exp(-(1.0 * (y - 0.5) + rng.normal(0, 1, 300))))
    return y, a, b


def _legend(ax) -> list[str]:
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_roc_and_pr_report_evalsuite_values(data) -> None:
    y, a, b = data
    ax = es.plot.roc(y, {"A": a, "B": b})
    assert _legend(ax)[:2] == [
        f"A: AUC = {float(es.roc_auc(y, a)):.3f}",
        f"B: AUC = {float(es.roc_auc(y, b)):.3f}",
    ]
    fpr, _, _ = es.roc_curve(y, a)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), fpr)
    assert ax.lines[0].get_linestyle() != ax.lines[1].get_linestyle()  # distinguishable without colour
    ax = es.plot.pr(y, a, label="model")
    assert _legend(ax)[0] == f"model: AP = {float(es.average_precision(y, a)):.3f}"
    assert _legend(ax)[1] == f"chance ({y.mean():.2f})"
    single = es.plot.roc(y, a, chance=False)
    assert _legend(single) == [f"AUC = {float(es.roc_auc(y, a)):.3f}"]


def test_calibration_confusion_residuals(data) -> None:
    y, a, _ = data
    ax = es.plot.calibration(y, {"A": a}, n_bins=8)
    assert f"ECE = {float(es.expected_calibration_error(y, a, n_bins=8)):.3f}" in _legend(ax)[1]
    rng = np.random.default_rng(2)
    yk = rng.integers(0, 3, 100)
    pk = np.where(rng.random(100) < 0.7, yk, rng.integers(0, 3, 100))
    ax = es.plot.confusion_matrix(yk, pk)
    texts = [t.get_text() for t in ax.texts]
    assert texts == [str(int(v)) for v in es.confusion_matrix(yk, pk).ravel()]
    ax = es.plot.confusion_matrix(["a", "b", "a"], ["a", "a", "a"], normalize="true", colorbar=False)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"] and "0.50" not in [
        t.get_text() for t in ax.texts
    ]
    yr = rng.normal(10, 2, 80)
    pr = yr + rng.normal(0, 1, 80)
    assert "RMSE" in es.plot.residuals(yr, pr).get_title()
    assert _legend(es.plot.residuals(yr, pr, kind="predicted")) == ["y = x"]
    with pytest.raises(es.InputValidationError, match="kind"):
        es.plot.residuals(yr, pr, kind="qq")
    with pytest.raises(es.InputValidationError, match="single-output"):
        es.plot.residuals(np.c_[yr, yr], np.c_[pr, pr])


def test_comparison_forest_plot(data) -> None:
    y, a, b = data
    c = es.compare(
        y,
        {"A": (a > 0.5).astype(int), "B": (b > 0.5).astype(int)},
        probabilities={"A": a, "B": b},
        random_state=0,
        n_resamples=200,
    )
    ax = es.plot.comparison(c, metrics=["accuracy", "roc_auc"])
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "accuracy · A",
        "accuracy · B",
        "roc_auc · A",
        "roc_auc · B",
    ]
    with pytest.raises(es.InputValidationError, match="not compared"):
        es.plot.comparison(c, metrics=["auc"])


def test_draws_on_given_axes_and_validates(data) -> None:
    y, a, _ = data
    _, ax = plt.subplots()
    assert es.plot.roc(y, a, ax=ax) is ax
    with pytest.raises(es.InputValidationError, match="at least one model"):
        es.plot.roc(y, {})


def test_matplotlib_is_optional(monkeypatch) -> None:
    out = subprocess.run(
        [sys.executable, "-c", "import sys, evalsuite; print('matplotlib' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"
    import builtins

    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(es.OptionalDependencyError, match=r"evalsuite-python\[plot\]"):
        es.plot.roc([0, 1], [0.2, 0.8])


def test_calibration_legend_below_by_default_and_configurable():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.3, 0.7, 0.9, 0.2, 0.6, 0.4])
    ax = es.plot.calibration(y, p)
    anchor = ax.get_legend().get_bbox_to_anchor().transformed(ax.transAxes.inverted())
    assert anchor.y0 < 0  # outside, under the axes
    ax2 = es.plot.calibration(y, p, legend_loc="upper left")
    assert ax2.get_legend()._loc == 2
