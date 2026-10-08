"""v0.2.0 outputs: calibration report, decision-curve plot, CLI commands."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import evalsuite as es


def _data(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, n)
    p = np.clip(1 / (1 + np.exp(-(2.0 * (y - 0.5) + rng.normal(0, 1.2, n)))), 0.001, 0.999)
    return y, p, (p >= 0.5).astype(int)


def test_calibration_report_values_and_exports(tmp_path) -> None:
    y, p, _ = _data()
    rep = es.calibration_report(y, p)
    assert rep["calibration_slope"] == pytest.approx(float(es.calibration_slope(y, p)))
    assert rep["expected_calibration_error"] == pytest.approx(float(es.expected_calibration_error(y, p)))
    assert rep["hosmer_lemeshow_p"] == pytest.approx(es.hosmer_lemeshow(y, p).p_value)
    assert "Calibration slope" in rep.summary() and "(ideal 1)" in rep.summary()
    assert json.loads(rep.to_json())["values"]["brier_score"] == pytest.approx(float(es.brier_score(y, p)))
    assert rep.to_markdown().startswith("| Measure | Value | Ideal |")
    assert "\\toprule" in rep.to_latex() and rep.to_csv().startswith("measure,value")
    assert rep.to_html().startswith("<!doctype html>") and len(rep.to_dataframe()) == 7
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        assert (tmp_path / f"c.{ext}").exists() is False
        rep.save(tmp_path / f"c.{ext}")
    d = es.diagnostic_report(y, (p > 0.5).astype(int))
    d.save(tmp_path / "d.csv")
    assert (tmp_path / "d.csv").read_text().startswith("measure,estimate")
    assert len(d.to_dataframe()) == 10 and next(iter(d)) == "sensitivity"
    with pytest.raises(KeyError):
        d["nope"]


def test_decision_curve_useful_range_and_plot() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    y, p, _ = _data()
    dc = es.decision_curve(y, p)
    rng = dc.useful_range()
    assert rng and 0 < rng[0][0] < rng[0][1] < 1
    ax = es.plot.decision_curve(y, {"good": p, "noise": np.random.default_rng(0).random(len(y))})
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["good", "noise", "treat all", "treat none"]
    assert ax.get_xlabel() == "Threshold probability"


def _cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "evalsuite", *args],
        capture_output=True,
        encoding="utf-8",
        timeout=300,
        env=env,
    )


def test_cli_output_survives_a_non_utf8_console(tmp_path) -> None:
    """Regression: '−' and 'χ²' crashed the CLI when stdout was cp1252 (Windows pipes)."""
    import os

    f = tmp_path / "d.csv"
    f.write_text("y,p,q\n1,1,0.9\n1,0,0.4\n0,0,0.2\n0,1,0.6\n1,1,0.8\n0,0,0.1\n", encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    out = _cli("diagnostic", str(f), "--y-true", "y", "--y-pred", "p", env=env)
    assert out.returncode == 0, out.stderr
    assert "LR−" in out.stdout


def test_cli_diagnostic_calibration_and_decision_plot(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    y, p, pred = _data()
    f = tmp_path / "preds.csv"
    pd.DataFrame({"label": y, "prob": p, "pred": pred}).to_csv(f, index=False)
    out = _cli("diagnostic", str(f), "--y-true", "label", "--y-pred", "pred", "-f", "json")
    assert out.returncode == 0, out.stderr
    measures = json.loads(out.stdout)["measures"]
    assert measures["sensitivity"]["estimate"] == pytest.approx(float(es.sensitivity(y, pred)))
    out = _cli("calibration", str(f), "--y-true", "label", "--y-prob", "prob")
    assert out.returncode == 0 and "Hosmer" in out.stdout
    out = _cli("calibration", str(f), "--y-true", "label", "--y-prob", "prob", "pred")
    assert out.returncode == 2 and "exactly one --y-prob" in out.stderr
    pytest.importorskip("matplotlib")
    img = tmp_path / "dca.png"
    out = _cli("plot", "decision", str(f), "--y-true", "label", "--y-prob", "prob", "-o", str(img))
    assert out.returncode == 0 and img.stat().st_size > 1000
    out = _cli("metrics", "--category", "clinical")
    assert "clinical.lr_positive" in out.stdout
