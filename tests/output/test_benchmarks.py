from __future__ import annotations

import json

import pytest

from evalsuite.benchmarks import run_benchmarks


def test_benchmarks_agree_with_sklearn_and_export(tmp_path) -> None:
    b = run_benchmarks(sizes=(500,), repeat=1, suite="core")
    assert len(b.rows) == 4
    for row in b.rows:
        assert row["evalsuite_ms"] > 0 and row["evalsuite_peak_mb"] >= 0
        assert row["sklearn_ms"] > 0 and row["speedup"] > 0
        assert row["max_abs_diff"] < 1e-9  # same numbers as scikit-learn
    text = b.summary()
    assert "Speed-up" in text and "scikit-learn" in text and repr(b) == text
    assert json.loads(b.to_json())["environment"]["repeat"] == 1
    assert b.to_markdown().startswith("| Case |")
    assert r"\toprule" in b.to_latex() and "<table" in b.to_html() and "<!doctype" in b.to_html(full=True)
    assert b.to_csv().splitlines()[0].startswith("case,n,reference,evalsuite_ms")
    b.to_csv(str(tmp_path / "b.csv"))
    assert len(b.to_dataframe()) == 4


def test_benchmarks_without_sklearn() -> None:
    b = run_benchmarks(sizes=(200,), repeat=1, compare_sklearn=False, suite="core")
    assert all(r["sklearn_ms"] is None and r["speedup"] is None for r in b.rows)
    assert "–" in b.summary()
    with pytest.raises(ValueError, match="repeat"):
        run_benchmarks(sizes=(10,), repeat=0)
    with pytest.raises(ValueError, match="suite"):
        run_benchmarks(sizes=(10,), suite="audio")


def test_v020_benchmarks_agree_with_their_references() -> None:
    pytest.importorskip("statsmodels")
    b = run_benchmarks(sizes=(2_000,), repeat=1, suite="clinical")
    refs = {r["case"]: r["reference"] for r in b.rows}
    assert len(b.rows) == 8
    assert refs["calibration: slope and intercept"] == "statsmodels"
    assert refs["statistics: Welch t-test"] == "SciPy"
    assert refs["decision curve: 99 thresholds"] == "NumPy loop"
    for row in b.rows:
        assert row["reference_ms"] > 0 and row["max_abs_diff"] < 1e-9, row["case"]
    assert "statsmodels" in b.summary().splitlines()[0]
    assert len(run_benchmarks(sizes=(300,), repeat=1, suite="all").rows) == 15
