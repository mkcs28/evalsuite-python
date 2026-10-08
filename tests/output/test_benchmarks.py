from __future__ import annotations

import json

import pytest

from evalsuite.benchmarks import run_benchmarks


def test_benchmarks_agree_with_sklearn_and_export(tmp_path) -> None:
    b = run_benchmarks(sizes=(500,), repeat=1)
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
    assert b.to_csv().splitlines()[0].startswith("case,n,evalsuite_ms")
    b.to_csv(str(tmp_path / "b.csv"))
    assert len(b.to_dataframe()) == 4


def test_benchmarks_without_sklearn() -> None:
    b = run_benchmarks(sizes=(200,), repeat=1, compare_sklearn=False)
    assert all(r["sklearn_ms"] is None and r["speedup"] is None for r in b.rows)
    assert "–" in b.summary()
    with pytest.raises(ValueError, match="repeat"):
        run_benchmarks(sizes=(10,), repeat=0)
