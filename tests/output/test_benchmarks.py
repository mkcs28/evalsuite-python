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
    assert len(run_benchmarks(sizes=(300,), repeat=1, suite="all").rows) == 27


def test_overall_summary_and_alphabetical_rows() -> None:
    b = run_benchmarks(sizes=(300,), repeat=1, suite="all")
    overall = b.overall()
    assert [o["group"] for o in overall] == [
        "Classification and regression",
        "Clinical, calibration and statistics",
        "LLM evaluation",
        "LLM systems",
        "Segmentation and object detection",
        "Overall",
    ]
    total = overall[-1]
    assert total["rows"] == 27 == sum(o["rows"] for o in overall[:-1])
    assert total["matching"] == total["compared"]  # every compared case agrees with its reference
    if total["compared"]:
        assert total["min_speedup"] <= total["geomean_speedup"] <= total["max_speedup"]
    cases = [r["case"].lower() for r in b.sorted_rows()]
    assert cases == sorted(cases)
    text = b.summary()
    assert text.index("Overall") < text.index("Case")  # the overall table comes first
    assert b.overall_markdown().startswith("| Suite |")
    assert json.loads(b.to_json())["overall"][-1]["group"] == "Overall"


def test_llm_benchmarks_agree_with_their_references() -> None:
    b = run_benchmarks(sizes=(2000,), repeat=1, suite="llm")
    assert len(b.rows) == 6
    refs = {r["case"].split(" (")[0]: r["reference"] for r in b.rows}
    assert refs["text: corpus BLEU and chrF"] == "sacreBLEU"
    assert refs["structured: JSON Schema compliance"] == "jsonschema"
    for row in b.rows:
        if row["reference_ms"] is not None:
            assert row["max_abs_diff"] < 1e-9, row["case"]


def test_every_metric_has_a_benchmark_row_that_agrees_with_its_reference() -> None:
    import evalsuite as es
    import evalsuite.stats as st

    b = run_benchmarks(sizes=(3_000,), repeat=1, suite="metrics")
    cases = {r["case"] for r in b.rows}
    stats_fns = {f"statistics.{n}" for n in st.__all__ if n[0].islower()}
    assert set(es.list_metrics()) | stats_fns == cases
    for row in b.rows:
        assert row["evalsuite_ms"] > 0
        if row["max_abs_diff"] is not None:
            assert row["max_abs_diff"] < 1e-9, row["case"]
    groups = {g["group"] for g in b.overall()}
    assert {"Classification and regression", "LLM evaluation", "Segmentation and object detection"} <= groups
    assert "Other" not in groups


def test_metrics_suite_without_references() -> None:
    b = run_benchmarks(sizes=(1_000,), repeat=1, suite="metrics", compare_sklearn=False)
    assert all(r["reference"] is None for r in b.rows)


def test_cases_needing_a_missing_optional_dependency_are_skipped(monkeypatch) -> None:
    import evalsuite as es
    from evalsuite import benchmarks

    def missing(*_a, **_k):
        raise es.OptionalDependencyError("nltk", "llm", "METEOR")

    monkeypatch.setattr(es, "meteor", missing)
    b = benchmarks.run_benchmarks(sizes=(500,), repeat=1, suite="llm")
    assert not any("METEOR" in r["case"] for r in b.rows) and b.rows


def test_llmsys_benchmarks_agree_with_their_references() -> None:
    b = run_benchmarks(sizes=(2000,), repeat=1, suite="llmsys")
    assert len(b.rows) == 6
    for row in b.rows:
        if row["max_abs_diff"] is not None:
            assert row["max_abs_diff"] < 1e-9, row["case"]
