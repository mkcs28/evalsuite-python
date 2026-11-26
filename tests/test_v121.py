"""1.2.1: lazy loading of the package, and regression tests for the bugs fixed in the 1.2.1 clean-up."""

import os
import subprocess
import sys

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
