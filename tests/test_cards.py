import json

import numpy as np
import pytest

import evalsuite as es
from evalsuite.core.exceptions import InputValidationError, OptionalDependencyError


def _data(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(x * 1.5 - 0.5)))
    y = (rng.random(n) < p).astype(int)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    return y, p, g


def test_tripod_card_contents():
    y, p, g = _data()
    card = es.report_card(y, p, groups=g, group_name="site", n_resamples=200)
    keys = {r["key"] for r in card.rows}
    assert {
        "roc_auc",
        "calibration_intercept",
        "calibration_slope",
        "brier_score",
        "sensitivity",
        "net_benefit",
    } <= keys
    auc = next(r for r in card.rows if r["key"] == "roc_auc")
    assert auc["value"] == pytest.approx(es.roc_auc(y, y_prob=p))
    assert auc["low"] < auc["value"] < auc["high"]
    assert [s["group"] for s in card.subgroups] == ["A", "B"]
    covered = {c["item"]: c["covered"] for c in card.checklist}
    assert covered["23b"] and covered["23g"] and not covered["12"]
    text = card.paragraph()
    assert "c-statistic" in text and "site" in text
    md = card.to_markdown()
    assert "Checklist" in md and "| site |" in md
    assert "\\begin{tabular}" in card.to_latex()
    assert "<table>" in card.to_html()
    json.loads(card.to_json())
    assert card.to_csv().startswith("measure")
    assert "report card" in card.summary()


def test_model_card_sections():
    y, p, _ = _data()
    card = es.report_card(
        y, p, standard="model-card", context={"intended_use": "Triage support."}, n_resamples=100
    )
    md = card.to_markdown()
    assert "## Intended use" in md and "Triage support." in md and "_To be completed._" in md
    assert "## Quantitative analyses" in md


def test_small_data_does_not_crash():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4])
    card = es.report_card(y, p, n_resamples=50)
    assert any(r["key"] == "roc_auc" for r in card.rows)


def test_validation():
    y, p, _ = _data(50)
    with pytest.raises(InputValidationError):
        es.report_card(y, p, standard="nope")
    with pytest.raises(InputValidationError):
        es.report_card(np.zeros(50), p)
    with pytest.raises(InputValidationError):
        es.report_card(y, p, context={"bogus": "x"})


def test_docx(tmp_path):
    y, p, _ = _data(100)
    card = es.report_card(y, p, n_resamples=50)
    try:
        import docx  # noqa: F401
    except ImportError:
        with pytest.raises(OptionalDependencyError):
            card.to_docx(str(tmp_path / "c.docx"))
    else:
        card.to_docx(str(tmp_path / "c.docx"))
        assert (tmp_path / "c.docx").exists()
