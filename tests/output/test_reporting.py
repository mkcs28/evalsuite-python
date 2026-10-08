from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import evalsuite as es


def _sk_report(y, p, w=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skm.classification_report(y, p, sample_weight=w, output_dict=True, zero_division=0)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("weighted", [False, True])
def test_classification_report_matches_sklearn(seed: int, k: int, weighted: bool) -> None:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, 120)
    p = np.where(rng.random(120) < 0.7, y, rng.integers(0, k, 120))
    w = rng.random(120) if weighted else None
    ours = es.classification_report(y, p, sample_weight=w, zero_division=0)
    ref = _sk_report(y, p, w)
    for row in ours.rows:
        r = ref[str(row["label"])]
        assert row["precision"] == pytest.approx(r["precision"])
        assert row["recall"] == pytest.approx(r["recall"])
        assert row["f1"] == pytest.approx(r["f1-score"])
        assert row["support"] == pytest.approx(r["support"])
        assert row["specificity"] == pytest.approx(
            float(es.specificity(y, p, average=None, sample_weight=w, zero_division=0).per_class()[row["label"]])
        )
    assert ours["accuracy"]["f1"] == pytest.approx(ref["accuracy"])
    for name in ("macro avg", "weighted avg"):
        for ours_key, ref_key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            assert ours[name][ours_key] == pytest.approx(ref[name][ref_key])


def test_multilabel_report_has_micro_average() -> None:
    rng = np.random.default_rng(3)
    Y = rng.integers(0, 2, (80, 3))
    P = np.where(rng.random((80, 3)) < 0.8, Y, 1 - Y)
    ours = es.classification_report(Y, P, zero_division=0)
    ref = _sk_report(Y, P)
    assert ours.target_type == "multilabel"
    for key, rkey in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
        assert ours["micro avg"][key] == pytest.approx(ref["micro avg"][rkey])
    with pytest.raises(KeyError, match="No row"):
        ours["accuracy"]


def test_report_exports(tmp_path) -> None:
    rep = es.classification_report(["cat", "dog", "dog", "cat", "bird"], ["cat", "dog", "cat", "cat", "bird"])
    text = rep.summary()
    assert "specificity" in text and "macro avg" in text and repr(rep) == text
    assert rep.to_markdown().startswith("| Class | precision |")
    assert r"\toprule" in rep.to_latex(label="tab:rep") and "Specificity" in rep.to_latex()
    html_doc = rep.to_html(full=True)
    assert html_doc.startswith("<!doctype html>") and 'class="avg"' in html_doc and "<script" not in html_doc
    csv_lines = rep.to_csv().splitlines()
    assert csv_lines[0] == "label,precision,recall,f1,specificity,support"
    assert csv_lines[4].startswith("accuracy,,,")  # undefined cells are empty, not 'nan'
    assert json.loads(rep.to_json())["classes"][0]["label"] == "bird"
    assert isinstance(rep.to_dataframe(), pd.DataFrame)
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        path = rep.save(str(tmp_path / f"report.{ext}"))
        assert (tmp_path / f"report.{ext}").stat().st_size > 50 and path.endswith(ext)
    with pytest.raises(es.InputValidationError, match="Supported extensions"):
        rep.save(str(tmp_path / "report.docx"))


def test_evaluation_and_metric_html_csv(tmp_path) -> None:
    r = es.evaluate([0, 1, 2, 2, 1], [0, 1, 2, 1, 1], metrics=["accuracy", "f1"], average=None)
    lines = r.to_csv().splitlines()
    assert lines[0] == "metric,name,label,value" and lines[1] == "accuracy,Accuracy,,0.8" and len(lines) == 5
    page = r.to_html(full=True)
    assert "F1 per class" in page and "Confusion matrix" in page and '<td class="num">1</td>' in page
    m = es.f1([0, 1], [0, 1])
    assert (
        m.to_csv().splitlines()[1] == "f1,F1,,1.0"
        and "<table" in m.to_html()
        and "<!doctype" in m.to_html(full=True)
    )
    assert "Label" in es.f1([0, 1, 2], [0, 1, 1], average=None, zero_division=0).to_html()
    m.to_csv(str(tmp_path / "m.csv"))
    r.to_csv(str(tmp_path / "r.csv"))
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        r.save(str(tmp_path / f"eval.{ext}"))
    assert (tmp_path / "eval.html").read_text(encoding="utf-8").startswith("<!doctype html>")


def test_html_is_escaped() -> None:
    r = es.evaluate(["<b>", "x&y"], ["<b>", "x&y"], metrics=["accuracy"])
    page = r.to_html()
    assert "&lt;b&gt;" in page and "x&amp;y" in page and "<b>" not in page


def test_comparison_html_csv_save(tmp_path) -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 150)
    preds = {"a": np.where(rng.random(150) < 0.85, y, 1 - y), "b": np.where(rng.random(150) < 0.7, y, 1 - y)}
    c = es.compare(y, preds, metrics=["accuracy", "f1"], random_state=0, n_resamples=200)
    assert c.to_csv().splitlines()[0] == "model,metric,estimate,low,high"
    assert c.to_csv(which="tests").splitlines()[0].startswith("metric,model_a,model_b,difference")
    with pytest.raises(es.InputValidationError):
        c.to_csv(which="nope")
    page = c.to_html(full=True)
    assert 'class="num best"' in page and "Pairwise tests" in page
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        c.save(str(tmp_path / f"cmp.{ext}"))
    c.to_csv(str(tmp_path / "cmp2.csv"))


def test_calibration_matches_sklearn_and_ece() -> None:
    from sklearn.calibration import calibration_curve as sk_cal

    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 600)
    p = 1 / (1 + np.exp(-(2 * (y - 0.5) + rng.normal(0, 1, 600))))
    for strategy in ("uniform", "quantile"):
        for bins in (5, 10, 15):
            ours = es.calibration_curve(y, p, n_bins=bins, strategy=strategy)
            ref = sk_cal(y, p, n_bins=bins, strategy=strategy)
            np.testing.assert_allclose(ours[0], ref[0])
            np.testing.assert_allclose(ours[1], ref[1])
    true_f, mean_p, w = es.calibration_curve(y, p)
    assert float(es.expected_calibration_error(y, p)) == pytest.approx(
        np.sum(w * np.abs(true_f - mean_p)) / w.sum()
    )
    assert float(es.expected_calibration_error([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])) == 0.0
    with pytest.raises(es.UnsupportedTaskError):
        es.calibration_curve([0, 1, 2], [[0.2, 0.3, 0.5]] * 3)
    with pytest.raises(es.InputValidationError, match="n_bins"):
        es.calibration_curve(y, p, n_bins=0)
    with pytest.raises(es.InputValidationError, match="strategy"):
        es.calibration_curve(y, p, strategy="kmeans")  # type: ignore[arg-type]
    assert "classification.expected_calibration_error" in es.list_metrics()
