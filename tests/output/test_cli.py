from __future__ import annotations

import json
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import evalsuite as es
from evalsuite.cli.main import main


@pytest.fixture
def csv_file(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    pa = np.where(rng.random(120) < 0.85, y, 1 - y)
    pb = np.where(rng.random(120) < 0.7, y, 1 - y)
    qa = 1 / (1 + np.exp(-(2.5 * (y - 0.5) + rng.normal(0, 1, 120))))
    qb = rng.random(120)
    yr = rng.normal(10, 2, 120)
    df = pd.DataFrame(
        {
            "label": y,
            "pred": pa,
            "pred_b": pb,
            "prob": qa,
            "prob_b": qb,
            "w": rng.random(120),
            "value": yr,
            "estimate": yr + rng.normal(0, 1, 120),
        }
    )
    path = tmp_path / "preds.csv"
    df.to_csv(path, index=False)
    return path, df


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_evaluate_matches_python_api(csv_file, capsys) -> None:
    path, df = csv_file
    code, out, _ = run(
        capsys,
        "evaluate",
        str(path),
        "--y-true",
        "label",
        "--y-pred",
        "pred",
        "--y-prob",
        "prob",
        "--format",
        "json",
    )
    assert code == 0
    payload = json.loads(out)
    expected = es.evaluate(df["label"], df["pred"], y_prob=df["prob"])
    assert payload["metrics"]["f1"]["value"] == pytest.approx(float(expected["f1"]))
    assert payload["metrics"]["roc_auc"]["value"] == pytest.approx(float(expected["roc_auc"]))
    code, out, _ = run(capsys, "evaluate", str(path), "--y-true", "value", "--y-pred", "estimate")
    assert code == 0 and "regression" in out and "RMSE" in out
    code, out, _ = run(
        capsys,
        "evaluate",
        str(path),
        "--y-true",
        "label",
        "--y-pred",
        "pred",
        "--metrics",
        "accuracy,mcc",
        "--weight",
        "w",
        "--format",
        "csv",
    )
    assert out.splitlines()[0] == "metric,name,label,value" and len(out.splitlines()) == 3


@pytest.mark.parametrize(
    "ext,start",
    [
        ("json", "{"),
        ("csv", "metric,"),
        ("md", "| Metric"),
        ("tex", r"\begin"),
        ("html", "<!doctype"),
        ("txt", "EvalSuite"),
    ],
)
def test_output_format_from_extension(csv_file, capsys, tmp_path, ext: str, start: str) -> None:
    path, _ = csv_file
    out_path = tmp_path / f"result.{ext}"
    code, _, err = run(capsys, "evaluate", str(path), "--y-true", "label", "--y-pred", "pred", "-o", str(out_path))
    assert code == 0 and "Wrote" in err
    assert out_path.read_text(encoding="utf-8").startswith(start)


def test_report_compare_plot(csv_file, capsys, tmp_path) -> None:
    path, _ = csv_file
    code, out, _ = run(
        capsys, "report", str(path), "--y-true", "label", "--y-pred", "pred", "--format", "markdown"
    )
    assert code == 0 and out.startswith("| Class |")
    fig = tmp_path / "forest.png"
    code, out, _ = run(
        capsys,
        "compare",
        str(path),
        "--y-true",
        "label",
        "--pred",
        "A=pred",
        "--pred",
        "B=pred_b",
        "--prob",
        "A=prob",
        "--prob",
        "B=prob_b",
        "--resamples",
        "200",
        "--plot",
        str(fig),
    )
    assert code == 0 and "best: A" in out and fig.stat().st_size > 1000
    for kind, flags in (
        ("roc", ["--y-prob", "prob"]),
        ("pr", ["--y-prob", "prob"]),
        ("calibration", ["--y-prob", "prob"]),
        ("confusion", ["--y-pred", "pred"]),
        ("residuals", ["--y-pred", "estimate"]),
        ("predicted", ["--y-pred", "estimate"]),
    ):
        target = "value" if kind in ("residuals", "predicted") else "label"
        img = tmp_path / f"{kind}.png"
        assert run(capsys, "plot", kind, str(path), "--y-true", target, *flags, "-o", str(img))[0] == 0
        assert img.stat().st_size > 1000


def test_metrics_info_benchmark_help(capsys) -> None:
    code, out, _ = run(capsys, "metrics", "--category", "regression")
    assert code == 0 and "regression.rmse" in out and "classification" not in out
    code, out, _ = run(capsys, "info", "classification.mcc")
    assert code == 0 and "Formula:" in out and "References:" in out and "higher is better" in out
    code, out, _ = run(capsys, "benchmark", "--sizes", "300", "--repeat", "1", "--format", "markdown")
    assert code == 0 and out.startswith("| Case |")
    assert (
        run(
            capsys,
        )[0]
        == 0
    )


@pytest.mark.parametrize(
    "argv,message",
    [
        (["evaluate", "missing.csv", "--y-true", "a", "--y-pred", "b"], "File not found"),
        (["evaluate", "{csv}", "--y-true", "nope", "--y-pred", "pred"], "Columns in the file"),
        (["evaluate", "{csv}", "--y-true", "label"], "--y-pred, --y-prob"),
        (
            ["evaluate", "{csv}", "--y-true", "label", "--y-pred", "pred", "--metrics", "auc"],
            "Unknown classification",
        ),
        (["compare", "{csv}", "--y-true", "label", "--pred", "A=pred"], "at least two models"),
        (["compare", "{csv}", "--y-true", "label", "--pred", "Apred", "--pred", "B=pred"], "NAME=COLUMN"),
        (["plot", "roc", "{csv}", "--y-true", "label", "-o", "x.png"], "needs --y-prob"),
        (["plot", "confusion", "{csv}", "--y-true", "label", "-o", "x.png"], "needs --y-pred"),
        (["info", "auc"], "Unknown metric"),
    ],
)
def test_errors_are_clear_and_exit_2(csv_file, capsys, argv: list[str], message: str) -> None:
    path, _ = csv_file
    code, _, err = run(capsys, *[str(path) if a == "{csv}" else a for a in argv])
    assert code == 2 and message in err and "Traceback" not in err


def test_tsv_and_installed_command(tmp_path) -> None:
    tsv = tmp_path / "p.tsv"
    tsv.write_text("y\tp\n0\t0\n1\t1\n1\t0\n", encoding="utf-8")
    assert main(["evaluate", str(tsv), "--y-true", "y", "--y-pred", "p", "--metrics", "accuracy"]) == 0
    out = subprocess.run(
        [sys.executable, "-m", "evalsuite", "--version"], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == f"evalsuite-python {es.__version__}"
