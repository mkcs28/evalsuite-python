"""Generate the example notebooks in notebooks/ (run: python scripts/make_notebooks.py; --check runs them)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
REPO = "mkcs28/evalsuite-python"
INSTALL = "%pip install -q evalsuite-python scikit-learn"

NOTEBOOKS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "01_classification_quickstart": (
        "EvalSuite quickstart: classification with confidence intervals",
        [
            (
                "md",
                "Every number comes with its uncertainty: `evaluate()` reports 95% intervals by default and can test a new model against a baseline on the same data.",
            ),
            ("code", INSTALL),
            (
                "code",
                """import evalsuite as es
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
base = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
new = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)""",
            ),
            (
                "code",
                """result = es.evaluate(
    y_te, new.predict(X_te), y_prob=new.predict_proba(X_te)[:, 1],
    baseline_pred=base.predict(X_te), baseline_prob=base.predict_proba(X_te)[:, 1],
)
print(result.summary())""",
            ),
            ("md", "Export straight into a paper or report:"),
            ("code", "print(result.to_markdown())"),
            (
                "code",
                """# Compare several models at once, with paired tests and Holm correction
cmp = es.compare(y_te, {"logistic": base.predict(X_te), "forest": new.predict(X_te)}, metrics=["accuracy", "f1"], random_state=0)
print(cmp)""",
            ),
        ],
    ),
    "02_clinical_tripod_ai": (
        "Clinical prediction models: TRIPOD+AI report card",
        [
            (
                "md",
                "`report_card` writes the performance section TRIPOD+AI asks for: discrimination, calibration, clinical utility and subgroups, with intervals, plus the checklist.",
            ),
            ("code", INSTALL),
            (
                "code",
                """import numpy as np
import evalsuite as es

rng = np.random.default_rng(0)
n = 1500
risk_score = rng.normal(size=n)
y_prob = 1 / (1 + np.exp(-(1.4 * risk_score - 1.2)))
y_true = (rng.random(n) < y_prob).astype(int)
site = rng.choice(["Hospital A", "Hospital B", "Hospital C"], size=n)""",
            ),
            (
                "code",
                """card = es.report_card(y_true, y_prob, threshold=0.2, groups=site, group_name="site")
print(card.paragraph())""",
            ),
            ("code", "print(card.to_markdown())"),
            (
                "code",
                """# Diagnostic accuracy at the threshold, and the decision curve
report = es.diagnostic_report(y_true, (y_prob >= 0.2).astype(int))
print(report)
curve = es.decision_curve(y_true, y_prob, thresholds=np.linspace(0.05, 0.5, 10))
print(curve)""",
            ),
        ],
    ),
    "03_llm_leaderboard": (
        "LLM evaluation: a leaderboard that knows when models are tied",
        [
            (
                "md",
                "Rank models with intervals and paired significance tests: models that cannot be told apart share a rank. Works on JSONL files, lists of dicts or DataFrames.",
            ),
            ("code", INSTALL),
            (
                "code",
                """import random
import evalsuite as es

random.seed(0)
questions = [{"id": i, "reference": str(i * 7)} for i in range(300)]
def run(skill):
    return [{"id": q["id"], "prediction": q["reference"] if random.random() < skill else "unknown"} for q in questions]

runs = {"model-a": run(0.78), "model-b": run(0.76), "model-c": run(0.61)}
board = es.leaderboard(runs, references=questions, scorer="exact_match")
print(board.summary())""",
            ),
            ("code", "print(board.to_markdown())"),
            (
                "md",
                "The same ranking runs in your browser, with no upload, on the [EvalSuite tools page](https://evalsuite-nine.vercel.app/tools).",
            ),
        ],
    ),
    "04_monitoring_drift": (
        "Monitoring a deployed model: windows, drift and alerts",
        [
            (
                "md",
                "`Monitor` recomputes metrics over a sliding window as labels arrive, compares scores with the validation distribution (PSI, Kolmogorov–Smirnov) and fires alerts.",
            ),
            ("code", INSTALL),
            (
                "code",
                """import numpy as np
import evalsuite as es

rng = np.random.default_rng(1)
def batch(n, shift=0.0):
    y = (rng.random(n) < 0.4).astype(int)
    p = np.clip(np.where(y == 1, 0.7 - shift, 0.3 + shift) + rng.normal(0, 0.15, n), 0.01, 0.99)
    return y, p

_, validation_scores = batch(2000)
mon = es.Monitor(window=500, alerts={"roc_auc": ("<", 0.8), "psi": (">", 0.25)},
                 reference_scores=validation_scores, min_n=200)""",
            ),
            (
                "code",
                """for week, shift in enumerate([0.0, 0.0, 0.05, 0.12, 0.2], start=1):
    y, p = batch(250, shift)
    fired = mon.update(y, y_pred=(p >= 0.5).astype(int), y_prob=p)
    print(f"week {week}:", fired or "ok")""",
            ),
            ("code", "print(mon.snapshot().summary())\nprint(mon.drift())"),
        ],
    ),
    "05_vision_segmentation": (
        "Computer vision: segmentation and detection metrics",
        [
            ("md", "Dice, IoU, Hausdorff (HD95) and surface distances for masks, and COCO-style mAP for boxes."),
            ("code", INSTALL),
            (
                "code",
                """import numpy as np
import evalsuite as es

yy, xx = np.mgrid[0:256, 0:256] - 128.0
truth = ((yy**2 + xx**2) < 60**2).astype(int)
pred = (((yy - 4)**2 + xx**2) < 57**2).astype(int)
print(es.dice(truth, pred, num_classes=2))
print(es.iou(truth, pred, num_classes=2))
print(es.hausdorff_distance(truth, pred, percentile=95))""",
            ),
        ],
    ),
}


def notebook(title: str, cells: list[tuple[str, str]], name: str) -> dict:
    badge = (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}.ipynb)"
    )
    out = [{"cell_type": "markdown", "metadata": {}, "source": f"# {title}\n\n{badge}"}]
    for kind, src in cells:
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            out.append(
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
            )
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, (title, cells) in NOTEBOOKS.items():
        (OUT / f"{name}.ipynb").write_text(json.dumps(notebook(title, cells, name), indent=1) + "\n")
        if "--check" in sys.argv:
            code = "\n".join(src for kind, src in cells if kind == "code" and not src.startswith("%pip"))
            print(f"== {name}")
            exec(compile(code, name, "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
