"""One call (or one command) from a table of predictions to a full HTML evaluation report.

>>> import evalsuite as es
>>> es.auto_report("predictions.csv")                 # works out the columns and the task
>>> es.auto_report(df, target="label", prob="risk", groups=["sex"], path="report.html")

From the shell: ``evalsuite report predictions.csv`` writes ``report.html`` next to the data.

Columns are recognised by name (case-insensitive): the outcome (``y_true``, ``label``, ``target``, ``outcome``,
``truth``, ``actual``, ``y``), predicted labels or values (``y_pred``, ``pred``, ``prediction``, ``predicted``),
predicted probabilities (``y_prob``, ``prob``, ``proba``, ``probability``, ``risk``, ``score``, or one
``prob_<class>`` column per class), subgroups (``sex``, ``gender``, ``age_group``, ``race``, ``ethnicity``,
``site``, ``group``) and clusters (``patient_id``, ``subject_id``, ``cluster``). Anything detected wrongly can be
given explicitly. The task follows from the outcome: two classes → binary (discrimination, calibration, TRIPOD+AI
card, fairness when groups exist), more classes → multiclass, otherwise regression.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from .core.exceptions import InputValidationError, OptionalDependencyError

__all__ = ["auto_report", "AutoReport", "detect_columns"]

_TARGET = ("y_true", "label", "labels", "target", "outcome", "truth", "ground_truth", "actual", "y", "class")
_PRED = ("y_pred", "pred", "prediction", "predicted", "predictions", "yhat", "y_hat")
_PROB = ("y_prob", "prob", "proba", "probability", "risk", "score", "y_score", "p")
_GROUPS = ("sex", "gender", "age_group", "agegroup", "race", "ethnicity", "site", "hospital", "group", "subgroup")
_CLUSTERS = ("patient_id", "subject_id", "patient", "subject", "cluster", "cluster_id")


def _find(cols: Sequence[str], names: Sequence[str]) -> Optional[str]:
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def detect_columns(columns: Sequence[str]) -> dict[str, Any]:
    """Guess which column plays which role (see the module docstring)."""
    cols = [str(c) for c in columns]
    found: dict[str, Any] = {
        "target": _find(cols, _TARGET),
        "pred": _find(cols, _PRED),
        "prob": _find(cols, _PROB),
        "groups": [c for c in cols if c.lower().strip() in _GROUPS],
        "clusters": _find(cols, _CLUSTERS),
    }
    per_class = [c for c in cols if re.fullmatch(r"(?i)(prob|proba|p|y_prob)[_:](.+)", c)]
    if found["prob"] is None and len(per_class) >= 2:
        found["prob"] = per_class
    return found


@dataclass(frozen=True)
class AutoReport:
    """The detected set-up, the results that went into the report, and the report's HTML."""

    task: str
    columns: dict[str, Any]
    sections: dict[str, Any] = field(repr=False)
    html: str = field(repr=False)
    path: Optional[str] = None

    def summary(self) -> str:
        c = self.columns
        lines = [f"Auto report: {self.task} task"]
        for role in ("target", "pred", "prob", "groups", "clusters"):
            if c.get(role):
                lines.append(f"  {role:<9} {c[role]}")
        lines.append(f"  Sections: {', '.join(self.sections)}")
        if self.path:
            lines.append(f"  Written to {self.path}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _load(data: Any) -> Any:
    import pandas as pd

    if isinstance(data, (str, os.PathLike)):
        path = str(data)
        if not os.path.exists(path):
            raise InputValidationError(f"File not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tsv", ".tab"):
            return pd.read_csv(path, sep="\t")
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext in (".json", ".jsonl"):
            return pd.read_json(path, lines=ext == ".jsonl")
        return pd.read_csv(path)
    if hasattr(data, "to_pandas") and not hasattr(data, "iloc"):
        return data.to_pandas()
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if hasattr(data, "iloc"):
        return data
    raise InputValidationError("data must be a file path, a DataFrame or a dict of columns.")


def auto_report(
    data: Any,
    *,
    target: Optional[str] = None,
    pred: Optional[str] = None,
    prob: Union[str, Sequence[str], None] = None,
    groups: Optional[Sequence[str]] = None,
    clusters: Optional[str] = None,
    task: Optional[str] = None,
    threshold: float = 0.5,
    path: Optional[str] = None,
    title: Optional[str] = None,
    random_state: Optional[int] = 0,
) -> AutoReport:
    """Detect the columns and task (see the module docstring), evaluate, and build one HTML report. With a file
    path as ``data``, the report is written next to it as ``<name>.report.html`` unless ``path`` is given."""
    import evalsuite as es

    df = _load(data)
    cols = [str(c) for c in df.columns]
    found = detect_columns(cols)
    if target is not None:
        found["target"] = target
    if pred is not None:
        found["pred"] = pred
    if prob is not None:
        found["prob"] = prob
    if groups is not None:
        found["groups"] = list(groups)
    if clusters is not None:
        found["clusters"] = clusters
    t = found["target"]
    if t is None:
        raise InputValidationError(
            f"Could not tell which column is the outcome among {cols[:10]}; pass target= (or --y-true)."
        )
    for role in ("target", "pred", "clusters"):
        if found[role] is not None and found[role] not in cols:
            raise InputValidationError(f"Column {found[role]!r} ({role}) is not in the data.")
    probs = found["prob"]
    prob_cols = [probs] if isinstance(probs, str) else list(probs or [])
    missing = [c for c in prob_cols + list(found["groups"]) if c not in cols]
    if missing:
        raise InputValidationError(f"Columns not in the data: {', '.join(missing)}.")
    if found["pred"] is None and not prob_cols:
        raise InputValidationError("Found no prediction or probability column; pass pred= or prob=.")
    y = df[t].to_numpy()
    keep = ~df[[t] + ([found["pred"]] if found["pred"] else []) + prob_cols].isna().any(axis=1).to_numpy()
    df = df.loc[keep].reset_index(drop=True)
    y = df[t].to_numpy()
    n_classes = len(np.unique(y))
    numeric_target = np.issubdtype(y.dtype, np.number)
    if task is None:
        task = "binary" if n_classes == 2 else "regression" if numeric_target and n_classes > 20 else "multiclass"
    yp = df[found["pred"]].to_numpy() if found["pred"] else None
    pp = None
    if prob_cols:
        pp = df[prob_cols[0]].to_numpy(dtype=float) if len(prob_cols) == 1 else df[prob_cols].to_numpy(dtype=float)
    cl = df[found["clusters"]].to_numpy() if found["clusters"] else None
    sections: dict[str, Any] = {}
    intro = (
        f"{len(df)} rows of {os.path.basename(str(data)) if isinstance(data, (str, os.PathLike)) else 'data'}"
        f" ({int((~keep).sum())} with missing values dropped). Outcome: {t}; task: {task}."
    )
    if task == "regression":
        if yp is None:
            raise InputValidationError("Regression needs a column of predicted values (pred=).")
        sections["Evaluation"] = es.evaluate(y, yp, clusters=cl, random_state=random_state)
        with contextlib.suppress(ImportError, OptionalDependencyError):  # plots need Matplotlib
            sections["Residuals"] = es.plot.residuals(y, yp)
    else:
        sections["Evaluation"] = es.evaluate(y, yp, y_prob=pp, clusters=cl, random_state=random_state)
        if yp is not None:
            sections["Per-class report"] = es.classification_report(y, yp)
        if task == "binary" and pp is not None and np.ndim(pp) == 1:
            pos = np.unique(y).max()
            yb = (y == pos).astype(int)
            sections["Calibration"] = es.calibration_report(yb, pp, n_resamples=500, random_state=random_state)
            grp = df[found["groups"][0]].to_numpy() if found["groups"] else None
            sections["TRIPOD+AI performance"] = es.report_card(
                yb, pp, threshold=threshold, groups=grp, random_state=random_state
            )
            with contextlib.suppress(ImportError, OptionalDependencyError):  # plots need Matplotlib
                sections["ROC curve"] = es.plot.roc(yb, pp)
                sections["Calibration curve"] = es.plot.calibration(yb, pp)
        if found["groups"] and task == "binary":
            yb = (y == np.unique(y).max()).astype(int)
            pb = (pp >= threshold).astype(int) if pp is not None and np.ndim(pp) == 1 else None
            yhat = pb if pb is not None else (yp == np.unique(y).max()).astype(int)
            gcols = {g: df[g].astype(str).to_numpy() for g in found["groups"]}
            sections["Fairness"] = es.fairness_report(
                yb,
                yhat,
                pp if pp is not None and np.ndim(pp) == 1 else None,
                groups=gcols,
                random_state=random_state,
            )
    page = es.html_report(
        sections,
        title=title or "Evaluation report",
        intro=intro,
        metadata={"Outcome": t, "Task": task},
    )
    out = path
    if out is None and isinstance(data, (str, os.PathLike)):
        out = os.path.splitext(str(data))[0] + ".report.html"
    if out is not None:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(page)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:  # pragma: no cover
        pass
    return AutoReport(task, found, sections, page, out)
