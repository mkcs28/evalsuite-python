"""Classification report: per-class precision, recall, F1, specificity and support, with averages."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from .core.context import ClassificationContext
from .core.result import _fmt, _json_safe, _latex_escape, _latex_table
from .core.types import ArrayLike, ZeroDivision
from .core.validation import safe_divide, validate_zero_division

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["ClassificationReport", "classification_report"]

_COLUMNS = ("precision", "recall", "f1", "specificity", "support")


@dataclass(frozen=True, eq=False)
class ClassificationReport:
    """Per-class rows plus summary rows (``accuracy``, ``micro avg``, ``macro avg``, ``weighted avg``)."""

    target_type: str
    rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    n_samples: int
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(r)) for r in self.rows))
        object.__setattr__(self, "summary_rows", tuple(MappingProxyType(dict(r)) for r in self.summary_rows))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __getitem__(self, label: Any) -> Any:
        """Row for a class label or a summary name (``"macro avg"``)."""
        for row in (*self.rows, *self.summary_rows):
            if row["label"] == label:
                return row
        raise KeyError(f"No row {label!r}. Labels: {[r['label'] for r in self.rows]}.")

    def _table(self, digits: int) -> tuple[list[str], list[list[str]]]:
        header = ["", "precision", "recall", "f1", "specificity", "support"]
        out = []
        for row in (*self.rows, *self.summary_rows):
            cells = [str(row["label"])]
            for col in _COLUMNS:
                v = row.get(col)
                if v is None:
                    cells.append("")
                elif col == "support":
                    cells.append(str(int(v)) if float(v).is_integer() else _fmt(v, 2))
                else:
                    cells.append(_fmt(v, digits))
            out.append(cells)
        return header, out

    def summary(self, *, digits: int = 4) -> str:
        """Fixed-width text table, like scikit-learn's report plus a specificity column."""
        header, rows = self._table(digits)
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

        def fmt(cells: list[str]) -> str:
            return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

        lines = [fmt(header), ""]
        for i, r in enumerate(rows):
            if i == len(self.rows):
                lines.append("")
            lines.append(fmt(r))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe(
                {
                    "target_type": self.target_type,
                    "n_samples": self.n_samples,
                    "params": dict(self.params),
                    "classes": [dict(r) for r in self.rows],
                    "summary": [dict(r) for r in self.summary_rows],
                }
            ),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame([dict(r) for r in (*self.rows, *self.summary_rows)]).set_index("label")
        return frame

    def to_csv(self, path: Optional[str] = None) -> str:
        from .core.export import csv_text

        rows = [
            [r["label"], *(r.get(c) if r.get(c) is not None else float("nan") for c in _COLUMNS)]
            for r in (*self.rows, *self.summary_rows)
        ]
        text = csv_text(["label", *_COLUMNS], rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_markdown(self, *, digits: int = 4) -> str:
        header, rows = self._table(digits)
        header[0] = "Class"
        lines = ["| " + " | ".join(header) + " |", "| --- |" + " ---: |" * (len(header) - 1)]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines)

    def to_latex(self, *, digits: int = 4, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        header, rows = self._table(digits)
        header = ["Class", "Precision", "Recall", "F1", "Specificity", "Support"]
        return _latex_table(
            header,
            [[_latex_escape(c) for c in r] for r in rows],
            caption=caption or "Classification report.",
            label=label,
        )

    def to_html(self, *, digits: int = 4, caption: Optional[str] = None, full: bool = False) -> str:
        from .core.export import html_document, html_table

        header, rows = self._table(digits)
        header[0] = "Class"
        classes = [""] * len(self.rows) + ["avg"] * len(self.summary_rows)
        table = html_table(header, rows, caption=caption or "Classification report", row_classes=classes)
        meta = f"{self.target_type}, n = {self.n_samples}"
        return html_document("EvalSuite classification report", table, meta) if full else table

    def save(self, path: str, *, digits: int = 4) -> str:
        """Save as .json .csv .md .tex .html or .txt (chosen by the extension)."""
        from .core.export import save_as

        return save_as(
            path,
            {
                "json": self.to_json,
                "csv": self.to_csv,
                "markdown": lambda: self.to_markdown(digits=digits) + "\n",
                "latex": lambda: self.to_latex(digits=digits) + "\n",
                "html": lambda: self.to_html(digits=digits, full=True),
                "text": lambda: self.summary(digits=digits) + "\n",
            },
        )


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    zero_division: ZeroDivision = "warn",
) -> ClassificationReport:
    """Per-class precision, recall, F1, specificity and support, plus accuracy (single-label targets),
    micro (multilabel), macro and support-weighted averages. Values match scikit-learn's
    ``classification_report`` where both define them; specificity is EvalSuite's addition.

    >>> import evalsuite as es
    >>> print(es.classification_report([0, 1, 1, 2], [0, 1, 2, 2]))  # doctest: +SKIP
    """
    zd = validate_zero_division(zero_division)
    ctx = ClassificationContext(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    c = ctx.counts
    tp, fp, fn, tn, support = c["tp"], c["fp"], c["fn"], c["tn"], c["support"]
    prec = safe_divide(tp, tp + fp, zero_division=zd, metric="Precision")
    rec = safe_divide(tp, tp + fn, zero_division=zd, metric="Recall")
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn, zero_division=zd, metric="F1")
    spec = safe_divide(tn, tn + fp, zero_division=zd, metric="Specificity")
    rows = [
        {
            "label": lab,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "specificity": float(s),
            "support": float(n),
        }
        for lab, p, r, f, s, n in zip(ctx.labels.tolist(), prec, rec, f1, spec, support)
    ]
    total = float(support.sum())
    summary: list[dict[str, Any]] = []
    if ctx.target_type == "multilabel":
        tps, fps, fns, tns = tp.sum(), fp.sum(), fn.sum(), tn.sum()
        summary.append(
            {
                "label": "micro avg",
                "precision": float(safe_divide(tps, tps + fps, zero_division=zd, metric="Precision")),
                "recall": float(safe_divide(tps, tps + fns, zero_division=zd, metric="Recall")),
                "f1": float(safe_divide(2 * tps, 2 * tps + fps + fns, zero_division=zd, metric="F1")),
                "specificity": float(safe_divide(tns, tns + fps, zero_division=zd, metric="Specificity")),
                "support": total,
            }
        )
    else:
        correct = float(np.trace(ctx.confusion_matrix))
        summary.append(
            {
                "label": "accuracy",
                "precision": None,
                "recall": None,
                "f1": correct / ctx.total_weight,
                "specificity": None,
                "support": ctx.total_weight,
            }
        )
    stacked = np.vstack([prec, rec, f1, spec])
    macro = stacked.mean(axis=1)
    weighted = (stacked * support).sum(axis=1) / total if total > 0 else np.full(4, np.nan)
    for name, vals in (("macro avg", macro), ("weighted avg", weighted)):
        summary.append(
            {
                "label": name,
                "precision": float(vals[0]),
                "recall": float(vals[1]),
                "f1": float(vals[2]),
                "specificity": float(vals[3]),
                "support": total,
            }
        )
    return ClassificationReport(
        ctx.target_type,
        tuple(rows),
        tuple(summary),
        ctx.n,
        {"zero_division": zero_division, "weighted": sample_weight is not None},
    )
