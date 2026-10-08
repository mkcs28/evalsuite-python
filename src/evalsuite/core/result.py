"""Immutable result objects with portable exports (dict, JSON, DataFrame, Markdown, LaTeX).

JSON uses the standard library only; NaN and infinity become ``null`` so the output is valid JSON.
Pickle is never used.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["EvaluationResult", "MetricResult"]

Number = Union[float, int]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "–"
    if isinstance(value, (float, np.floating)):
        return "NaN" if math.isnan(value) else f"{float(value):.{digits}f}"
    return str(value)


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in text)


@dataclass(frozen=True, eq=False)
class MetricResult:
    """One metric's value.

    ``value`` is a float, or a read-only array of per-class values when ``average=None``
    (aligned with ``labels``). ``params`` records how it was computed (average, zero_division, ...).
    """

    metric: str
    name: str
    value: Union[float, np.ndarray]
    params: Mapping[str, Any] = field(default_factory=dict)
    labels: Optional[tuple[Any, ...]] = None

    def __post_init__(self) -> None:
        if isinstance(self.value, np.ndarray):
            arr = np.array(self.value, dtype=np.float64)
            arr.setflags(write=False)
            object.__setattr__(self, "value", arr)
        else:
            object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __float__(self) -> float:
        if isinstance(self.value, np.ndarray):
            raise TypeError(f"{self.name} has per-class values; use .value, .per_class() or .to_dict().")
        return float(self.value)

    # Scalar results behave like numbers: f"{r:.3f}", r > 0.8, round(r, 3), r == 0.75.
    def __format__(self, spec: str) -> str:
        return format(float(self), spec) if spec else repr(self)

    def __round__(self, ndigits: Optional[int] = None) -> float:
        return round(float(self), ndigits)

    def __lt__(self, other: Number) -> bool:
        return float(self) < float(other)

    def __le__(self, other: Number) -> bool:
        return float(self) <= float(other)

    def __gt__(self, other: Number) -> bool:
        return float(self) > float(other)

    def __ge__(self, other: Number) -> bool:
        return float(self) >= float(other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (int, float, np.floating, np.integer)) and not isinstance(self.value, np.ndarray):
            return float(self.value) == float(other)
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        if isinstance(self.value, np.ndarray):
            pairs = ", ".join(f"{lab}: {_fmt(v, 4)}" for lab, v in zip(self.labels or (), self.value))
            return f"MetricResult({self.metric}: {{{pairs}}})"
        return f"MetricResult({self.metric}={_fmt(self.value, 6)})"

    def per_class(self) -> dict[Any, float]:
        if not isinstance(self.value, np.ndarray):
            raise TypeError(f"{self.name} is a single value; compute it with average=None for per-class values.")
        return {lab: float(v) for lab, v in zip(self.labels or (), self.value)}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"metric": self.metric, "name": self.name, "value": _json_safe(self.value)}
        if self.labels is not None:
            out["labels"] = _json_safe(list(self.labels))
        if self.params:
            out["params"] = _json_safe(dict(self.params))
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame
        if isinstance(self.value, np.ndarray):
            frame = pd.DataFrame({"label": list(self.labels or ()), self.metric: self.value})
        else:
            frame = pd.DataFrame({"metric": [self.metric], "name": [self.name], "value": [self.value]})
        return frame

    def to_markdown(self, *, digits: int = 4) -> str:
        if isinstance(self.value, np.ndarray):
            rows = [f"| {lab} | {_fmt(v, digits)} |" for lab, v in zip(self.labels or (), self.value)]
            return "\n".join([f"| Label | {self.name} |", "| --- | ---: |", *rows])
        return "\n".join(["| Metric | Value |", "| --- | ---: |", f"| {self.name} | {_fmt(self.value, digits)} |"])

    def to_latex(self, *, digits: int = 4) -> str:
        return _latex_table(
            ["Metric", "Value"] if not isinstance(self.value, np.ndarray) else ["Label", _latex_escape(self.name)],
            [[_latex_escape(self.name), _fmt(self.value, digits)]]
            if not isinstance(self.value, np.ndarray)
            else [[_latex_escape(str(lab)), _fmt(v, digits)] for lab, v in zip(self.labels or (), self.value)],
        )

    def _rows(self) -> list[list[Any]]:
        if isinstance(self.value, np.ndarray):
            return [[self.metric, self.name, lab, float(v)] for lab, v in zip(self.labels or (), self.value)]
        return [[self.metric, self.name, "", float(self.value)]]

    def to_csv(self, path: Optional[str] = None) -> str:
        """CSV with columns metric, name, label, value (one row per class for per-class results)."""
        from .export import csv_text

        text = csv_text(["metric", "name", "label", "value"], self._rows())
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 4, full: bool = False) -> str:
        """HTML table (``full=True``: a standalone page)."""
        from .export import html_document, html_table

        if isinstance(self.value, np.ndarray):
            table = html_table(
                ["Label", self.name], [[lab, _fmt(v, digits)] for lab, v in zip(self.labels or (), self.value)]
            )
        else:
            table = html_table(["Metric", "Value"], [[self.name, _fmt(self.value, digits)]])
        return html_document(self.name, table) if full else table


def _latex_table(
    header: list[str], rows: list[list[str]], caption: Optional[str] = None, label: Optional[str] = None
) -> str:
    cols = "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[ht]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class EvaluationResult(Mapping[str, MetricResult]):
    """All metrics from one evaluation, keyed by metric id (e.g. ``"accuracy"``)."""

    task: str
    metrics: Mapping[str, MetricResult]
    n_samples: int
    target_type: str
    labels: Optional[tuple[Any, ...]] = None
    confusion_matrix: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.confusion_matrix is not None:
            cm = np.array(self.confusion_matrix)
            cm.setflags(write=False)
            object.__setattr__(self, "confusion_matrix", cm)

    # Mapping interface: result["f1"]
    def __getitem__(self, key: str) -> MetricResult:
        try:
            return self.metrics[key]
        except KeyError:
            raise KeyError(f"No metric '{key}' in this result. Available: {', '.join(self.metrics)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.metrics)

    def __len__(self) -> int:
        return len(self.metrics)

    @property
    def value(self) -> dict[str, Any]:
        return {k: m.value for k, m in self.metrics.items()}

    def summary(self, *, digits: int = 4) -> str:
        width = max((len(m.name) for m in self.metrics.values()), default=6)
        lines = [f"EvalSuite {self.task} evaluation ({self.target_type}, n={self.n_samples})"]
        for m in self.metrics.values():
            if isinstance(m.value, np.ndarray):
                lines.append(
                    f"  {m.name:<{width}}  "
                    + ", ".join(f"{lab}={_fmt(v, digits)}" for lab, v in zip(m.labels or (), m.value))
                )
            else:
                lines.append(f"  {m.name:<{width}}  {_fmt(m.value, digits)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "task": self.task,
            "target_type": self.target_type,
            "n_samples": self.n_samples,
            "metrics": {k: m.to_dict() for k, m in self.metrics.items()},
            "metadata": _json_safe(dict(self.metadata)),
        }
        if self.labels is not None:
            out["labels"] = _json_safe(list(self.labels))
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = _json_safe(self.confusion_matrix)
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def _scalar_rows(self) -> list[MetricResult]:
        return [m for m in self.metrics.values() if not isinstance(m.value, np.ndarray)]

    def to_dataframe(self) -> pd.DataFrame:
        """One row per scalar metric (per-class metrics: use ``result[id].to_dataframe()``)."""
        import pandas as pd

        rows = self._scalar_rows()
        frame: pd.DataFrame = pd.DataFrame(
            {
                "metric": [m.metric for m in rows],
                "name": [m.name for m in rows],
                "value": [m.value for m in rows],
            }
        )
        return frame

    def to_markdown(self, *, digits: int = 4) -> str:
        rows = [f"| {m.name} | {_fmt(m.value, digits)} |" for m in self._scalar_rows()]
        return "\n".join(["| Metric | Value |", "| --- | ---: |", *rows])

    def to_latex(self, *, digits: int = 4, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        rows = [[_latex_escape(m.name), _fmt(m.value, digits)] for m in self._scalar_rows()]
        return _latex_table(["Metric", "Value"], rows, caption=caption, label=label)

    def _csv_rows(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for m in self.metrics.values():
            rows.extend(m._rows())
        return rows

    def to_csv(self, path: Optional[str] = None) -> str:
        """CSV with columns metric, name, label, value; per-class metrics get one row per label."""
        from .export import csv_text

        text = csv_text(["metric", "name", "label", "value"], self._csv_rows())
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def _meta_line(self) -> str:
        return (
            f"{self.task}, {self.target_type}, n = {self.n_samples}; EvalSuite "
            f"{self.metadata.get('evalsuite_version', '')}"
        )

    def to_html(self, *, digits: int = 4, caption: Optional[str] = None, full: bool = False) -> str:
        """HTML report: the metric table, per-class tables and the confusion matrix (``full=True``: standalone
        page with inline CSS and no scripts)."""
        from .export import html_document, html_table

        parts = [
            html_table(
                ["Metric", "Value"],
                [[m.name, _fmt(m.value, digits)] for m in self._scalar_rows()],
                caption=caption or "Metrics",
            )
        ]
        for m in self.metrics.values():
            if isinstance(m.value, np.ndarray):
                parts.append(
                    html_table(
                        ["Label", m.name],
                        [[lab, _fmt(v, digits)] for lab, v in zip(m.labels or (), m.value)],
                        caption=f"{m.name} per class",
                    )
                )
        if self.confusion_matrix is not None and self.labels is not None:
            cm = self.confusion_matrix
            integral = bool(np.all(np.mod(cm, 1) == 0))
            parts.append(
                html_table(
                    ["True \\ predicted", *(str(lab) for lab in self.labels)],
                    [
                        [lab, *(int(v) if integral else _fmt(v, digits) for v in row)]
                        for lab, row in zip(self.labels, cm)
                    ],
                    caption="Confusion matrix (rows: true, columns: predicted)",
                )
            )
        body = "\n".join(parts)
        return html_document("EvalSuite evaluation report", body, self._meta_line()) if full else body

    def save(self, path: str, *, digits: int = 4) -> str:
        """Save in the format given by the extension: .json .csv .md .tex .html .txt."""
        from .export import save_as

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
