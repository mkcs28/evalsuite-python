"""One report for text generation: the reference-based metrics computed together, with exports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, cast

from ..core.exceptions import InputValidationError
from ..core.export import PathLike, csv_text, html_document, html_table, save_as
from ..core.result import _fmt, _json_safe, _latex_escape, _latex_table
from ._common import as_references, as_texts

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["TextReport", "text_report"]

# (key, label, scale note) in report order
_ALL = (
    ("bleu", "BLEU", "0–100, sacreBLEU"),
    ("chrf", "chrF", "0–100"),
    ("chrf_pp", "chrF++", "0–100"),
    ("ter", "TER", "0–∞, lower is better"),
    ("rouge_1", "ROUGE-1", "F1, 0–1"),
    ("rouge_2", "ROUGE-2", "F1, 0–1"),
    ("rouge_l", "ROUGE-L", "F1, 0–1"),
    ("rouge_lsum", "ROUGE-Lsum", "F1, 0–1"),
    ("meteor", "METEOR", "0–1"),
    ("exact_match", "Exact match", "0–1, SQuAD normalisation"),
    ("token_f1", "Token F1", "0–1, SQuAD normalisation"),
)
_KEYS = tuple(k for k, _, _ in _ALL)


@dataclass(frozen=True, eq=False)
class TextReport:
    """Reference-based text metrics for one system. ``report["bleu"]`` gives one value."""

    values: Any
    params: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __getitem__(self, name: str) -> float:
        try:
            return float(self.values[name])
        except KeyError:
            raise KeyError(f"No value {name!r}. Available: {', '.join(self.values)}.") from None

    def _rows(self, digits: int) -> list[list[str]]:
        return [[label, _fmt(self.values[k], digits), note] for k, label, note in _ALL if k in self.values]

    def summary(self, *, digits: int = 4) -> str:
        p = self.params
        lines = [f"EvalSuite text evaluation ({p['n_examples']} examples, {p['max_references']} reference(s) max)"]
        rows = self._rows(digits)
        width = max(len(r[0]) for r in rows)
        lines += [f"  {r[0]:<{width}}  {r[1]:>8}   {r[2]}" for r in rows]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _json_safe({"values": dict(self.values), "params": dict(self.params)}))

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame({"value": dict(self.values)})
        return frame

    def to_markdown(self, *, digits: int = 4) -> str:
        lines = ["| Metric | Value | Scale |", "| --- | ---: | --- |"]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in self._rows(digits)])

    def to_latex(self, *, digits: int = 4, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        rows = [[_latex_escape(c) for c in r[:2]] for r in self._rows(digits)]
        return _latex_table(["Metric", "Value"], rows, caption or "Text generation metrics.", label)

    def to_csv(self, path: Optional[PathLike] = None) -> str:
        text = csv_text(["metric", "value"], [[k, float(v)] for k, v in self.values.items()])
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_html(self, *, digits: int = 4, full: bool = True) -> str:
        body = html_table(["Metric", "Value", "Scale"], self._rows(digits), caption="Text generation metrics")
        return html_document("EvalSuite text evaluation", body) if full else body

    def save(self, path: PathLike) -> str:
        return save_as(
            path,
            {
                "json": self.to_json,
                "csv": self.to_csv,
                "markdown": self.to_markdown,
                "latex": self.to_latex,
                "html": self.to_html,
                "text": self.summary,
            },
        )


def text_report(
    references: Any,
    predictions: Any,
    *,
    metrics: Optional[Any] = None,
    meteor_stemmer: Any = "porter",
) -> TextReport:
    """BLEU, chrF, chrF++, TER (sacreBLEU conventions), ROUGE-1/2/L/Lsum (rouge-score), METEOR (NLTK), exact
    match and token F1 (SQuAD) in one report. ``metrics`` restricts the list (keys as in the report); METEOR
    is skipped automatically when NLTK is missing and ``meteor_stemmer="porter"``."""
    from . import generation as g
    from . import qa

    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    wanted = list(_KEYS) if metrics is None else list(metrics)
    unknown = [m for m in wanted if m not in _KEYS]
    if unknown:
        raise InputValidationError(f"Unknown report metric(s) {unknown}; choose from {list(_KEYS)}.")
    fns = {
        "bleu": lambda: g.bleu(refs, preds),
        "chrf": lambda: g.chrf(refs, preds),
        "chrf_pp": lambda: g.chrf(refs, preds, word_order=2),
        "ter": lambda: g.ter(refs, preds),
        "rouge_1": lambda: g.rouge_1(refs, preds),
        "rouge_2": lambda: g.rouge_2(refs, preds),
        "rouge_l": lambda: g.rouge_l(refs, preds),
        "rouge_lsum": lambda: g.rouge_lsum(refs, preds),
        "meteor": lambda: g.meteor(refs, preds, stemmer=meteor_stemmer),
        "exact_match": lambda: qa.exact_match(refs, preds),
        "token_f1": lambda: qa.token_f1(refs, preds),
    }
    values: dict[str, float] = {}
    skipped = []
    for key in wanted:
        try:
            values[key] = float(fns[key]())
        except ImportError:
            if key == "meteor" and metrics is None:
                skipped.append("meteor (install nltk)")
                continue
            raise
    return TextReport(
        values,
        {"n_examples": len(preds), "max_references": max(len(r) for r in refs), "skipped": skipped},
    )
