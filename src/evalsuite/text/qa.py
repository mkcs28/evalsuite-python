"""Question-answering and correctness metrics: exact match and token-level F1 (SQuAD convention)."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import as_references, as_texts, per_item

__all__ = ["exact_match", "normalize_answer", "token_f1"]

_C = "qa"
_REF_SQUAD = (
    "Rajpurkar P, Zhang J, Lopyrev K, Liang P. SQuAD: 100,000+ questions for machine comprehension of text. "
    "EMNLP. 2016:2383-2392."
)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation and the articles a/an/the, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _norm(normalize: Any) -> Any:
    if normalize is True:
        return normalize_answer
    if normalize is False or normalize is None:
        return lambda s: s
    if callable(normalize):
        return normalize
    raise InputValidationError("normalize must be True (SQuAD), False or a callable str -> str.")


def _f1(pred: str, ref: str) -> float:
    p, r = pred.split(), ref.split()
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if not p or not r:
        return float(p == r)  # both empty -> 1, one empty -> 0
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def _finish(metric: str, name: str, scores: list[float], average: Optional[str], normalize: Any) -> MetricResult:
    arr = np.array(scores, dtype=float)
    value: Any = arr if average is None else float(arr.mean())
    norm = "squad" if normalize is True else ("none" if not normalize else "custom")
    return MetricResult(metric, name, value, {"normalize": norm, "n_examples": len(scores)})


@register(
    category=_C,
    task="question-answering",
    name="Exact match",
    definition="Share of predictions identical to a reference after normalization (SQuAD: lowercase, no "
    "punctuation or articles, single spaces); with several references, any match counts.",
    formula="mean_i max_r [norm(prediction_i) = norm(reference_ir)]",
    range="[0, 1]",
    input_requirements=("references", "predictions"),
    references=(_REF_SQUAD,),
)
@per_item
def exact_match(
    references: Any, predictions: Any, *, normalize: Any = True, average: Optional[str] = "mean"
) -> MetricResult:
    """Exact match (0–1). ``normalize=False`` compares raw strings; pass a callable for custom rules."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    f = _norm(normalize)
    scores = [float(any(f(p) == f(r) for r in rs)) for p, rs in zip(preds, refs)]
    return _finish("exact_match", "Exact match", scores, average, normalize)


@register(
    category=_C,
    task="question-answering",
    name="Token F1",
    definition="Harmonic mean of token precision and recall between the normalized prediction and reference "
    "(bag of words, SQuAD); best reference per example, averaged over examples.",
    formula="F1 = 2·P·R/(P + R), P = |common|/|pred tokens|, R = |common|/|ref tokens|",
    range="[0, 1]",
    input_requirements=("references", "predictions"),
    references=(_REF_SQUAD,),
)
@per_item
def token_f1(
    references: Any, predictions: Any, *, normalize: Any = True, average: Optional[str] = "mean"
) -> MetricResult:
    """Token-level F1 as in the official SQuAD v1.1 evaluation script (0–1)."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    f = _norm(normalize)
    scores = [max(_f1(f(p), f(r)) for r in rs) for p, rs in zip(preds, refs)]
    return _finish("token_f1", "Token F1", scores, average, normalize)
