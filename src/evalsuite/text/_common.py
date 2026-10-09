"""Shared input handling for text, question-answering, retrieval and reasoning metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Callable, TypeVar, Union

import numpy as np

from ..core.exceptions import InputValidationError

F = TypeVar("F", bound=Callable[..., Any])

Texts = Sequence[str]
References = Union[Sequence[str], Sequence[Sequence[str]]]


def per_item(fn: F) -> F:
    """Mark a metric whose observations are whole items (texts, ranked lists, sample sets).

    ``bootstrap_ci``, ``compare`` and the paired tests then resample items as Python objects and never stratify
    by value, so a corpus metric such as BLEU is recomputed on each resampled corpus."""
    fn.__evalsuite_items__ = True  # type: ignore[attr-defined]
    return fn


def _seq(x: Any, name: str) -> list[Any]:
    if x is None:
        raise InputValidationError(f"{name} is required but was None.")
    if isinstance(x, (str, bytes)):
        raise InputValidationError(f"{name} must be a list (one entry per example), not a single string.")
    if isinstance(x, np.ndarray):
        x = x.tolist()
    items = list(x)
    if not items:
        raise InputValidationError(f"{name} is empty. Provide at least one example.")
    return items


def as_texts(x: Any, name: str) -> list[str]:
    """A list of strings, one per example."""
    items = _seq(x, name)
    for i, t in enumerate(items):
        if not isinstance(t, str):
            raise InputValidationError(f"{name}[{i}] must be a string; got {type(t).__name__}.")
    return items


def as_references(x: Any, n: int, name: str = "references") -> list[list[str]]:
    """References as one list of strings per example. Accepts one string per example or a list of strings per
    example (multiple references)."""
    items = _seq(x, name)
    if len(items) != n:
        raise InputValidationError(
            f"{name} and predictions must contain the same number of examples. Received {len(items)} and {n}."
        )
    out: list[list[str]] = []
    for i, r in enumerate(items):
        refs = [r] if isinstance(r, str) else list(r) if isinstance(r, (list, tuple, np.ndarray)) else None
        if refs is None or not refs or not all(isinstance(s, str) for s in refs):
            raise InputValidationError(
                f"{name}[{i}] must be a string or a non-empty list of strings (several references)."
            )
        out.append(refs)
    return out


def word_ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def check_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise InputValidationError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def as_mapping_or_set(x: Any, name: str) -> Mapping[Any, float]:
    """Relevance judgements for one query: a set/list of relevant ids (all grade 1) or a mapping id -> grade."""
    if isinstance(x, Mapping):
        out = {}
        for k, v in x.items():
            g = float(v)
            if not np.isfinite(g) or g < 0:
                raise InputValidationError(
                    f"{name}: relevance grades must be finite and >= 0; got {v!r} for {k!r}."
                )
            if g > 0:
                out[k] = g
        return out
    if isinstance(x, (str, bytes)):
        raise InputValidationError(f"{name} must be a set, list or mapping of relevant ids, not a string.")
    return {k: 1.0 for k in x}
