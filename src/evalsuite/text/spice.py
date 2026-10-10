"""SPICE: semantic propositional image caption evaluation (Anderson et al. 2016).

SPICE compares the scene graphs of a candidate caption and its references: the F-score between their semantic
tuples — objects ``(o,)``, attributes ``(o, a)`` and relations ``(s, r, o)`` — where two tuples match when every
element matches exactly or as synonyms. The reference implementation parses captions with the Stanford
Scene Graph Parser (Java); EvalSuite takes the tuples directly, or any ``parser`` callable that turns a caption
into tuples (the Java parser's output, a Python scene-graph parser, an LLM), so the scoring itself runs without
Java.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import numpy as np

from ..core.exceptions import InputValidationError, OptionalDependencyError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import per_item

__all__ = ["spice"]

Tuple_ = tuple[str, ...]
_REF_SPICE = (
    "Anderson P, Fernando B, Johnson M, Gould S. SPICE: semantic propositional image caption evaluation. ECCV. "
    "2016:382-398."
)


def _tuples(x: Any, name: str) -> list[Tuple_]:
    if isinstance(x, (str, bytes)) or not isinstance(x, Iterable):
        raise InputValidationError(f"{name} must be a list of tuples (or pass parser= to parse captions).")
    out = []
    for t in x:
        tt = (t,) if isinstance(t, str) else tuple(t)
        if not 1 <= len(tt) <= 3 or not all(isinstance(e, str) and e.strip() for e in tt):
            raise InputValidationError(f"{name}: every tuple needs 1-3 non-empty strings; got {t!r}.")
        out.append(tuple(e.strip().lower() for e in tt))
    return out


def _wordnet_synonyms() -> Callable[[str], set[str]]:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:  # pragma: no cover - nltk is installed with the llm extra
        raise OptionalDependencyError("nltk", "llm", "SPICE synonym matching (synonyms='wordnet')") from exc

    def syn(word: str) -> set[str]:
        try:
            return {s.name() for s in wn.synsets(word.replace(" ", "_"))}
        except LookupError as exc:  # pragma: no cover - depends on the local NLTK data
            raise OptionalDependencyError("nltk wordnet data", "llm", "run nltk.download('wordnet')") from exc

    return syn


def _match_count(cand: set[Tuple_], ref: set[Tuple_], syn: Optional[Callable[[str], set[str]]]) -> int:
    """Number of candidate tuples matched by some reference tuple (each reference tuple used once)."""
    if syn is None:
        return len(cand & ref)
    cache: dict[str, set[str]] = {}

    def keys(w: str) -> set[str]:
        if w not in cache:
            cache[w] = {w} | set(syn(w))
        return cache[w]

    def same(a: Tuple_, b: Tuple_) -> bool:
        return len(a) == len(b) and all(x == y or keys(x) & keys(y) for x, y in zip(a, b))

    pool = list(ref)
    n = 0
    for c in sorted(cand, key=lambda t: (t not in ref, t)):  # exact matches first
        for i, r in enumerate(pool):
            if same(c, r):
                n += 1
                pool.pop(i)
                break
    return n


@register(
    category="text",
    task="captioning",
    name="SPICE",
    definition="F-score between the semantic tuples (objects, attributes, relations) of the candidate caption's "
    "scene graph and the union of the references' scene graphs, with exact or synonym matching; averaged over "
    "images.",
    formula="P = |T(c) ⊗ T(S)| / |T(c)|, R = |T(c) ⊗ T(S)| / |T(S)|, SPICE = 2PR / (P + R)",
    range="[0, 1]",
    input_requirements=("references", "candidates"),
    references=(_REF_SPICE,),
)
@per_item
def spice(
    references: Any,
    candidates: Any,
    *,
    parser: Optional[Callable[[str], Iterable[Any]]] = None,
    synonyms: Union[str, Callable[[str], Iterable[str]], None] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """``references``: per image, the reference tuples (or, with ``parser``, a list of reference captions).
    ``candidates``: per image, the candidate's tuples (or its caption). ``synonyms``: ``None`` (exact match),
    ``"wordnet"`` (shared WordNet synset, as SPICE does) or a callable word -> synonym keys. Precision and recall
    are in ``params``; tuple-type F-scores (object / attribute / relation) too."""
    refs = list(references) if not isinstance(references, (str, bytes)) else []
    cands = list(candidates) if not isinstance(candidates, (str, bytes)) else []
    if not refs or len(refs) != len(cands):
        raise InputValidationError("references and candidates must be non-empty and the same length.")
    if average not in ("mean", None):
        raise InputValidationError("average must be 'mean' or None (per image).")
    syn: Optional[Callable[[str], set[str]]]
    if synonyms is None:
        syn = None
    elif synonyms == "wordnet":
        syn = _wordnet_synonyms()
    elif callable(synonyms):
        user = synonyms
        syn = lambda w: set(user(w))  # noqa: E731
    else:
        raise InputValidationError("synonyms must be None, 'wordnet' or a callable.")
    f1s, ps, rs = [], [], []
    by_type = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}  # matched, candidate, reference
    for i, (r, c) in enumerate(zip(refs, cands)):
        if parser is not None:
            rcaps = [r] if isinstance(r, str) else list(r)
            ref_t = {t for cap in rcaps for t in _tuples(parser(cap), f"parser(references[{i}])")}
            cand_t = set(_tuples(parser(c) if isinstance(c, str) else c, f"parser(candidates[{i}])"))
        else:
            ref_t, cand_t = set(_tuples(r, f"references[{i}]")), set(_tuples(c, f"candidates[{i}]"))
        m = _match_count(cand_t, ref_t, syn)
        p = m / len(cand_t) if cand_t else 0.0
        rc = m / len(ref_t) if ref_t else 0.0
        ps.append(p)
        rs.append(rc)
        f1s.append(0.0 if p + rc == 0 else 2 * p * rc / (p + rc))
        for k in by_type:
            ck = {t for t in cand_t if len(t) == k}
            rk = {t for t in ref_t if len(t) == k}
            by_type[k][0] += _match_count(ck, rk, syn)
            by_type[k][1] += len(ck)
            by_type[k][2] += len(rk)
    f = np.array(f1s)
    type_f = {}
    for k, name in ((1, "object"), (2, "attribute"), (3, "relation")):
        mt, nc, nr = by_type[k]
        p_, r_ = (mt / nc if nc else 0.0), (mt / nr if nr else 0.0)
        type_f[name] = 0.0 if p_ + r_ == 0 else 2 * p_ * r_ / (p_ + r_)
    return MetricResult(
        "spice",
        "SPICE",
        f if average is None else float(f.mean()),
        {
            "precision": float(np.mean(ps)),
            "recall": float(np.mean(rs)),
            "by_type": type_f,
            "synonyms": str(synonyms),
        },
    )
