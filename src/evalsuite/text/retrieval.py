"""Ranked-retrieval metrics: Precision@k, Recall@k, Hit rate@k, MRR, MAP and NDCG@k.

``relevant`` holds, per query, the relevant document ids (a set or list) or graded judgements (a mapping
id -> grade, grade 0 meaning not relevant). ``retrieved`` holds, per query, the ranked list of returned ids,
best first. Scores are averaged over queries, as in trec_eval and ranx.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from .._native import PY_SUM_COMPENSATED, core
from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import _seq, as_mapping_or_set, check_positive_int, per_item

__all__ = ["hit_rate_at_k", "mean_average_precision_at_k", "mrr", "ndcg_at_k", "precision_at_k", "recall_at_k"]

_C = "retrieval"
_REF_MANNING = (
    "Manning CD, Raghavan P, Schütze H. Introduction to Information Retrieval. Cambridge University Press; "
    "2008. Chapter 8."
)
_REF_NDCG = (
    "Järvelin K, Kekäläinen J. Cumulated gain-based evaluation of IR techniques. ACM TOIS. 2002;20(4):422-446."
)
_REF_BURGES = "Burges C, et al. Learning to rank using gradient descent. ICML. 2005:89-96."
_REF_MRR = "Voorhees EM. The TREC-8 question answering track report. TREC. 1999."


def _queries(relevant: Any, retrieved: Any) -> list[tuple[Mapping[Any, float], list[Any]]]:
    rel = _seq(relevant, "relevant")
    ret = _seq(retrieved, "retrieved")
    if len(rel) != len(ret):
        raise InputValidationError(
            f"relevant and retrieved must have one entry per query. Received {len(rel)} and {len(ret)}."
        )
    out = []
    for i, (r, d) in enumerate(zip(rel, ret)):
        judged = as_mapping_or_set(r, f"relevant[{i}]")
        if isinstance(d, (str, bytes)) or not hasattr(d, "__iter__"):
            raise InputValidationError(f"retrieved[{i}] must be a ranked list of ids.")
        ranking = list(d)
        if len(set(map(_hashable, ranking))) != len(ranking):
            raise InputValidationError(f"retrieved[{i}] lists the same id more than once.")
        out.append((judged, ranking))
    return out


def _hashable(x: Any) -> Any:
    return x.item() if isinstance(x, np.generic) else x


def _k(k: Optional[int]) -> Optional[int]:
    return None if k is None else check_positive_int(k, "k")


def _finish(
    metric: str, name: str, per_query: list[float], k: Optional[int], average: Optional[str], **extra: Any
) -> MetricResult:
    arr = np.array(per_query, dtype=float)
    value: Any = arr if average is None else float(np.nanmean(arr)) if arr.size else math.nan
    return MetricResult(metric, name, value, {"k": k, "n_queries": len(per_query), **extra})


def _need_relevant(q: list[tuple[Mapping[Any, float], list[Any]]], what: str) -> None:
    _raise_empty([i for i, (rel, _) in enumerate(q) if not rel], what)


def _raise_empty(empty: list[int], what: str) -> None:
    if empty:
        raise InputValidationError(
            f"{what} is undefined for queries without relevant documents (queries {empty[:5]}); "
            "remove them or judge at least one document relevant."
        )


@register(
    category=_C,
    task="retrieval",
    name="Precision@k",
    definition="Share of the top k retrieved documents that are relevant (k in the denominator even when fewer "
    "are returned), averaged over queries.",
    formula="|relevant ∩ top_k| / k",
    range="[0, 1]",
    input_requirements=("relevant", "retrieved"),
    references=(_REF_MANNING,),
)
def _native_scores(
    kind: str, relevant: Any, retrieved: Any, k: Optional[int], what: str = "", gains: str = "linear"
) -> Optional[list[float]]:
    """Per-query scores from the compiled core, which validates and looks up ids with Python's own dict / set
    semantics and repeats the arithmetic below operation by operation (bit-identical); None without the core."""
    c = core
    if c is None:
        return None
    rel, ret = _seq(relevant, "relevant"), _seq(retrieved, "retrieved")
    err, scores, empty = c.retrieval_scores(kind, rel, ret, k, gains == "exponential", PY_SUM_COMPENSATED)
    if err is not None:
        raise InputValidationError(err)
    if what:
        _raise_empty(empty, what)
    scores_list: list[float] = scores
    return scores_list


@per_item
def precision_at_k(relevant: Any, retrieved: Any, *, k: int = 10, average: Optional[str] = "mean") -> MetricResult:
    """Mean Precision@k over queries."""
    k = check_positive_int(k, "k")
    scores = _native_scores("precision", relevant, retrieved, k)
    if scores is None:
        scores = [sum(d in rel for d in run[:k]) / k for rel, run in _queries(relevant, retrieved)]
    return _finish("precision_at_k", f"Precision@{k}", scores, k, average)


@register(
    category=_C,
    task="retrieval",
    name="Recall@k",
    definition="Share of all relevant documents that appear in the top k, averaged over queries.",
    formula="|relevant ∩ top_k| / |relevant|",
    range="[0, 1]",
    input_requirements=("relevant", "retrieved"),
    references=(_REF_MANNING,),
)
@per_item
def recall_at_k(relevant: Any, retrieved: Any, *, k: int = 10, average: Optional[str] = "mean") -> MetricResult:
    """Mean Recall@k over queries (every query needs at least one relevant document)."""
    k = check_positive_int(k, "k")
    scores = _native_scores("recall", relevant, retrieved, k, "Recall@k")
    if scores is None:
        q = _queries(relevant, retrieved)
        _need_relevant(q, "Recall@k")
        scores = [sum(d in rel for d in run[:k]) / len(rel) for rel, run in q]
    return _finish("recall_at_k", f"Recall@{k}", scores, k, average)


@register(
    category=_C,
    task="retrieval",
    name="Hit rate@k",
    definition="Share of queries with at least one relevant document in the top k.",
    formula="mean_q [|relevant ∩ top_k| > 0]",
    range="[0, 1]",
    input_requirements=("relevant", "retrieved"),
    references=(_REF_MANNING,),
)
@per_item
def hit_rate_at_k(relevant: Any, retrieved: Any, *, k: int = 10, average: Optional[str] = "mean") -> MetricResult:
    """Hit rate@k (success@k)."""
    k = check_positive_int(k, "k")
    scores = _native_scores("hit", relevant, retrieved, k)
    if scores is None:
        scores = [float(any(d in rel for d in run[:k])) for rel, run in _queries(relevant, retrieved)]
    return _finish("hit_rate_at_k", f"Hit rate@{k}", scores, k, average)


@register(
    category=_C,
    task="retrieval",
    name="Mean reciprocal rank",
    definition="Average over queries of 1 / rank of the first relevant document (0 when none is retrieved "
    "within the cutoff).",
    formula="MRR = mean_q 1 / rank_q",
    range="[0, 1]",
    input_requirements=("relevant", "retrieved"),
    references=(_REF_MRR,),
)
@per_item
def mrr(
    relevant: Any, retrieved: Any, *, k: Optional[int] = None, average: Optional[str] = "mean"
) -> MetricResult:
    """Mean reciprocal rank, optionally only within the top ``k``."""
    k = _k(k)
    native = _native_scores("mrr", relevant, retrieved, k)
    if native is not None:
        return _finish("mrr", "MRR" if k is None else f"MRR@{k}", native, k, average)
    scores = []
    for rel, run in _queries(relevant, retrieved):
        rr = 0.0
        for i, d in enumerate(run[:k] if k else run, 1):
            if d in rel:
                rr = 1 / i
                break
        scores.append(rr)
    return _finish("mrr", "MRR" if k is None else f"MRR@{k}", scores, k, average)


@register(
    category=_C,
    task="retrieval",
    name="Mean average precision",
    definition="For each query, the mean of Precision@i over the ranks i of relevant retrieved documents, "
    "divided by the number of relevant documents (unretrieved ones count as 0); averaged over queries "
    "(trec_eval convention).",
    formula="AP = (1/|R|) Σ_i P@i · rel_i ; MAP = mean_q AP_q",
    range="[0, 1]",
    input_requirements=("relevant", "retrieved"),
    references=(_REF_MANNING,),
)
@per_item
def mean_average_precision_at_k(
    relevant: Any, retrieved: Any, *, k: Optional[int] = None, average: Optional[str] = "mean"
) -> MetricResult:
    """MAP (``k=None``) or MAP@k: only the top ``k`` documents count, the denominator stays |relevant|."""
    k = _k(k)
    native = _native_scores("map", relevant, retrieved, k, "Average precision")
    if native is not None:
        return _finish("mean_average_precision_at_k", "MAP" if k is None else f"MAP@{k}", native, k, average)
    q = _queries(relevant, retrieved)
    _need_relevant(q, "Average precision")
    scores = []
    for rel, run in q:
        hits, total = 0, 0.0
        for i, d in enumerate(run[:k] if k else run, 1):
            if d in rel:
                hits += 1
                total += hits / i
        scores.append(total / len(rel))
    return _finish("mean_average_precision_at_k", "MAP" if k is None else f"MAP@{k}", scores, k, average)


@register(
    category=_C,
    task="retrieval",
    name="NDCG@k",
    definition="Discounted cumulative gain of the top k (graded relevance, log2 rank discount) divided by the "
    "best possible DCG for the query; linear gains (Järvelin & Kekäläinen, trec_eval) or exponential "
    "2^rel − 1 (Burges).",
    formula="DCG@k = Σ_{i≤k} gain(rel_i) / log2(i + 1); NDCG@k = DCG@k / IDCG@k",
    range="[0, 1]",
    input_requirements=("relevant", "retrieved"),
    references=(_REF_NDCG, _REF_BURGES),
)
@per_item
def ndcg_at_k(
    relevant: Any,
    retrieved: Any,
    *,
    k: Optional[int] = 10,
    gains: str = "linear",
    average: Optional[str] = "mean",
) -> MetricResult:
    """NDCG@k. ``gains="linear"`` uses the grade itself, ``"exponential"`` uses 2^grade − 1."""
    k = _k(k)
    if gains not in ("linear", "exponential"):
        raise InputValidationError("gains must be 'linear' or 'exponential'.")
    native = _native_scores("ndcg", relevant, retrieved, k, "NDCG", gains)
    if native is not None:
        return _finish("ndcg_at_k", "NDCG" if k is None else f"NDCG@{k}", native, k, average, gains=gains)
    q = _queries(relevant, retrieved)
    _need_relevant(q, "NDCG")

    def g(x: float) -> float:
        return x if gains == "linear" else 2.0**x - 1

    scores = []
    for rel, run in q:
        top = run[:k] if k else run
        dcg = sum(g(rel.get(d, 0.0)) / math.log2(i + 1) for i, d in enumerate(top, 1))
        ideal = sorted(rel.values(), reverse=True)[: k if k else None]
        idcg = sum(g(x) / math.log2(i + 1) for i, x in enumerate(ideal, 1))
        scores.append(dcg / idcg)
    return _finish("ndcg_at_k", "NDCG" if k is None else f"NDCG@{k}", scores, k, average, gains=gains)
