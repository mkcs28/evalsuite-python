"""Retrieval-augmented generation beyond ranked retrieval: context precision, recall and relevance, retrieval
latency, end-to-end task success and failure attribution.

Retrieval ranking metrics (Precision@k, Recall@k, Hit rate@k, MRR, MAP, NDCG) live in
``evalsuite.text.retrieval``; answer-level faithfulness, correctness and citations in
``evalsuite.text.factuality``. The functions here take per-chunk or per-claim judgements that a human, a
retrieval check or an LLM judge produced.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import per_item

__all__ = [
    "context_precision",
    "context_recall",
    "context_relevance",
    "failure_attribution",
    "latency_summary",
    "task_success_rate",
]

_C = "rag"
_REF_RAGAS = (
    "Es S, James J, Espinosa-Anke L, Schockaert S. RAGAS: automated evaluation of retrieval augmented "
    "generation. EACL (demos). 2024:150-158."
)
_REF_ARES = (
    "Saad-Falcon J, Khattab O, Potts C, Zaharia M. ARES: an automated evaluation framework for retrieval-"
    "augmented generation systems. NAACL. 2024:338-354."
)
_REF_RAGCHECKER = (
    "Ru D, Qiu L, Hu X, et al. RAGChecker: a fine-grained framework for diagnosing retrieval-augmented "
    "generation. NeurIPS Datasets and Benchmarks. 2024."
)
_REF_WILSON = (
    "Wilson EB. Probable inference, the law of succession, and statistical inference. JASA. 1927;22:209-212."
)


def _flag_lists(x: Any, name: str) -> list[list[bool]]:
    rows = list(x) if not isinstance(x, (str, bytes)) else []
    if not rows:
        raise InputValidationError(f"{name} must be a non-empty list with one list of booleans per query.")
    return [[bool(v) for v in r] for r in rows]


def _finish(
    metric: str, name: str, per: np.ndarray, average: Optional[str], extra: dict[str, Any]
) -> MetricResult:
    if average not in ("mean", None):
        raise InputValidationError("average must be 'mean' or None (per query).")
    keep = ~np.isnan(per)
    if not keep.any():
        raise InputValidationError(f"{name} is undefined for every query.")
    value: Any = per if average is None else float(per[keep].mean())
    return MetricResult(
        metric, name, value, {"n_queries": int(per.size), "n_undefined": int((~keep).sum()), **extra}
    )


@register(
    category=_C,
    task="rag",
    name="Context precision",
    definition="How well the retriever ranks useful chunks first: the mean of precision@k at the rank of each "
    "relevant chunk, over the retrieved list (RAGAS context precision; average precision over retrieved "
    "chunks).",
    formula="Σ_k (precision@k · rel_k) / Σ_k rel_k",
    range="[0, 1]",
    input_requirements=("chunk_relevance",),
    references=(_REF_RAGAS,),
)
@per_item
def context_precision(chunk_relevance: Any, *, average: Optional[str] = "mean") -> MetricResult:
    """``chunk_relevance``: per query, the relevance verdict of each retrieved chunk in rank order. Queries with
    no relevant chunk score 0 (nothing useful was retrieved)."""
    rows = _flag_lists(chunk_relevance, "chunk_relevance")
    per = np.empty(len(rows))
    for i, r in enumerate(rows):
        a = np.asarray(r, dtype=float)
        if a.size == 0 or a.sum() == 0:
            per[i] = 0.0
            continue
        prec = np.cumsum(a) / np.arange(1, a.size + 1)
        per[i] = float((prec * a).sum() / a.sum())
    return _finish("context_precision", "Context precision", per, average, {})


@register(
    category=_C,
    task="rag",
    name="Context recall",
    definition="Share of the reference answer's claims that can be attributed to the retrieved context (RAGAS "
    "context recall): did retrieval find the evidence needed?",
    formula="reference claims supported by the context / reference claims",
    range="[0, 1]",
    input_requirements=("claim_attributed",),
    references=(_REF_RAGAS, _REF_RAGCHECKER),
)
@per_item
def context_recall(claim_attributed: Any, *, average: Optional[str] = "mean") -> MetricResult:
    """``claim_attributed``: per query, one boolean per claim of the reference answer (supported by the
    retrieved context or not). Queries whose reference has no claims are undefined (NaN) and skipped."""
    rows = _flag_lists(claim_attributed, "claim_attributed")
    per = np.array([np.mean(r) if r else np.nan for r in rows], dtype=float)
    return _finish("context_recall", "Context recall", per, average, {})


@register(
    category=_C,
    task="rag",
    name="Context relevance",
    definition="Share of the retrieved context that is relevant to the question (per chunk or per sentence), "
    "a measure of retrieval noise (ARES context relevance).",
    formula="relevant units / retrieved units",
    range="[0, 1]",
    input_requirements=("unit_relevance",),
    references=(_REF_ARES,),
)
@per_item
def context_relevance(unit_relevance: Any, *, average: Optional[str] = "mean") -> MetricResult:
    """``unit_relevance``: per query, one boolean per retrieved chunk or sentence."""
    rows = _flag_lists(unit_relevance, "unit_relevance")
    per = np.array([np.mean(r) if r else np.nan for r in rows], dtype=float)
    return _finish("context_relevance", "Context relevance", per, average, {})


@register(
    category=_C,
    task="operations",
    name="Latency summary",
    definition="Distribution of a latency (retrieval, time to first token, end-to-end): mean and the 50th, "
    "90th, 95th and 99th percentiles (linear interpolation), in the input's unit.",
    formula="mean and percentiles of the observed latencies",
    range="[0, ∞)",
    input_requirements=("latencies",),
    references=("Dean J, Barroso LA. The tail at scale. Commun ACM. 2013;56(2):74-80.",),
    higher_is_better=False,
)
def latency_summary(latencies: Any, *, statistic: str = "p95", unit: str = "ms") -> MetricResult:
    """Latency summary; ``statistic`` (``"mean"``, ``"p50"``, ``"p90"``, ``"p95"`` or ``"p99"``) is the
    headline value and all of them are in ``params``."""
    a = np.asarray(latencies, dtype=float).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise InputValidationError("latencies must be non-empty, finite and non-negative.")
    summary = {
        "mean": float(a.mean()),
        **{f"p{q}": float(np.percentile(a, q)) for q in (50, 90, 95, 99)},
        "max": float(a.max()),
    }
    if statistic not in summary or statistic == "max":
        raise InputValidationError("statistic must be 'mean', 'p50', 'p90', 'p95' or 'p99'.")
    return MetricResult(
        "latency_summary", f"Latency {statistic}", summary[statistic], {"unit": unit, "n": int(a.size), **summary}
    )


def _wilson(k: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


@register(
    category=_C,
    task="rag",
    name="Task success rate",
    definition="Share of requests the whole system resolved to specification (end-to-end query resolution, "
    "tool-assisted task success), with a Wilson 95% interval.",
    formula="successful tasks / tasks",
    range="[0, 1]",
    input_requirements=("success",),
    references=(_REF_WILSON,),
)
@per_item
def task_success_rate(success: Any) -> MetricResult:
    """Success rate from one boolean per task."""
    a = np.asarray(success, dtype=bool).ravel()
    if a.size == 0:
        raise InputValidationError("success is empty.")
    low, high = _wilson(float(a.sum()), int(a.size))
    return MetricResult(
        "task_success_rate",
        "Task success rate",
        float(a.mean()),
        {"n": int(a.size), "ci_low": low, "ci_high": high},
    )


_STAGES = ("retrieval", "generation", "evidence", "orchestration", "other")


@register(
    category=_C,
    task="rag",
    name="Failure attribution",
    definition="Distribution of failed requests over the stage that caused them: retrieval (evidence not "
    "found), evidence (found but insufficient or conflicting), generation (evidence ignored or misused), "
    "orchestration (tools, timeouts, routing) or other.",
    formula="failures attributed to the stage / failures",
    range="[0, 1] per stage",
    input_requirements=("stages",),
    references=(_REF_RAGCHECKER,),
    higher_is_better=None,
)
def failure_attribution(stages: Any, *, stage_names: tuple[str, ...] = _STAGES) -> MetricResult:
    """``stages``: the attributed stage of each failed request (``None`` or ``"none"`` for successes, which are
    ignored). Returns the share per stage, aligned with ``labels``."""
    names = tuple(stage_names)
    vals = [None if s is None else str(s).strip().lower() for s in stages]
    failed = [s for s in vals if s not in (None, "none", "success", "")]
    unknown = sorted({s for s in failed if s not in names})
    if unknown:
        raise InputValidationError(f"Unknown stage(s) {unknown}; expected one of {list(names)}.")
    if not failed:
        raise InputValidationError("There are no failures to attribute.")
    counts = Counter(failed)
    share = np.array([counts.get(n, 0) / len(failed) for n in names])
    return MetricResult(
        "failure_attribution",
        "Failure attribution",
        share,
        {"n_failures": len(failed), "n_requests": len(vals), "counts": dict(counts)},
        labels=names,
    )
