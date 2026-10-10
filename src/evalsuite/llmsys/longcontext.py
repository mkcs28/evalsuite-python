"""Long-context and summarization evaluation (v0.5.0).

ROUGE-L and BERTScore for summaries are the existing ``es.rouge_l`` and ``es.bertscore``; summary factual
consistency can also be measured with ``es.faithfulness`` over claim verdicts. Truncation robustness is
``es.truncation_sensitivity``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import bools, floats, lists_of, rate, same_length, seq

__all__ = [
    "citation_coverage",
    "compression_ratio",
    "context_utilization",
    "cross_document_consistency",
    "lost_in_the_middle",
    "needle_in_haystack",
    "position_accuracy",
    "retrieval_accuracy_by_length",
    "summary_coverage",
]

_C = "long_context"
_REF_NIAH = (
    "Kamradt G. Needle in a haystack: pressure testing LLMs. GitHub repository "
    "gkamradt/LLMTest_NeedleInAHaystack. 2023."
)
_REF_RULER = (
    "Hsieh CP, Sun S, Kriman S, et al. RULER: what's the real context size of your long-context language models? "
    "COLM. 2024."
)
_REF_LITM = (
    "Liu NF, Lin K, Hewitt J, et al. Lost in the middle: how language models use long contexts. TACL. "
    "2024;12:157-173."
)
_REF_LONGBENCH = (
    "Bai Y, Lv X, Zhang J, et al. LongBench: a bilingual, multitask benchmark for long context understanding. "
    "ACL. 2024:3119-3137."
)
_REF_FACTSCORE = (
    "Min S, Krishna K, Lyu X, et al. FActScore: fine-grained atomic evaluation of factual precision in long form "
    "text generation. EMNLP. 2023:12076-12100."
)
_REF_PYRAMID = (
    "Nenkova A, Passonneau R. Evaluating content selection in summarization: the pyramid method. HLT-NAACL. "
    "2004:145-152."
)
_REF_GRUSKY = (
    "Grusky M, Naaman M, Artzi Y. Newsroom: a dataset of 1.3 million summaries with diverse extractive "
    "strategies. "
    "NAACL. 2018:708-719."
)
_REF_ALCE = (
    "Gao T, Yen H, Yu J, Chen D. Enabling large language models to generate text with citations. EMNLP. "
    "2023:6465-6488."
)


def _bin_accuracy(c: np.ndarray, x: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & ((x < hi) if hi != edges[-1] else (x <= hi))
        if m.any():
            out[f"{lo:g}-{hi:g}"] = float(c[m].mean())
    return out


@register(
    category=_C,
    task="long-context",
    name="Long-context retrieval accuracy",
    definition="Accuracy of retrieving or answering from long contexts, overall and per context-length level "
    "(RULER reports the accuracy at each length; the effective length is the longest one above a threshold).",
    formula="acc_L = mean[correct | length = L]",
    range="[0, 1]",
    input_requirements=("correct", "context_lengths"),
    references=(_REF_RULER, _REF_LONGBENCH),
)
def retrieval_accuracy_by_length(correct: Any, context_lengths: Any, *, threshold: float = 0.85) -> MetricResult:
    c = bools(correct, "correct")
    L = floats(context_lengths, "context_lengths", lo=0)
    same_length(("correct", c), ("context_lengths", L))
    levels = np.unique(L)
    acc = {float(v): float(c[v == L].mean()) for v in levels}
    ok = [v for v in levels if acc[float(v)] >= threshold]
    effective = None
    for v in levels:  # longest length such that every shorter length also passes
        if acc[float(v)] >= threshold:
            effective = float(v)
        else:
            break
    return MetricResult(
        "retrieval_accuracy_by_length",
        "Long-context retrieval accuracy",
        float(c.mean()),
        {
            "by_length": {f"{k:g}": a for k, a in acc.items()},
            "effective_length": effective,
            "threshold": threshold,
            "n_passing_lengths": len(ok),
        },
    )


@register(
    category=_C,
    task="long-context",
    name="Needle-in-a-haystack accuracy",
    definition="Retrieval of a planted fact (the needle) from contexts of varying length and insertion depth: "
    "overall accuracy and the length × depth accuracy grid (the usual heat map).",
    formula="acc = mean[correct]; grid[L, d] = mean[correct | L, d]",
    range="[0, 1]",
    input_requirements=("correct", "context_lengths", "depths"),
    references=(_REF_NIAH, _REF_RULER),
)
def needle_in_haystack(correct: Any, context_lengths: Any, depths: Any) -> MetricResult:
    """``depths``: where the needle was placed, as a fraction of the context in [0, 1]."""
    c = floats(np.asarray(seq(correct, "correct"), dtype=float), "correct", lo=0, hi=1)
    L = floats(context_lengths, "context_lengths", lo=0)
    d = floats(depths, "depths", lo=0, hi=1)
    same_length(("correct", c), ("context_lengths", L), ("depths", d))
    ls, ds = np.unique(L), np.unique(d)
    grid = np.full((ls.size, ds.size), np.nan)
    for i, lv in enumerate(ls):
        for j, dv in enumerate(ds):
            m = (lv == L) & (d == dv)
            if m.any():
                grid[i, j] = c[m].mean()
    return MetricResult(
        "needle_in_haystack",
        "Needle-in-a-haystack accuracy",
        float(c.mean()),
        {
            "lengths": ls.tolist(),
            "depths": ds.tolist(),
            "grid": grid.tolist(),
            "worst_cell": float(np.nanmin(grid)),
            "by_length": {f"{v:g}": float(c[v == L].mean()) for v in ls},
            "by_depth": {f"{v:g}": float(c[d == v].mean()) for v in ds},
        },
    )


@register(
    category=_C,
    task="long-context",
    name="Position-dependent retrieval accuracy",
    definition="Accuracy as a function of where the relevant information sits in the context (relative position "
    "binned into equal-width bins), with the spread between the best and worst bin.",
    formula="acc_b = mean[correct | position ∈ bin b]; spread = max_b − min_b",
    range="[0, 1]",
    input_requirements=("correct", "positions"),
    references=(_REF_LITM,),
)
def position_accuracy(correct: Any, positions: Any, *, n_bins: int = 5) -> MetricResult:
    """``positions``: relative position of the relevant passage in [0, 1] (0 = start)."""
    c = floats(np.asarray(seq(correct, "correct"), dtype=float), "correct", lo=0, hi=1)
    p = floats(positions, "positions", lo=0, hi=1)
    same_length(("correct", c), ("positions", p))
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise InputValidationError("n_bins must be an integer >= 2.")
    by = _bin_accuracy(c, p, np.linspace(0, 1, n_bins + 1))
    vals = list(by.values())
    return MetricResult(
        "position_accuracy",
        "Position spread (best − worst bin)",
        float(max(vals) - min(vals)),
        {"by_position": by, "accuracy": float(c.mean())},
    )


@register(
    category=_C,
    task="long-context",
    name="Lost-in-the-middle sensitivity",
    definition="How much worse the model does when the relevant passage is in the middle of the context than at "
    "its edges: mean accuracy at the start and end positions minus accuracy in the middle (Liu et al. U-curve).",
    formula="(acc_start + acc_end)/2 − acc_middle",
    range="[−1, 1] (0 = position-invariant)",
    input_requirements=("correct", "positions"),
    references=(_REF_LITM,),
    higher_is_better=False,
)
def lost_in_the_middle(correct: Any, positions: Any, *, edge: float = 0.2) -> MetricResult:
    c = floats(np.asarray(seq(correct, "correct"), dtype=float), "correct", lo=0, hi=1)
    p = floats(positions, "positions", lo=0, hi=1)
    same_length(("correct", c), ("positions", p))
    if not 0 < edge < 0.5:
        raise InputValidationError("edge must be in (0, 0.5).")
    start, end = p <= edge, p >= 1 - edge
    mid = ~start & ~end
    if not (start.any() and end.any() and mid.any()):
        raise InputValidationError("Need examples at the start, the middle and the end of the context.")
    a_s, a_e, a_m = float(c[start].mean()), float(c[end].mean()), float(c[mid].mean())
    return MetricResult(
        "lost_in_the_middle",
        "Lost-in-the-middle gap",
        (a_s + a_e) / 2 - a_m,
        {"start_accuracy": a_s, "middle_accuracy": a_m, "end_accuracy": a_e, "edge": edge},
    )


@register(
    category=_C,
    task="long-context",
    name="Context utilization / retention",
    definition="Share of the relevant facts provided in the context that the output actually uses (or recalls "
    "later in a conversation), pooled over examples.",
    formula="Σ |used ∩ provided| / Σ |provided|",
    range="[0, 1]",
    input_requirements=("used_facts", "provided_facts"),
    references=(_REF_LONGBENCH,),
)
def context_utilization(used_facts: Any, provided_facts: Any) -> MetricResult:
    used, prov = lists_of(used_facts, "used_facts"), lists_of(provided_facts, "provided_facts")
    same_length(("used_facts", used), ("provided_facts", prov))
    hit = tot = 0
    per = []
    for u, p in zip(used, prov):
        ps, us = set(map(str, p)), set(map(str, u))
        if not ps:
            continue
        hit += len(ps & us)
        tot += len(ps)
        per.append(len(ps & us) / len(ps))
    if tot == 0:
        raise InputValidationError("No example provides any fact.")
    return MetricResult(
        "context_utilization", "Context utilization", hit / tot, {"mean_per_example": float(np.mean(per))}
    )


@register(
    category=_C,
    task="summarization",
    name="Summary factual consistency / coverage / completeness",
    definition="Coverage: share of the reference key points (pyramid content units) the summary covers, weighted "
    "by importance when weights are given; factual consistency (share of summary claims supported by the "
    "source) is reported when claim verdicts are passed.",
    formula="coverage = Σ w·covered / Σ w; consistency = supported claims / claims",
    range="[0, 1]",
    input_requirements=("key_points_covered",),
    references=(_REF_PYRAMID, _REF_FACTSCORE),
)
def summary_coverage(
    key_points_covered: Any, *, weights: Any = None, claims_supported: Any = None
) -> MetricResult:
    """``key_points_covered``: per summary, one boolean per reference key point. ``weights``: per summary, the
    importance of each key point (pyramid weights). ``claims_supported``: per summary, one boolean per claim."""
    rows = lists_of(key_points_covered, "key_points_covered")
    ws = lists_of(weights, "weights") if weights is not None else [[1.0] * len(r) for r in rows]
    same_length(("key_points_covered", rows), ("weights", ws))
    per = []
    for i, (r, w) in enumerate(zip(rows, ws)):
        if len(r) != len(w) or not r:
            raise InputValidationError(
                f"Summary {i}: key points and weights must be non-empty and the same length."
            )
        wa = floats(w, f"weights[{i}]", lo=0)
        per.append(float(np.dot(wa, [bool(v) for v in r]) / wa.sum()) if wa.sum() else 0.0)
    extra: dict[str, Any] = {}
    if claims_supported is not None:
        cl = lists_of(claims_supported, "claims_supported")
        same_length(("key_points_covered", rows), ("claims_supported", cl))
        flat = [bool(v) for r in cl for v in r]
        extra["factual_consistency"] = float(np.mean(flat)) if flat else float("nan")
        extra["fully_consistent_rate"] = float(np.mean([all(map(bool, r)) for r in cl]))
    return MetricResult("summary_coverage", "Summary coverage", float(np.mean(per)), extra)


@register(
    category=_C,
    task="summarization",
    name="Summary compression ratio",
    definition="How much shorter summaries are than their sources: source words / summary words (Newsroom "
    "compression), the mean over documents, with its median.",
    formula="mean_d |source_d| / |summary_d|",
    range="[1, ∞) typically",
    input_requirements=("sources", "summaries"),
    references=(_REF_GRUSKY,),
    higher_is_better=None,
)
def compression_ratio(
    sources: Any, summaries: Any, *, tokenizer: Optional[Callable[[str], list[str]]] = None
) -> MetricResult:
    src, summ = seq(sources, "sources"), seq(summaries, "summaries")
    same_length(("sources", src), ("summaries", summ))
    tok = tokenizer or str.split
    ratios = []
    for i, (a, b) in enumerate(zip(src, summ)):
        if not isinstance(a, str) or not isinstance(b, str):
            raise InputValidationError(f"sources[{i}] and summaries[{i}] must be strings.")
        nb = len(tok(b))
        if nb == 0:
            raise InputValidationError(f"summaries[{i}] is empty.")
        ratios.append(len(tok(a)) / nb)
    r = np.array(ratios)
    return MetricResult("compression_ratio", "Compression ratio", float(r.mean()), {"median": float(np.median(r))})


@register(
    category=_C,
    task="summarization",
    name="Citation coverage",
    definition="Share of output statements that carry at least one citation (ALCE: statements that should be "
    "attributable must cite something), pooled over outputs.",
    formula="statements with ≥1 citation / statements",
    range="[0, 1]",
    input_requirements=("citations_per_statement",),
    references=(_REF_ALCE,),
)
def citation_coverage(citations_per_statement: Any) -> MetricResult:
    """``citations_per_statement``: per output, the number of citations attached to each statement."""
    rows = lists_of(citations_per_statement, "citations_per_statement")
    flat = np.array([float(v) > 0 for r in rows for v in r])
    if flat.size == 0:
        raise InputValidationError("No statements given.")
    per = [np.mean([float(v) > 0 for v in r]) for r in rows if r]
    return rate("citation_coverage", "Citation coverage", flat, {"mean_per_output": float(np.mean(per))})


@register(
    category=_C,
    task="long-context",
    name="Cross-document consistency",
    definition="Agreement of answers to the same question across different (orderings or subsets of) documents "
    "or truncation levels: share of questions answered identically and mean pairwise agreement.",
    formula="mean_q 1[all answers equal]",
    range="[0, 1]",
    input_requirements=("answers",),
    references=(_REF_LITM, _REF_LONGBENCH),
)
def cross_document_consistency(answers: Any, *, normalize: Optional[Callable[[Any], Any]] = None) -> MetricResult:
    from itertools import combinations

    rows = seq(answers, "answers")
    full, pair = [], []
    for i, r in enumerate(rows):
        vals = list(r.values()) if isinstance(r, Mapping) else list(r) if isinstance(r, (list, tuple)) else None
        if vals is None or len(vals) < 2:
            raise InputValidationError(f"answers[{i}] needs at least two answers.")
        v = [
            normalize(a) if normalize else (" ".join(a.lower().split()) if isinstance(a, str) else a) for a in vals
        ]
        full.append(all(x == v[0] for x in v))
        pair.append(np.mean([x == y for x, y in combinations(v, 2)]))
    return MetricResult(
        "cross_document_consistency",
        "Cross-document consistency",
        float(np.mean(full)),
        {"pairwise_agreement": float(np.mean(pair))},
    )
