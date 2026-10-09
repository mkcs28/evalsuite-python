"""Factuality, hallucination, citation and abstention metrics.

These metrics score *verdicts* that a verification step produced: a human annotator, an NLI model, a
retrieval check or an LLM judge decides, for each claim of an answer, whether the evidence supports it.
EvalSuite aggregates those verdicts consistently (micro or macro), so different verification pipelines can be
compared with intervals and paired tests. How claims are extracted and judged must be reported alongside.

Claim verdicts may be booleans (``True`` = supported) or the strings ``"supported"``, ``"contradicted"`` and
``"unsupported"`` (also ``"not enough info"`` / ``"nei"``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import per_item

__all__ = [
    "abstention_accuracy",
    "answer_correctness",
    "answer_relevance",
    "citation_precision",
    "citation_recall",
    "claim_verification_accuracy",
    "faithfulness",
    "groundedness",
    "hallucination_rate",
    "knowledge_consistency",
]

_C = "text"
_REF_FACTSCORE = (
    "Min S, Krishna K, Lyu X, et al. FActScore: fine-grained atomic evaluation of factual precision in long "
    "form text generation. EMNLP. 2023:12076-12100."
)
_REF_RAGAS = (
    "Es S, James J, Espinosa-Anke L, Schockaert S. RAGAS: automated evaluation of retrieval augmented "
    "generation. EACL (demos). 2024:150-158."
)
_REF_ALCE = (
    "Gao T, Yen H, Yu J, Chen D. Enabling large language models to generate text with citations. EMNLP. "
    "2023:6465-6488."
)
_REF_FEVER = (
    "Thorne J, Vlachos A, Christodoulopoulos C, Mittal A. FEVER: a large-scale dataset for fact extraction and "
    "VERification. NAACL. 2018:809-819."
)
_REF_ABSTAIN = (
    "Feng S, Shi W, Wang Y, Ding W, Balachandran V, Tsvetkov Y. Don't hallucinate, abstain: identifying LLM "
    "knowledge gaps via multi-LLM collaboration. ACL. 2024:14664-14690."
)

_SUPPORTED = {"supported", "support", "entailed", "entailment", "true", "yes"}
_CONTRA = {"contradicted", "contradiction", "refuted", "false"}
_UNSUP = {"unsupported", "not enough info", "nei", "neutral", "unknown", "no"}


def _verdict(v: Any, where: str) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "supported" if v else "unsupported"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _SUPPORTED:
            return "supported"
        if s in _CONTRA:
            return "contradicted"
        if s in _UNSUP:
            return "unsupported"
    raise InputValidationError(
        f"{where}: verdict must be a boolean or one of 'supported', 'contradicted', 'unsupported'; got {v!r}."
    )


def _claims(claim_verdicts: Any) -> list[list[str]]:
    if claim_verdicts is None or isinstance(claim_verdicts, (str, bytes)):
        raise InputValidationError("claim_verdicts must be a list with one list of claim verdicts per answer.")
    answers = list(claim_verdicts)
    if not answers:
        raise InputValidationError("claim_verdicts is empty.")
    out = []
    for i, a in enumerate(answers):
        if isinstance(a, (str, bytes, bool)) or not hasattr(a, "__iter__"):
            raise InputValidationError(f"claim_verdicts[{i}] must be a list of verdicts (one per claim).")
        out.append([_verdict(v, f"claim_verdicts[{i}][{j}]") for j, v in enumerate(a)])
    return out


def _aggregate(
    metric: str, name: str, hits: list[int], totals: list[int], average: Optional[str], extra: dict[str, Any]
) -> MetricResult:
    hits_a, totals_a = np.asarray(hits, float), np.asarray(totals, float)
    if average == "micro":
        if totals_a.sum() == 0:
            raise InputValidationError(f"{name} is undefined: there are no claims to score.")
        value: Any = float(np.sum(hits_a)) / float(np.sum(totals_a))
    elif average in ("macro", None):
        keep = totals_a > 0
        if not keep.any():
            raise InputValidationError(f"{name} is undefined: every answer has zero claims.")
        per = np.full(totals_a.shape, np.nan)
        per[keep] = hits_a[keep] / totals_a[keep]
        value = per if average is None else float(np.nanmean(per))
    else:
        raise InputValidationError("average must be 'micro' (pool all claims), 'macro' (mean per answer) or None.")
    return MetricResult(
        metric,
        name,
        value,
        {"average": average, "n_answers": len(totals), "n_claims": int(totals_a.sum()), **extra},
    )


# ---------------------------------------------------------------- claims
@register(
    category=_C,
    task="factuality",
    name="Faithfulness",
    definition="Share of an answer's claims that the evidence supports (FActScore factual precision; RAGAS "
    "faithfulness when the evidence is the retrieved context).",
    formula="supported claims / claims",
    range="[0, 1]",
    input_requirements=("claim_verdicts",),
    references=(_REF_FACTSCORE, _REF_RAGAS),
)
@per_item
def faithfulness(claim_verdicts: Any, *, average: Optional[str] = "macro") -> MetricResult:
    """Faithfulness / factual precision from per-claim verdicts (``average="macro"`` per answer, as FActScore
    and RAGAS; ``"micro"`` pools all claims). Answers without claims are skipped in the macro mean."""
    claims = _claims(claim_verdicts)
    return _aggregate(
        "faithfulness",
        "Faithfulness",
        [sum(v == "supported" for v in c) for c in claims],
        [len(c) for c in claims],
        average,
        {},
    )


@register(
    category=_C,
    task="factuality",
    name="Hallucination rate (unsupported-claim rate)",
    definition="Share of an answer's claims that the evidence does not support, i.e. contradicted or "
    "unverifiable claims, under the stated verification protocol.",
    formula="(contradicted + unsupported claims) / claims",
    range="[0, 1]",
    input_requirements=("claim_verdicts",),
    references=(_REF_FACTSCORE,),
    higher_is_better=False,
)
@per_item
def hallucination_rate(
    claim_verdicts: Any, *, count: str = "unsupported", average: Optional[str] = "micro"
) -> MetricResult:
    """Unsupported-claim rate. ``count="unsupported"`` counts every claim that is not supported;
    ``"contradicted"`` counts only claims the evidence contradicts (an intrinsic-hallucination rate)."""
    if count not in ("unsupported", "contradicted"):
        raise InputValidationError("count must be 'unsupported' or 'contradicted'.")
    claims = _claims(claim_verdicts)
    if count == "unsupported":
        hits = [sum(v != "supported" for v in c) for c in claims]
    else:
        hits = [sum(v == "contradicted" for v in c) for c in claims]
    return _aggregate(
        "hallucination_rate", "Hallucination rate", hits, [len(c) for c in claims], average, {"count": count}
    )


@register(
    category=_C,
    task="factuality",
    name="Groundedness",
    definition="Average support score of an answer's content given the supplied context, from graded scores in "
    "[0, 1] (e.g. NLI entailment probabilities or judge ratings) rather than binary verdicts.",
    formula="mean over claims (or sentences) of support score",
    range="[0, 1]",
    input_requirements=("support_scores",),
    references=(_REF_RAGAS,),
)
@per_item
def groundedness(support_scores: Any, *, average: Optional[str] = "macro") -> MetricResult:
    """Groundedness from graded support scores, one list of scores in [0, 1] per answer."""
    answers = list(support_scores) if not isinstance(support_scores, (str, bytes)) else []
    if not answers:
        raise InputValidationError("support_scores must be a non-empty list of score lists (one per answer).")
    sums, counts = [], []
    for i, a in enumerate(answers):
        arr = np.asarray(list(a), dtype=float)
        if arr.size and (not np.all(np.isfinite(arr)) or arr.min() < 0 or arr.max() > 1):
            raise InputValidationError(f"support_scores[{i}] must be finite values in [0, 1].")
        sums.append(float(arr.sum()))
        counts.append(int(arr.size))
    if average == "micro":
        value: Any = float(sum(sums) / sum(counts)) if sum(counts) else None
        if value is None:
            raise InputValidationError("Groundedness is undefined: there are no scores.")
        return MetricResult("groundedness", "Groundedness", value, {"average": "micro", "n_answers": len(answers)})
    per = np.array([s / c if c else np.nan for s, c in zip(sums, counts)])
    if np.all(np.isnan(per)):
        raise InputValidationError("Groundedness is undefined: every answer has zero scores.")
    if average not in ("macro", None):
        raise InputValidationError("average must be 'micro', 'macro' or None.")
    value = per if average is None else float(np.nanmean(per))
    return MetricResult("groundedness", "Groundedness", value, {"average": average, "n_answers": len(answers)})


# ---------------------------------------------------------------- citations (ALCE)
def _citation_inputs(citations: Any) -> list[list[tuple[bool, list[bool]]]]:
    """Per answer, per statement: (statement supported by its citations, [each citation relevant])."""
    answers = list(citations) if not isinstance(citations, (str, bytes)) else []
    if not answers:
        raise InputValidationError("citations must be a non-empty list (one list of statements per answer).")
    out = []
    for i, ans in enumerate(answers):
        stmts = []
        for j, st in enumerate(ans):
            if not isinstance(st, dict) or "supported" not in st:
                raise InputValidationError(
                    f"citations[{i}][{j}] must be a dict with 'supported' (bool) and optional 'citations' "
                    "(list of "
                    "bools: does each cited passage support or partly support the statement)."
                )
            cites = [bool(c) for c in st.get("citations", [])]
            stmts.append((bool(st["supported"]) and len(cites) > 0, cites))
        out.append(stmts)
    return out


@register(
    category=_C,
    task="factuality",
    name="Citation recall",
    definition="Share of statements whose cited passages, taken together, support them (statements without "
    "citations count as unsupported); ALCE citation recall.",
    formula="statements fully supported by their citations / statements",
    range="[0, 1]",
    input_requirements=("citations",),
    references=(_REF_ALCE,),
)
@per_item
def citation_recall(citations: Any, *, average: Optional[str] = "macro") -> MetricResult:
    """Citation recall. Each statement is ``{"supported": bool, "citations": [bool, ...]}``: whether all its
    citations together entail it, and whether each cited passage is relevant to it."""
    ans = _citation_inputs(citations)
    return _aggregate(
        "citation_recall",
        "Citation recall",
        [sum(s for s, _ in a) for a in ans],
        [len(a) for a in ans],
        average,
        {},
    )


@register(
    category=_C,
    task="factuality",
    name="Citation precision",
    definition="Share of citations that are relevant to the statement they are attached to (ALCE: a citation "
    "counts if it supports, or is needed to support, the statement).",
    formula="relevant citations / citations",
    range="[0, 1]",
    input_requirements=("citations",),
    references=(_REF_ALCE,),
)
@per_item
def citation_precision(citations: Any, *, average: Optional[str] = "macro") -> MetricResult:
    """Citation precision with the same input as ``citation_recall``."""
    ans = _citation_inputs(citations)
    return _aggregate(
        "citation_precision",
        "Citation precision",
        [sum(sum(c) for _, c in a) for a in ans],
        [sum(len(c) for _, c in a) for a in ans],
        average,
        {},
    )


# ---------------------------------------------------------------- verification and consistency
@register(
    category=_C,
    task="factuality",
    name="Claim verification accuracy",
    definition="Accuracy of supported / contradicted / not-enough-info verdicts against gold labels (FEVER "
    "label accuracy); macro F1 is reported alongside because the classes are often imbalanced.",
    formula="correct verdicts / claims",
    range="[0, 1]",
    input_requirements=("gold_verdicts", "predicted_verdicts"),
    references=(_REF_FEVER,),
)
def claim_verification_accuracy(gold_verdicts: Any, predicted_verdicts: Any) -> MetricResult:
    """Label accuracy for claim verification, with macro F1 over the three verdict classes in ``params``."""
    gold = [_verdict(v, "gold_verdicts") for v in gold_verdicts]
    pred = [_verdict(v, "predicted_verdicts") for v in predicted_verdicts]
    if not gold or len(gold) != len(pred):
        raise InputValidationError("gold_verdicts and predicted_verdicts must be non-empty and the same length.")
    g, p = np.array(gold), np.array(pred)
    f1s = {}
    for c in ("supported", "contradicted", "unsupported"):
        tp = int(np.sum((g == c) & (p == c)))
        fp, fn = int(np.sum((g != c) & (p == c))), int(np.sum((g == c) & (p != c)))
        if tp + fp + fn:
            f1s[c] = 2 * tp / (2 * tp + fp + fn)
    return MetricResult(
        "claim_verification_accuracy",
        "Claim verification accuracy",
        float(np.mean(g == p)),
        {"macro_f1": float(np.mean(list(f1s.values()))), "per_class_f1": f1s, "n_claims": len(gold)},
    )


@register(
    category=_C,
    task="factuality",
    name="Knowledge consistency",
    definition="Share of facts extracted from the outputs (e.g. subject–relation–object triples) that are present "
    "in a specified trusted knowledge source.",
    formula="|extracted facts ∩ knowledge base| / |extracted facts|",
    range="[0, 1]",
    input_requirements=("extracted_facts", "knowledge_base"),
    references=(_REF_FACTSCORE,),
)
@per_item
def knowledge_consistency(
    extracted_facts: Any, knowledge_base: Any, *, average: Optional[str] = "micro"
) -> MetricResult:
    """Knowledge consistency: ``extracted_facts`` is one iterable of hashable facts per output (strings or
    tuples), ``knowledge_base`` a set of trusted facts. Normalise facts the same way on both sides."""
    kb = set(knowledge_base)
    if not kb:
        raise InputValidationError("knowledge_base is empty.")
    outs = list(extracted_facts)
    if not outs:
        raise InputValidationError("extracted_facts is empty.")
    hits, totals = [], []
    for facts in outs:
        fs = list(facts)
        hits.append(sum(f in kb for f in fs))
        totals.append(len(fs))
    return _aggregate(
        "knowledge_consistency", "Knowledge consistency", hits, totals, average, {"kb_size": len(kb)}
    )


# ---------------------------------------------------------------- answers
@register(
    category=_C,
    task="question-answering",
    name="Answer correctness",
    definition="Agreement of an answer with the reference at the level of statements: F1 over true-positive "
    "(in both), false-positive (only in the answer) and false-negative (only in the reference) statements, "
    "optionally blended with a semantic similarity score (RAGAS answer correctness).",
    formula="w_f · TP/(TP + ½(FP + FN)) + w_s · similarity",
    range="[0, 1]",
    input_requirements=("tp", "fp", "fn"),
    references=(_REF_RAGAS,),
)
@per_item
def answer_correctness(
    tp: Any,
    fp: Any,
    fn: Any,
    *,
    similarity: Any = None,
    weights: tuple[float, float] = (0.75, 0.25),
    average: Optional[str] = "mean",
) -> MetricResult:
    """Answer correctness from statement counts per answer (from a judge or annotators). Without
    ``similarity`` it is the statement F1; with it, ``weights`` blend F1 and similarity (RAGAS default
    0.75 / 0.25)."""
    t, f_p, f_n = (np.asarray(v, dtype=float).ravel() for v in (tp, fp, fn))
    if not (t.shape == f_p.shape == f_n.shape) or t.size == 0:
        raise InputValidationError("tp, fp and fn must be non-empty arrays of the same length.")
    if np.any(t < 0) or np.any(f_p < 0) or np.any(f_n < 0):
        raise InputValidationError("Statement counts must be non-negative.")
    denom = t + 0.5 * (f_p + f_n)
    f1 = np.where(denom > 0, t / np.where(denom > 0, denom, 1), 1.0)  # nothing to state and nothing stated -> 1
    if similarity is None:
        scores = f1
    else:
        sim = np.asarray(similarity, dtype=float).ravel()
        if sim.shape != t.shape or np.any(sim < 0) or np.any(sim > 1):
            raise InputValidationError("similarity must hold one value in [0, 1] per answer.")
        wf, ws = weights
        if wf < 0 or ws < 0 or wf + ws == 0:
            raise InputValidationError("weights must be non-negative and not both zero.")
        scores = (wf * f1 + ws * sim) / (wf + ws)
    value: Any = scores if average is None else float(scores.mean())
    return MetricResult(
        "answer_correctness", "Answer correctness", value, {"weights": weights if similarity is not None else None}
    )


@register(
    category=_C,
    task="question-answering",
    name="Answer relevance",
    definition="How directly an answer addresses the question: mean cosine similarity between the question's "
    "embedding and embeddings of questions regenerated from the answer (RAGAS answer relevancy).",
    formula="mean_k cos(e(question), e(regenerated question_k))",
    range="[-1, 1]",
    input_requirements=("question_embeddings", "generated_question_embeddings"),
    references=(_REF_RAGAS,),
)
@per_item
def answer_relevance(
    question_embeddings: Any, generated_question_embeddings: Any, *, average: Optional[str] = "mean"
) -> MetricResult:
    """Answer relevance from embeddings: ``question_embeddings`` is ``(n, dim)``; ``generated_question_embeddings``
    holds, per answer, a ``(k, dim)`` array of embeddings of questions an LLM regenerated from the answer."""
    q = np.asarray(question_embeddings, dtype=float)
    gens = list(generated_question_embeddings)
    if q.ndim != 2 or len(gens) != q.shape[0] or q.shape[0] == 0:
        raise InputValidationError("question_embeddings must be (n, dim) with one generated set per question.")
    scores = np.empty(q.shape[0])
    for i, g in enumerate(gens):
        ga = np.atleast_2d(np.asarray(g, dtype=float))
        if ga.shape[1] != q.shape[1] or ga.shape[0] == 0:
            raise InputValidationError(f"generated_question_embeddings[{i}] must be (k, {q.shape[1]}).")
        qn, gn = np.linalg.norm(q[i]), np.linalg.norm(ga, axis=1)
        if qn == 0 or np.any(gn == 0):
            raise InputValidationError("Embeddings must not be zero vectors.")
        scores[i] = float(np.mean(ga @ q[i] / (gn * qn)))
    value: Any = scores if average is None else float(scores.mean())
    return MetricResult("answer_relevance", "Answer relevance", value, {"n_examples": len(gens)})


@register(
    category=_C,
    task="question-answering",
    name="Abstention accuracy",
    definition="Whether a system abstains exactly when it should (the evidence is insufficient or it would "
    "otherwise be wrong): accuracy of the abstain / answer decision; abstention precision, recall and the "
    "accuracy on answered questions are reported alongside.",
    formula="mean[abstained_i = should_abstain_i]",
    range="[0, 1]",
    input_requirements=("should_abstain", "abstained"),
    references=(_REF_ABSTAIN,),
)
@per_item
def abstention_accuracy(should_abstain: Any, abstained: Any, *, correct: Any = None) -> MetricResult:
    """Abstention accuracy. ``correct`` (optional, one bool per question) gives the selective accuracy on the
    questions the system did answer, and its coverage."""
    s = np.asarray(should_abstain, dtype=bool).ravel()
    a = np.asarray(abstained, dtype=bool).ravel()
    if s.size == 0 or s.shape != a.shape:
        raise InputValidationError("should_abstain and abstained must be non-empty and the same length.")
    tp = int(np.sum(s & a))
    params: dict[str, Any] = {
        "abstention_precision": tp / int(a.sum()) if a.any() else float("nan"),
        "abstention_recall": tp / int(s.sum()) if s.any() else float("nan"),
        "abstention_rate": float(a.mean()),
        "n_questions": int(s.size),
    }
    if correct is not None:
        c = np.asarray(correct, dtype=bool).ravel()
        if c.shape != s.shape:
            raise InputValidationError("correct must have one value per question.")
        answered = ~a
        params["coverage"] = float(answered.mean())
        params["answered_accuracy"] = float(c[answered].mean()) if answered.any() else float("nan")
    return MetricResult("abstention_accuracy", "Abstention accuracy", float(np.mean(s == a)), params)
