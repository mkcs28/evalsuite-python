"""Multilingual and cross-lingual evaluation (v0.5.0).

Translation quality uses the existing ``es.bleu``, ``es.chrf`` and ``es.model_score`` (COMET); cross-lingual
retrieval uses ``es.recall_at_k`` and ``es.mrr``. This module adds language identification, bitext mining over
multilingual embeddings, performance parity across languages, code-switching robustness, z-normalised direct
assessment (adequacy / fluency), cultural appropriateness ratings and language / cross-lingual consistency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Callable, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import bools, by_group, floats, rate, same_length, seq

__all__ = [
    "bitext_mining_accuracy",
    "code_switching_robustness",
    "cross_lingual_consistency",
    "cultural_appropriateness",
    "direct_assessment",
    "language_consistency",
    "language_id_accuracy",
    "language_parity",
]

_C = "multilingual"
_REF_LANGID = "Lui M, Baldwin T. langid.py: an off-the-shelf language identification tool. ACL Demos. 2012:25-30."
_REF_LASER = (
    "Artetxe M, Schwenk H. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and "
    "beyond. TACL. 2019;7:597-610."
)
_REF_XTREME = (
    "Hu J, Ruder S, Siddhant A, Neubig G, Firat O, Johnson M. XTREME: a massively multilingual multi-task "
    "benchmark "
    "for evaluating cross-lingual generalization. ICML. 2020:4411-4421."
)
_REF_LINCE = (
    "Aguilar G, Kar S, Solorio T. LinCE: a centralized benchmark for linguistic code-switching evaluation. LREC. "
    "2020:1803-1813."
)
_REF_DA = (
    "Graham Y, Baldwin T, Moffat A, Zobel J. Continuous measurement scales in human evaluation of machine "
    "translation. LAW. 2013:33-41."
)
_REF_BLEND = (
    "Myung J, Lee N, Zhou Y, et al. BLEnD: a benchmark for LLMs on everyday knowledge in diverse cultures and "
    "languages. NeurIPS Datasets and Benchmarks. 2024."
)
_REF_XLC = (
    "Qi J, Fernández R, Bisazza A. Cross-lingual consistency of factual knowledge in multilingual language "
    "models. "
    "EMNLP. 2023:10650-10666."
)
_REF_MARCHISIO = (
    "Marchisio K, Ko WY, Bérard A, Dehaze T, Ruder S. Understanding and mitigating language confusion in LLMs. "
    "EMNLP. 2024:6653-6677."
)


@register(
    category=_C,
    task="language-identification",
    name="Language identification accuracy",
    definition="Accuracy of predicted language labels, with macro-F1 over languages and per-language recall.",
    formula="mean[pred = true]",
    range="[0, 1]",
    input_requirements=("true_languages", "predicted_languages"),
    references=(_REF_LANGID,),
)
def language_id_accuracy(true_languages: Any, predicted_languages: Any) -> MetricResult:
    t = [str(v) for v in seq(true_languages, "true_languages")]
    p = [str(v) for v in seq(predicted_languages, "predicted_languages")]
    same_length(("true_languages", t), ("predicted_languages", p))
    labels = sorted(set(t) | set(p))
    f1s, recall = [], {}
    for lab in labels:
        tp = sum(a == lab and b == lab for a, b in zip(t, p))
        fp = sum(a != lab and b == lab for a, b in zip(t, p))
        fn = sum(a == lab and b != lab for a, b in zip(t, p))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
        if tp + fn:
            recall[lab] = tp / (tp + fn)
    acc = float(np.mean([a == b for a, b in zip(t, p)]))
    return MetricResult(
        "language_id_accuracy", "Language-ID accuracy", acc, {"macro_f1": float(np.mean(f1s)), "recall": recall}
    )


def _unit(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        raise InputValidationError(f"{name} must be a 2-D array with at least two rows (one per sentence).")
    if not np.all(np.isfinite(a)):
        raise InputValidationError(f"{name} contains NaN or infinite values.")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n == 0):
        raise InputValidationError(f"{name} contains a zero vector.")
    return np.asarray(a / n, dtype=np.float64)


@register(
    category=_C,
    task="semantic-similarity",
    name="Multilingual semantic similarity (bitext mining)",
    definition="For aligned sentence pairs in two languages, the share whose nearest neighbour by cosine (or "
    "margin) similarity is the true translation, averaged over both directions (Tatoeba / BUCC style), with the "
    "mean cosine of the true pairs.",
    formula="acc = ½ (mean_i 1[argmax_j cos(s_i, t_j) = i] + mean_j 1[argmax_i cos(s_i, t_j) = j])",
    range="[0, 1]",
    input_requirements=("source_embeddings", "target_embeddings"),
    references=(_REF_LASER, _REF_XTREME),
)
def bitext_mining_accuracy(
    source_embeddings: Any, target_embeddings: Any, *, scoring: str = "cosine", k: int = 4
) -> MetricResult:
    """``scoring="margin"`` uses the ratio margin of Artetxe & Schwenk with ``k`` neighbours, which removes hub
    effects."""
    s = _unit(source_embeddings, "source_embeddings")
    t = _unit(target_embeddings, "target_embeddings")
    if s.shape != t.shape:
        raise InputValidationError(
            "source_embeddings and target_embeddings must have the same shape (aligned pairs)."
        )
    sim = s @ t.T
    if scoring == "margin":
        kk = min(k, sim.shape[0])
        rs = np.sort(sim, axis=1)[:, -kk:].mean(1)
        rt = np.sort(sim, axis=0)[-kk:, :].mean(0)
        sim = sim / ((rs[:, None] + rt[None, :]) / 2)
    elif scoring != "cosine":
        raise InputValidationError("scoring must be 'cosine' or 'margin'.")
    idx = np.arange(sim.shape[0])
    fwd = float(np.mean(sim.argmax(1) == idx))
    bwd = float(np.mean(sim.argmax(0) == idx))
    return MetricResult(
        "bitext_mining_accuracy",
        "Bitext mining accuracy",
        (fwd + bwd) / 2,
        {
            "source_to_target": fwd,
            "target_to_source": bwd,
            "mean_pair_cosine": float(np.mean(np.sum(s * t, axis=1))),
            "scoring": scoring,
        },
    )


@register(
    category=_C,
    task="fairness",
    name="Language-specific performance parity",
    definition="How evenly a model performs across languages: the worst-to-best ratio (the value; 1 = parity), "
    "the largest gap, the standard deviation, and per-language scores.",
    formula="parity = min_l score_l / max_l score_l",
    range="[0, 1]",
    input_requirements=("scores_by_language",),
    references=(_REF_XTREME,),
)
def language_parity(scores_by_language: Mapping[str, Any]) -> MetricResult:
    """``scores_by_language``: language -> per-example scores (or one aggregate score)."""
    if not isinstance(scores_by_language, Mapping) or len(scores_by_language) < 2:
        raise InputValidationError("scores_by_language must map at least two languages to scores.")
    means = {}
    for lang, v in scores_by_language.items():
        arr = floats(v if isinstance(v, (list, tuple, np.ndarray)) else [v], f"scores_by_language[{lang!r}]")
        means[str(lang)] = float(arr.mean())
    vals = np.array(list(means.values()))
    if vals.max() <= 0:
        raise InputValidationError("Scores must be positive for some language to compute parity.")
    worst = min(means, key=lambda k: means[k])
    return MetricResult(
        "language_parity",
        "Language parity (worst / best)",
        float(max(0.0, vals.min()) / vals.max()),
        {
            "max_gap": float(vals.max() - vals.min()),
            "std": float(vals.std(ddof=1)),
            "worst_language": worst,
            "by_language": means,
        },
    )


@register(
    category=_C,
    task="robustness",
    name="Code-switching robustness",
    definition="Accuracy on code-switched inputs (mixing languages within an utterance) compared with the same "
    "content in one language: code-switched accuracy, drop and flip rate.",
    formula="acc_cs; drop = acc_mono − acc_cs",
    range="[0, 1]",
    input_requirements=("monolingual_correct", "code_switched_correct"),
    references=(_REF_LINCE,),
)
def code_switching_robustness(monolingual_correct: Any, code_switched_correct: Any) -> MetricResult:
    m, c = bools(monolingual_correct, "monolingual_correct"), bools(code_switched_correct, "code_switched_correct")
    same_length(("monolingual_correct", m), ("code_switched_correct", c))
    am, ac = float(m.mean()), float(c.mean())
    return MetricResult(
        "code_switching_robustness",
        "Code-switched accuracy",
        ac,
        {
            "monolingual_accuracy": am,
            "absolute_drop": am - ac,
            "flip_rate": float((m & ~c).sum() / m.sum()) if m.any() else float("nan"),
        },
    )


@register(
    category=_C,
    task="translation",
    name="Translation adequacy / fluency (direct assessment)",
    definition="Human direct-assessment scores (0–100 adequacy or fluency) standardised per rater (z-scores) to "
    "remove rater bias, then averaged per system, as in WMT DA.",
    formula="z = (score − mean_rater) / sd_rater; system score = mean z",
    range="(−∞, ∞) for z; [0, 100] raw",
    input_requirements=("scores", "raters"),
    references=(_REF_DA,),
)
def direct_assessment(
    scores: Any, raters: Sequence[Any], *, systems: Optional[Sequence[Any]] = None
) -> MetricResult:
    s = floats(scores, "scores")
    r = [str(v) for v in seq(raters, "raters")]
    same_length(("scores", s), ("raters", r))
    z = np.empty_like(s)
    for rater in set(r):
        m = np.array([x == rater for x in r])
        sd = s[m].std(ddof=1) if m.sum() > 1 else 0.0
        z[m] = (s[m] - s[m].mean()) / sd if sd > 0 else 0.0
    extra: dict[str, Any] = {"raw_mean": float(s.mean()), "n_raters": len(set(r))}
    if systems is not None:
        extra["by_system_z"] = by_group(z, systems, "systems")
        extra["by_system_raw"] = by_group(s, systems, "systems")
    return MetricResult("direct_assessment", "Direct assessment (mean z)", float(z.mean()), extra)


@register(
    category=_C,
    task="culture",
    name="Cultural appropriateness",
    definition="Mean rating of how appropriate responses are for the target culture, rescaled to [0, 1] from the "
    "rating scale, overall and per culture / region (the gap between the best and worst region is reported).",
    formula="mean (rating − lo) / (hi − lo)",
    range="[0, 1]",
    input_requirements=("ratings",),
    references=(_REF_BLEND,),
)
def cultural_appropriateness(
    ratings: Any, *, scale: tuple[float, float] = (1, 5), regions: Optional[Sequence[Any]] = None
) -> MetricResult:
    lo, hi = scale
    if hi <= lo:
        raise InputValidationError("scale must be (low, high) with high > low.")
    r = floats(ratings, "ratings", lo=lo, hi=hi)
    v = (r - lo) / (hi - lo)
    extra: dict[str, Any] = {"scale": scale}
    if regions is not None:
        by = by_group(v, regions, "regions")
        extra["by_region"] = by
        extra["region_gap"] = max(by.values()) - min(by.values())
    return MetricResult("cultural_appropriateness", "Cultural appropriateness", float(v.mean()), extra)


@register(
    category=_C,
    task="consistency",
    name="Language consistency",
    definition="Share of responses written in the language the user asked or wrote in (language confusion), "
    "given the detected language of each response.",
    formula="mean[detected = expected]",
    range="[0, 1]",
    input_requirements=("expected_languages", "detected_languages"),
    references=(_REF_MARCHISIO,),
)
def language_consistency(expected_languages: Any, detected_languages: Any) -> MetricResult:
    e = [str(v) for v in seq(expected_languages, "expected_languages")]
    d = [str(v) for v in seq(detected_languages, "detected_languages")]
    same_length(("expected_languages", e), ("detected_languages", d))
    flags = np.array([a == b for a, b in zip(e, d)])
    return rate(
        "language_consistency", "Language consistency", flags, {"by_language": by_group(flags.astype(float), e)}
    )


@register(
    category=_C,
    task="consistency",
    name="Cross-lingual factual consistency",
    definition="Whether the model gives the same answer to the same question asked in different languages: share "
    "of questions answered identically in every language and mean pairwise agreement (after mapping answers to a "
    "common form with ``normalize``).",
    formula="mean_q 1[all languages agree]",
    range="[0, 1]",
    input_requirements=("answers_by_language",),
    references=(_REF_XLC,),
)
def cross_lingual_consistency(
    answers_by_language: Any, *, normalize: Optional[Callable[[Any], Any]] = None
) -> MetricResult:
    """``answers_by_language``: per question, a mapping language -> answer (at least two languages)."""
    rows = seq(answers_by_language, "answers_by_language")
    full, pair = [], []
    for i, r in enumerate(rows):
        if not isinstance(r, Mapping) or len(r) < 2:
            raise InputValidationError(f"answers_by_language[{i}] must map at least two languages to answers.")
        v = [
            normalize(a) if normalize else (" ".join(a.lower().split()) if isinstance(a, str) else a)
            for a in r.values()
        ]
        full.append(all(x == v[0] for x in v))
        pair.append(np.mean([x == y for x, y in combinations(v, 2)]))
    return MetricResult(
        "cross_lingual_consistency",
        "Cross-lingual consistency",
        float(np.mean(full)),
        {"pairwise_agreement": float(np.mean(pair)), "n_questions": len(rows)},
    )
