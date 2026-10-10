"""Calibration and uncertainty for LLM answers (v0.5.0).

Inputs are, per question, whether the answer was correct and the model's confidence in it (a verbalised
probability, a token probability or a self-consistency share). ECE, MCE, the Brier score and log loss are the
existing ``es.expected_calibration_error``, ``es.maximum_calibration_error``, ``es.brier_score`` and
``es.log_loss`` applied to (correct, confidence); abstention quality is ``es.abstention_accuracy``. This module
adds adaptive calibration error and selective prediction: risk-coverage curves, AURC, risk at a coverage and
coverage at a risk, and confidence-accuracy correlation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import bools, floats, same_length

__all__ = [
    "adaptive_calibration_error",
    "aurc",
    "confidence_accuracy_correlation",
    "coverage_at_risk",
    "risk_at_coverage",
    "risk_coverage_curve",
    "selective_risk",
]

_C = "uncertainty"
_REF_NIXON = (
    "Nixon J, Dusenberry M, Jerfel G, Nguyen T, Liu J, Zhang L, Tran D. Measuring calibration in deep learning. "
    "CVPR Workshops. 2019."
)
_REF_GEIFMAN = (
    "Geifman Y, Uziel G, El-Yaniv R. Bias-reduced uncertainty estimation for deep neural classifiers. ICLR. 2019."
)
_REF_ELYANIV = (
    "El-Yaniv R, Wiener Y. On the foundations of noise-free selective classification. JMLR. 2010;11:1605-1641."
)
_REF_KADAVATH = (
    "Kadavath S, Conerly T, Askell A, et al. Language models (mostly) know what they know. arXiv:2207.05221. 2022."
)


def _inputs(correct: Any, confidence: Any) -> tuple[np.ndarray, np.ndarray]:
    c = bools(correct, "correct")
    p = floats(confidence, "confidence", lo=0, hi=1)
    same_length(("correct", c), ("confidence", p))
    return c, p


def risk_coverage_curve(correct: Any, confidence: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coverage, selective risk, confidence threshold) when answering the k most confident questions, for
    k = 1..n. Tied confidences are kept together (the curve only has points between distinct values)."""
    c, p = _inputs(correct, confidence)
    order = np.argsort(-p, kind="mergesort")
    err = (~c[order]).astype(float)
    ps = p[order]
    k = np.arange(1, c.size + 1)
    risk = np.cumsum(err) / k
    last = np.r_[ps[1:] != ps[:-1], True]  # end of each block of tied confidences
    return k[last] / c.size, risk[last], ps[last]


@register(
    category=_C,
    task="calibration",
    name="Adaptive calibration error (ACE)",
    definition="Calibration error with equal-mass bins: the unweighted mean |accuracy − confidence| over R bins "
    "that each hold the same number of predictions, so sparse high-confidence regions do not hide errors.",
    formula="ACE = (1/R) Σ_r |acc(r) − conf(r)|, bins of equal count",
    range="[0, 1]",
    input_requirements=("correct", "confidence"),
    references=(_REF_NIXON,),
    higher_is_better=False,
)
def adaptive_calibration_error(correct: Any, confidence: Any, *, n_bins: int = 15) -> MetricResult:
    c, p = _inputs(correct, confidence)
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise InputValidationError("n_bins must be a positive integer.")
    r = min(n_bins, c.size)
    order = np.argsort(p, kind="mergesort")
    gaps = [abs(c[idx].mean() - p[idx].mean()) for idx in np.array_split(order, r)]
    return MetricResult(
        "adaptive_calibration_error", "Adaptive calibration error", float(np.mean(gaps)), {"n_bins": r}
    )


@register(
    category=_C,
    task="selective-prediction",
    name="Selective risk",
    definition="Error rate on the questions the model answers when it abstains on the least confident ones so "
    "that a given share (coverage) is answered.",
    formula="risk(c) = errors among the ⌈c·n⌉ most confident / ⌈c·n⌉",
    range="[0, 1]",
    input_requirements=("correct", "confidence"),
    references=(_REF_ELYANIV, _REF_GEIFMAN),
    higher_is_better=False,
)
def selective_risk(correct: Any, confidence: Any, *, coverage: float = 0.8) -> MetricResult:
    if not 0 < coverage <= 1:
        raise InputValidationError("coverage must be in (0, 1].")
    c, p = _inputs(correct, confidence)
    k = int(np.ceil(coverage * c.size))
    order = np.argsort(-p, kind="mergesort")
    risk = float((~c[order[:k]]).mean())
    return MetricResult(
        "selective_risk",
        "Selective risk",
        risk,
        {"coverage": k / c.size, "threshold": float(p[order[k - 1]]), "full_coverage_risk": float((~c).mean())},
    )


@register(
    category=_C,
    task="selective-prediction",
    name="Area under the risk-coverage curve (AURC)",
    definition="Mean selective risk over all coverages when questions are answered in order of decreasing "
    "confidence; lower means confidence ranks errors last. E-AURC subtracts the AURC of a perfect ranking.",
    formula="AURC = (1/n) Σ_k risk(k/n); E-AURC = AURC − AURC*",
    range="[0, 1]",
    input_requirements=("correct", "confidence"),
    references=(_REF_GEIFMAN,),
    higher_is_better=False,
)
def aurc(correct: Any, confidence: Any) -> MetricResult:
    c, p = _inputs(correct, confidence)
    order = np.argsort(-p, kind="mergesort")
    n = c.size
    k = np.arange(1, n + 1)
    value = float(np.mean(np.cumsum(~c[order]) / k))
    errors = int((~c).sum())
    optimal_curve = np.maximum(0, k - (n - errors)) / k
    opt = float(np.mean(optimal_curve))
    return MetricResult("aurc", "AURC", value, {"e_aurc": value - opt, "optimal_aurc": opt, "n": n})


@register(
    category=_C,
    task="selective-prediction",
    name="Risk at fixed coverage",
    definition="Selective risk at the target coverage, read from the risk-coverage curve at the smallest "
    "confidence threshold whose coverage reaches the target (ties are answered together).",
    formula="risk at the first curve point with coverage ≥ target",
    range="[0, 1]",
    input_requirements=("correct", "confidence"),
    references=(_REF_ELYANIV, _REF_GEIFMAN),
    higher_is_better=False,
)
def risk_at_coverage(correct: Any, confidence: Any, *, coverage: float = 0.8) -> MetricResult:
    if not 0 < coverage <= 1:
        raise InputValidationError("coverage must be in (0, 1].")
    cov, risk, thr = risk_coverage_curve(correct, confidence)
    i = int(np.searchsorted(cov, coverage - 1e-12))
    return MetricResult(
        "risk_at_coverage",
        "Risk at coverage",
        float(risk[i]),
        {"target_coverage": coverage, "achieved_coverage": float(cov[i]), "threshold": float(thr[i])},
    )


@register(
    category=_C,
    task="selective-prediction",
    name="Coverage at fixed risk",
    definition="The largest share of questions the model can answer (abstaining on the rest by confidence) while "
    "keeping the selective error rate at or below the target risk.",
    formula="max coverage over curve points with risk ≤ target",
    range="[0, 1]",
    input_requirements=("correct", "confidence"),
    references=(_REF_ELYANIV, _REF_GEIFMAN),
)
def coverage_at_risk(correct: Any, confidence: Any, *, risk: float = 0.05) -> MetricResult:
    if not 0 <= risk < 1:
        raise InputValidationError("risk must be in [0, 1).")
    cov, r, thr = risk_coverage_curve(correct, confidence)
    ok = np.flatnonzero(r <= risk + 1e-12)
    if ok.size == 0:
        return MetricResult("coverage_at_risk", "Coverage at risk", 0.0, {"target_risk": risk, "threshold": None})
    i = int(ok[-1])
    return MetricResult(
        "coverage_at_risk",
        "Coverage at risk",
        float(cov[i]),
        {"target_risk": risk, "achieved_risk": float(r[i]), "threshold": float(thr[i])},
    )


@register(
    category=_C,
    task="calibration",
    name="Confidence–accuracy correlation",
    definition="How well confidence discriminates correct from incorrect answers: the AUROC of confidence for "
    "correctness (P(true) discrimination, Kadavath et al.), with the point-biserial and Spearman correlations.",
    formula="AUROC = P(conf_correct > conf_wrong) + ½ P(tie)",
    range="[0, 1] (0.5 = no signal)",
    input_requirements=("correct", "confidence"),
    references=(_REF_KADAVATH,),
)
def confidence_accuracy_correlation(correct: Any, confidence: Any) -> MetricResult:
    from scipy import stats

    c, p = _inputs(correct, confidence)
    if c.all() or not c.any():
        raise InputValidationError("Need both correct and incorrect answers.")
    ranks = stats.rankdata(p)
    n1 = int(c.sum())
    auroc = float((ranks[c].sum() - n1 * (n1 + 1) / 2) / (n1 * (c.size - n1)))
    if np.ptp(p) == 0:
        pb = sp = float("nan")
    else:
        pb = float(stats.pearsonr(c.astype(float), p)[0])
        sp = float(stats.spearmanr(c.astype(float), p)[0])
    return MetricResult(
        "confidence_accuracy_correlation",
        "Confidence–accuracy AUROC",
        auroc,
        {"point_biserial": pb, "spearman": sp, "accuracy": float(c.mean()), "mean_confidence": float(p.mean())},
    )
