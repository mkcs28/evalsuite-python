"""Sampling-based correctness: pass@k (unbiased estimator) and majority-vote accuracy."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import _seq, as_references, check_positive_int, per_item
from .qa import _norm

__all__ = ["benchmark_accuracy", "extract_answer", "majority_vote_accuracy", "pass_at_k"]

_C = "reasoning"
_REF_CHEN = "Chen M, et al. Evaluating large language models trained on code. arXiv:2107.03374. 2021."
_REF_WANG = (
    "Wang X, Wei J, Schuurmans D, et al. Self-consistency improves chain of thought reasoning in language "
    "models. ICLR. 2023."
)


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@register(
    category=_C,
    task="sampling",
    name="pass@k",
    definition="Probability that at least one of k samples drawn without replacement from the n generated for "
    "a problem passes its tests, estimated without bias from the c passing samples and averaged over problems.",
    formula="pass@k = mean_problems [1 − C(n − c, k) / C(n, k)]",
    range="[0, 1]",
    input_requirements=("n_samples", "n_correct"),
    references=(_REF_CHEN,),
)
@per_item
def pass_at_k(n_samples: Any, n_correct: Any, *, k: int = 1, average: Optional[str] = "mean") -> MetricResult:
    """Unbiased pass@k (Chen et al. 2021) from per-problem sample counts ``n`` and passing counts ``c``.

    ``n_samples`` may be one integer for all problems. Every problem needs ``n >= k``."""
    k = check_positive_int(k, "k")
    c = np.asarray(_seq(n_correct, "n_correct"))
    n = np.full(c.shape, n_samples) if np.ndim(n_samples) == 0 else np.asarray(_seq(n_samples, "n_samples"))
    if n.shape != c.shape:
        raise InputValidationError(f"n_samples and n_correct differ in length: {n.size} and {c.size}.")
    if not (np.issubdtype(n.dtype, np.integer) and np.issubdtype(c.dtype, np.integer)):
        raise InputValidationError("n_samples and n_correct must be integers (counts of samples).")
    if np.any(c < 0) or np.any(c > n):
        raise InputValidationError("Each n_correct must be between 0 and its n_samples.")
    if np.any(n < k):
        raise InputValidationError(
            f"pass@{k} needs at least k = {k} samples per problem; the smallest n is {n.min()}."
        )
    scores = np.array([_pass_at_k(int(a), int(b), k) for a, b in zip(n, c)])
    value: Any = scores if average is None else float(scores.mean())
    return MetricResult("pass_at_k", f"pass@{k}", value, {"k": k, "n_problems": int(c.size)})


@register(
    category=_C,
    task="sampling",
    name="Majority-vote accuracy",
    definition="Accuracy of the most frequent answer among several samples per question (self-consistency); "
    "ties go to the answer seen first.",
    formula="mean_i [mode(samples_i) matches a reference]",
    range="[0, 1]",
    input_requirements=("references", "samples"),
    references=(_REF_WANG,),
)
@per_item
def majority_vote_accuracy(
    references: Any, samples: Any, *, normalize: Any = True, average: Optional[str] = "mean"
) -> MetricResult:
    """Majority-vote (self-consistency) accuracy. ``samples[i]`` is the list of answers sampled for question i;
    answers are compared after SQuAD normalization unless ``normalize=False``."""
    groups = _seq(samples, "samples")
    refs = as_references(references, len(groups))
    f = _norm(normalize)
    scores = []
    for i, (answers, rs) in enumerate(zip(groups, refs)):
        if isinstance(answers, str) or not answers:
            raise InputValidationError(f"samples[{i}] must be a non-empty list of answer strings.")
        votes = Counter(f(str(a)) for a in answers)
        top = votes.most_common(1)[0][0]  # Counter keeps first-seen order among ties
        scores.append(float(any(top == f(r) for r in rs)))
    arr = np.array(scores)
    value: Any = arr if average is None else float(arr.mean())
    return MetricResult("majority_vote_accuracy", "Majority-vote accuracy", value, {"n_questions": len(scores)})


# ---------------------------------------------------------------- answer extraction and benchmark accuracy
_REF_GSM8K = (
    "Cobbe K, Kosaraju V, Bavarian M, et al. Training verifiers to solve math word problems. "
    "arXiv:2110.14168. 2021."
)
_REF_MATH = (
    "Hendrycks D, Burns C, Kadavath S, et al. Measuring mathematical problem solving with the MATH dataset. "
    "NeurIPS Datasets and Benchmarks. 2021."
)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def _boxed(text: str) -> Optional[str]:
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _clean_number(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def extract_answer(text: str, *, style: str = "gsm8k") -> Optional[str]:
    """Extract a final answer from a model output.

    - ``"gsm8k"``: the number after ``####`` if present, else the last number in the text (commas, ``$`` and
      trailing ``.0`` removed);
    - ``"boxed"``: the content of the last ``\\boxed{...}`` (MATH convention);
    - ``"choice"``: the last standalone option letter A–J (multiple choice, e.g. MMLU, GPQA, ARC);
    - ``"last_line"``: the last non-empty line.

    Returns ``None`` when nothing is found."""
    if not isinstance(text, str):
        raise InputValidationError("text must be a string.")
    if style == "gsm8k":
        if "####" in text:
            tail = text.split("####")[-1]
            m = _NUMBER.search(tail)
            return _clean_number(m.group()) if m else tail.strip() or None
        nums = _NUMBER.findall(text)
        return _clean_number(nums[-1]) if nums else None
    if style == "boxed":
        b = _boxed(text)
        return b.strip() if b is not None else None
    if style == "choice":
        found = re.findall(r"(?:^|[^A-Za-z])\(?([A-J])\)?(?=[^A-Za-z]|$)", text)
        return found[-1] if found else None
    if style == "last_line":
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        return lines[-1] if lines else None
    raise InputValidationError("style must be 'gsm8k', 'boxed', 'choice' or 'last_line'.")


@register(
    category=_C,
    task="reasoning",
    name="Benchmark accuracy",
    definition="Share of benchmark questions answered correctly after extracting the final answer with the "
    "benchmark's convention (GSM8K '####' or last number, MATH \\boxed{}, multiple-choice letter) and "
    "comparing with the gold answer.",
    formula="mean_i [extract(output_i) = gold_i]",
    range="[0, 1]",
    input_requirements=("references", "outputs"),
    references=(_REF_GSM8K, _REF_MATH),
)
@per_item
def benchmark_accuracy(
    references: Any, outputs: Any, *, style: str = "gsm8k", normalize: Any = None, average: Optional[str] = "mean"
) -> MetricResult:
    """Accuracy with answer extraction. Gold answers are cleaned the same way (for ``"gsm8k"``, the gold
    solution's ``####`` line is used when present). ``normalize`` optionally maps both sides (for MATH, pass
    an equivalence-normalising function such as the one from the MATH repository)."""
    outs = [str(o) for o in _seq(outputs, "outputs")]
    golds = [str(g) for g in _seq(references, "references")]
    if len(outs) != len(golds):
        raise InputValidationError("references and outputs must have the same length.")
    fix = normalize if callable(normalize) else (lambda s: s)
    hits, missing = [], 0
    for g, o in zip(golds, outs):
        gold = (
            extract_answer(g, style=style)
            if style in ("gsm8k", "boxed") and (("####" in g) or ("\\boxed" in g))
            else g.strip()
        )
        if style == "gsm8k" and gold is not None:
            gold = _clean_number(gold)
        pred = extract_answer(o, style=style)
        if pred is None:
            missing += 1
            hits.append(False)
            continue
        hits.append(fix(pred) == fix(gold))
    arr = np.asarray(hits, dtype=float)
    value: Any = arr if average is None else float(arr.mean())
    return MetricResult(
        "benchmark_accuracy",
        "Benchmark accuracy",
        value,
        {"style": style, "n": len(outs), "no_answer_found": missing},
    )
