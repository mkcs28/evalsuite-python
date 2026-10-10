"""Robustness and reliability (v0.5.0): adversarial, paraphrase, noise, out-of-distribution and distribution
shift, counterfactual invariance, response stability, failures and recovery, prompt sensitivity and
long-context truncation.

Correctness verdicts come from your own scorer (exact match, a judge, unit tests); these functions compare
them across conditions. ``add_typos`` generates reproducible typographical noise for robustness tests.
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Callable, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import bools, floats, lists_of, rate, same_length, seq

__all__ = [
    "add_typos",
    "adversarial_robustness",
    "contradiction_rate",
    "distribution_shift_drop",
    "error_rate",
    "invariance_violation_rate",
    "noise_robustness",
    "ood_accuracy",
    "paraphrase_consistency",
    "prompt_sensitivity",
    "recovery_success_rate",
    "response_stability",
    "truncation_sensitivity",
]

_C = "robustness"
_REF_ADVGLUE = (
    "Wang B, Xu C, Wang S, et al. Adversarial GLUE: a multi-task benchmark for robustness evaluation of language "
    "models. NeurIPS Datasets and Benchmarks. 2021."
)
_REF_CHECKLIST = (
    "Ribeiro MT, Wu T, Guestrin C, Singh S. Beyond accuracy: behavioral testing of NLP models with CheckList. "
    "ACL. 2020:4902-4912."
)
_REF_PROMPTBENCH = (
    "Zhu K, Wang J, Zhou J, et al. PromptRobust: towards evaluating the robustness of large language models on "
    "adversarial prompts. arXiv:2306.04528. 2023."
)
_REF_WILDS = (
    "Koh PW, Sagawa S, Marklund H, et al. WILDS: a benchmark of in-the-wild distribution shifts. ICML. "
    "2021:5637-5664."
)
_REF_HELM = "Liang P, Bommasani R, Lee T, et al. Holistic evaluation of language models. TMLR. 2023."
_REF_SCLAR = (
    "Sclar M, Choi Y, Tsvetkov Y, Suhr A. Quantifying language models' sensitivity to spurious features in prompt "
    "design. ICLR. 2024."
)
_REF_ELAZAR = (
    "Elazar Y, Kassner N, Ravfogel S, et al. Measuring and improving consistency in pretrained language models. "
    "TACL. 2021;9:1012-1031."
)
_REF_MANAKUL = (
    "Manakul P, Liusie A, Gales MJF. SelfCheckGPT: zero-resource black-box hallucination detection for generative "
    "large language models. EMNLP. 2023:9004-9017."
)
_REF_KAUSHIK = (
    "Kaushik D, Hovy E, Lipton ZC. Learning the difference that makes a difference with counterfactually-"
    "augmented data. ICLR. 2020."
)
_REF_BELINKOV = (
    "Belinkov Y, Bisk Y. Synthetic and natural noise both break neural machine translation. ICLR. 2018."
)
_REF_HSIEH = (
    "Hsieh CP, Sun S, Kriman S, et al. RULER: what's the real context size of your long-context language models? "
    "COLM. 2024."
)


def _pair(clean: Any, perturbed: Any, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    c, p = bools(clean, a), bools(perturbed, b)
    same_length((a, c), (b, p))
    return c, p


def _drop_result(
    metric: str, name: str, c: np.ndarray, p: np.ndarray, extra: Optional[dict[str, Any]] = None
) -> MetricResult:
    ca, pa = float(c.mean()), float(p.mean())
    flipped = int((c & ~p).sum())
    params = {
        "clean_accuracy": ca,
        "perturbed_accuracy": pa,
        "absolute_drop": ca - pa,
        "relative_drop": (ca - pa) / ca if ca > 0 else float("nan"),
        "flip_rate": flipped / int(c.sum()) if c.any() else float("nan"),
        "n": int(c.size),
        **(extra or {}),
    }
    return MetricResult(metric, name, pa, params)


@register(
    category=_C,
    task="robustness",
    name="Adversarial robustness",
    definition="Accuracy under adversarial perturbation (robust accuracy), with the clean accuracy, the drop and "
    "the attack success rate: the share of originally correct examples the attack flips.",
    formula="robust acc = mean[correct_adv]; ASR = #(correct_clean ∧ ¬correct_adv) / #correct_clean",
    range="[0, 1]",
    input_requirements=("clean_correct", "adversarial_correct"),
    references=(_REF_ADVGLUE, _REF_PROMPTBENCH),
)
def adversarial_robustness(clean_correct: Any, adversarial_correct: Any) -> MetricResult:
    c, p = _pair(clean_correct, adversarial_correct, "clean_correct", "adversarial_correct")
    r = _drop_result("adversarial_robustness", "Robust accuracy", c, p)
    return MetricResult(r.metric, r.name, r.value, {**r.params, "attack_success_rate": r.params["flip_rate"]})


@register(
    category=_C,
    task="robustness",
    name="Typographical-noise robustness",
    definition="Accuracy on inputs with typos (character swaps, deletions, insertions, substitutions) relative to "
    "clean inputs: perturbed accuracy, absolute and relative drop and flip rate. ``add_typos`` makes the noisy "
    "inputs reproducibly.",
    formula="acc_noisy; drop = acc_clean − acc_noisy",
    range="[0, 1]",
    input_requirements=("clean_correct", "noisy_correct"),
    references=(_REF_BELINKOV, _REF_CHECKLIST),
)
def noise_robustness(clean_correct: Any, noisy_correct: Any) -> MetricResult:
    c, p = _pair(clean_correct, noisy_correct, "clean_correct", "noisy_correct")
    return _drop_result("noise_robustness", "Noisy-input accuracy", c, p)


def add_typos(texts: Any, *, rate: float = 0.05, random_state: Optional[int] = None) -> list[str]:
    """Introduce character-level typos into each text: each letter is perturbed with probability ``rate`` by a
    swap with its neighbour, deletion, insertion or substitution (equally likely). Reproducible with
    ``random_state``."""
    items = seq(texts, "texts")
    if not 0 <= rate <= 1:
        raise InputValidationError("rate must be in [0, 1].")
    rng = np.random.default_rng(random_state)
    letters = string.ascii_lowercase
    out = []
    for i, t in enumerate(items):
        if not isinstance(t, str):
            raise InputValidationError(f"texts[{i}] must be a string.")
        chars = list(t)
        j = 0
        res: list[str] = []
        while j < len(chars):
            ch = chars[j]
            if ch.isalpha() and rng.random() < rate:
                op = int(rng.integers(4))
                if op == 0 and j + 1 < len(chars):
                    res += [chars[j + 1], ch]
                    j += 2
                    continue
                if op == 1:
                    j += 1
                    continue
                if op == 2:
                    res += [ch, letters[int(rng.integers(26))]]
                else:
                    res.append(letters[int(rng.integers(26))])
            else:
                res.append(ch)
            j += 1
        out.append("".join(res))
    return out


def _norm(v: Any, normalize: Optional[Callable[[Any], Any]]) -> Any:
    if normalize is not None:
        return normalize(v)
    return " ".join(v.lower().split()) if isinstance(v, str) else v


@register(
    category=_C,
    task="consistency",
    name="Paraphrase consistency",
    definition="Whether the model gives the same answer to paraphrases of the same question: the share of items "
    "where all paraphrases agree, and the mean pairwise agreement (ParaRel consistency).",
    formula="mean_items 1[all answers equal]; pairwise = mean over pairs 1[a_i = a_j]",
    range="[0, 1]",
    input_requirements=("answers",),
    references=(_REF_ELAZAR, _REF_CHECKLIST),
)
def paraphrase_consistency(answers: Any, *, normalize: Optional[Callable[[Any], Any]] = None) -> MetricResult:
    """``answers``: per item, the answers to each paraphrase (at least two). Strings are compared after
    lower-casing and whitespace normalisation unless ``normalize`` is given."""
    rows = lists_of(answers, "answers")
    if any(len(r) < 2 for r in rows):
        raise InputValidationError("Every item needs answers to at least two paraphrases.")
    full, pair = [], []
    for r in rows:
        v = [_norm(a, normalize) for a in r]
        full.append(all(x == v[0] for x in v))
        pair.append(np.mean([x == y for x, y in combinations(v, 2)]))
    return MetricResult(
        "paraphrase_consistency",
        "Paraphrase consistency",
        float(np.mean(full)),
        {"pairwise_agreement": float(np.mean(pair)), "n_items": len(rows)},
    )


@register(
    category=_C,
    task="generalization",
    name="Out-of-distribution accuracy",
    definition="Accuracy on out-of-distribution examples, with in-distribution accuracy and the gap (WILDS "
    "reports both; the gap is the generalisation cost).",
    formula="acc_OOD = mean[correct | OOD]; gap = acc_ID − acc_OOD",
    range="[0, 1]",
    input_requirements=("correct", "is_ood"),
    references=(_REF_WILDS,),
)
def ood_accuracy(correct: Any, is_ood: Any) -> MetricResult:
    c, o = _pair(correct, is_ood, "correct", "is_ood")
    if not o.any() or o.all():
        raise InputValidationError("Need both in-distribution and out-of-distribution examples.")
    ood, ind = float(c[o].mean()), float(c[~o].mean())
    return MetricResult(
        "ood_accuracy",
        "Out-of-distribution accuracy",
        ood,
        {"id_accuracy": ind, "gap": ind - ood, "n_ood": int(o.sum()), "n_id": int((~o).sum())},
    )


@register(
    category=_C,
    task="generalization",
    name="Distribution-shift performance drop",
    definition="Change in a per-example score from the source to the shifted (target) distribution: absolute and "
    "relative drop with a 95% Welch interval for the absolute drop.",
    formula="drop = mean(source) − mean(target); relative = drop / mean(source)",
    range="(−∞, ∞) (0 = no drop)",
    input_requirements=("source_scores", "target_scores"),
    references=(_REF_WILDS, _REF_HELM),
    higher_is_better=False,
)
def distribution_shift_drop(source_scores: Any, target_scores: Any) -> MetricResult:
    s = floats(source_scores, "source_scores")
    t = floats(target_scores, "target_scores")
    from scipy import stats

    drop = float(s.mean() - t.mean())
    if s.size > 1 and t.size > 1:
        vs, vt = s.var(ddof=1) / s.size, t.var(ddof=1) / t.size
        se = float(np.sqrt(vs + vt))
        dof = (vs + vt) ** 2 / (vs**2 / (s.size - 1) + vt**2 / (t.size - 1)) if vs + vt > 0 else np.inf
        q = float(stats.t.ppf(0.975, dof)) if np.isfinite(dof) else 1.959963984540054
        ci = (drop - q * se, drop + q * se)
    else:
        ci = (float("nan"), float("nan"))
    return MetricResult(
        "distribution_shift_drop",
        "Distribution-shift performance drop",
        drop,
        {
            "source_mean": float(s.mean()),
            "target_mean": float(t.mean()),
            "relative_drop": drop / float(s.mean()) if s.mean() != 0 else float("nan"),
            "ci_low": ci[0],
            "ci_high": ci[1],
        },
    )


@register(
    category=_C,
    task="consistency",
    name="Counterfactual invariance violation rate",
    definition="Share of examples whose prediction changes when only a protected or irrelevant attribute is "
    "changed (names, gender terms, dialect): violations of an invariance test.",
    formula="mean[pred(x) ≠ pred(x')]",
    range="[0, 1]",
    input_requirements=("original_predictions", "counterfactual_predictions"),
    references=(_REF_CHECKLIST, _REF_KAUSHIK),
    higher_is_better=False,
)
def invariance_violation_rate(
    original_predictions: Any,
    counterfactual_predictions: Any,
    *,
    normalize: Optional[Callable[[Any], Any]] = None,
    groups: Optional[Sequence[Any]] = None,
) -> MetricResult:
    o = seq(original_predictions, "original_predictions")
    c = seq(counterfactual_predictions, "counterfactual_predictions")
    same_length(("original_predictions", o), ("counterfactual_predictions", c))
    flags = np.array([_norm(a, normalize) != _norm(b, normalize) for a, b in zip(o, c)])
    extra: dict[str, Any] = {"counterfactual_consistency": 1 - float(flags.mean())}
    if groups is not None:
        from ._common import by_group

        extra["by_group"] = by_group(flags.astype(float), groups)
    return rate("invariance_violation_rate", "Invariance violation rate", flags, extra)


@register(
    category=_C,
    task="consistency",
    name="Response stability",
    definition="Agreement among repeated samples for the same prompt: the mean pairwise agreement and the share "
    "of prompts where every sample agrees, plus the mean share held by the most common answer.",
    formula="mean_prompts mean_{i<j} 1[a_i = a_j]",
    range="[0, 1]",
    input_requirements=("samples",),
    references=(_REF_MANAKUL, _REF_ELAZAR),
)
def response_stability(samples: Any, *, normalize: Optional[Callable[[Any], Any]] = None) -> MetricResult:
    rows = lists_of(samples, "samples")
    if any(len(r) < 2 for r in rows):
        raise InputValidationError("Every prompt needs at least two samples.")
    pair, unanimous, modal = [], [], []
    for r in rows:
        v = [_norm(a, normalize) for a in r]
        pair.append(np.mean([x == y for x, y in combinations(v, 2)]))
        unanimous.append(all(x == v[0] for x in v))
        modal.append(Counter(map(repr, v)).most_common(1)[0][1] / len(v))
    return MetricResult(
        "response_stability",
        "Response stability",
        float(np.mean(pair)),
        {
            "unanimous_rate": float(np.mean(unanimous)),
            "modal_share": float(np.mean(modal)),
            "n_prompts": len(rows),
        },
    )


_NLI = {"contradiction", "entailment", "neutral"}


@register(
    category=_C,
    task="consistency",
    name="Contradiction rate",
    definition="Share of compared response pairs (repeated samples, or answers to related questions) that an NLI "
    "model or judge labels as contradicting each other (SelfCheckGPT-NLI style).",
    formula="#contradiction / #pairs",
    range="[0, 1]",
    input_requirements=("pair_labels",),
    references=(_REF_MANAKUL,),
    higher_is_better=False,
)
def contradiction_rate(pair_labels: Any) -> MetricResult:
    """``pair_labels``: per compared pair, ``"contradiction"``, ``"entailment"`` or ``"neutral"`` (or a list of
    such labels per prompt; all pairs are pooled)."""
    items = seq(pair_labels, "pair_labels")
    flat = []
    for i, v in enumerate(items):
        for lab in v if isinstance(v, (list, tuple)) else [v]:
            if lab not in _NLI:
                raise InputValidationError(f"pair_labels[{i}] must be one of {sorted(_NLI)}; got {lab!r}.")
            flat.append(lab == "contradiction")
    if not flat:
        raise InputValidationError("pair_labels contains no pairs.")
    return rate("contradiction_rate", "Contradiction rate", np.array(flat))


_STATUSES = ("ok", "error", "timeout")


@register(
    category=_C,
    task="reliability",
    name="Failure / timeout / error rate",
    definition="Share of requests that did not complete normally, split into errors and timeouts (any other "
    "status label is counted as its own failure kind).",
    formula="(#error + #timeout + …) / #requests",
    range="[0, 1]",
    input_requirements=("statuses",),
    references=(_REF_HELM,),
    higher_is_better=False,
)
def error_rate(statuses: Any, *, ok: Sequence[Any] = ("ok", "success", 200, True)) -> MetricResult:
    """``statuses``: one status per request (``"ok"``, ``"error"``, ``"timeout"``, an HTTP code, ...). Values in
    ``ok`` count as success."""
    items = seq(statuses, "statuses")
    oks = set(ok)
    fail = np.array([s not in oks for s in items])
    counts = Counter(str(s) for s in items if s not in oks)
    n = len(items)
    return rate(
        "error_rate",
        "Failure rate",
        fail,
        {"by_status": {k: v / n for k, v in sorted(counts.items())}, "timeout_rate": counts.get("timeout", 0) / n},
    )


@register(
    category=_C,
    task="reliability",
    name="Recovery success rate",
    definition="Among episodes in which a failure occurred (an error, a failed tool call, a wrong intermediate "
    "step), the share from which the system recovered and still completed the task.",
    formula="#(failed ∧ recovered) / #failed",
    range="[0, 1]",
    input_requirements=("failed", "recovered"),
    references=(_REF_HELM,),
)
def recovery_success_rate(failed: Any, recovered: Any) -> MetricResult:
    f, r = _pair(failed, recovered, "failed", "recovered")
    if (r & ~f).any():
        raise InputValidationError("recovered is True for an episode where failed is False.")
    if not f.any():
        raise InputValidationError("No episode failed: the recovery rate is undefined.")
    return rate("recovery_success_rate", "Recovery success rate", r[f], {"failure_rate": float(f.mean())})


@register(
    category=_C,
    task="robustness",
    name="Prompt wording sensitivity",
    definition="Spread of task performance across semantically equivalent prompt templates: the range (best − "
    "worst accuracy), standard deviation and worst-case accuracy over templates (FormatSpread).",
    formula="spread = max_t acc_t − min_t acc_t",
    range="[0, 1]",
    input_requirements=("correct_by_template",),
    references=(_REF_SCLAR, _REF_PROMPTBENCH),
    higher_is_better=False,
)
def prompt_sensitivity(correct_by_template: Any) -> MetricResult:
    """``correct_by_template``: mapping template name -> per-example correctness (or scores), the same examples
    under every template; or a 2-D array (templates × examples)."""
    if isinstance(correct_by_template, Mapping):
        names = [str(k) for k in correct_by_template]
        mat = [
            floats(np.asarray(v, dtype=float), f"correct_by_template[{k!r}]")
            for k, v in correct_by_template.items()
        ]
    else:
        arr = np.asarray(correct_by_template, dtype=float)
        if arr.ndim != 2:
            raise InputValidationError(
                "correct_by_template must be a mapping or a 2-D array (templates × examples)."
            )
        names = [str(i) for i in range(arr.shape[0])]
        mat = [floats(row, f"correct_by_template[{i}]") for i, row in enumerate(arr)]
    if len(mat) < 2:
        raise InputValidationError("Need at least two prompt templates.")
    if len({m.size for m in mat}) != 1:
        raise InputValidationError("Every template must be scored on the same examples.")
    acc = np.array([m.mean() for m in mat])
    per_example = np.vstack(mat)
    flips = float(np.mean(per_example.max(0) != per_example.min(0)))
    return MetricResult(
        "prompt_sensitivity",
        "Prompt sensitivity (spread)",
        float(acc.max() - acc.min()),
        {
            "std": float(acc.std(ddof=1)),
            "worst": float(acc.min()),
            "best": float(acc.max()),
            "mean": float(acc.mean()),
            "by_template": dict(zip(names, acc.tolist())),
            "example_disagreement_rate": flips,
        },
    )


@register(
    category=_C,
    task="long-context",
    name="Long-context robustness / truncation sensitivity",
    definition="How performance changes with context length or truncation: accuracy per length bucket and the "
    "least-squares slope of accuracy against log2 length (negative = degrades as context grows).",
    formula="slope of acc_b on log2(len_b)",
    range="(−∞, ∞)",
    input_requirements=("correct", "context_lengths"),
    references=(_REF_HSIEH,),
    higher_is_better=True,
)
def truncation_sensitivity(correct: Any, context_lengths: Any) -> MetricResult:
    c = floats(np.asarray(seq(correct, "correct"), dtype=float), "correct")
    lens = floats(context_lengths, "context_lengths", lo=1)
    same_length(("correct", c), ("context_lengths", lens))
    levels = np.unique(lens)
    if levels.size < 2:
        raise InputValidationError("Need at least two distinct context lengths.")
    acc = np.array([c[lens == v].mean() for v in levels])
    x = np.log2(levels)
    slope = float(np.polyfit(x, acc, 1)[0])
    return MetricResult(
        "truncation_sensitivity",
        "Accuracy slope per doubling of context",
        slope,
        {
            "by_length": {str(int(v) if float(v).is_integer() else v): float(a) for v, a in zip(levels, acc)},
            "drop_shortest_to_longest": float(acc[0] - acc[-1]),
        },
    )
