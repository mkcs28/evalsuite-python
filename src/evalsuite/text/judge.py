"""LLM-as-a-judge and preference evaluation: rubric scores, win rates, Bradley–Terry and Elo ratings,
inter-rater agreement and judge bias checks.

The judge itself (a model, a panel or human raters) is yours; these functions turn its outputs into
reported numbers with the usual EvalSuite guarantees (validated inputs, documented conventions, intervals).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
from scipy import stats

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import per_item

__all__ = [
    "bradley_terry",
    "elo_ratings",
    "fleiss_kappa",
    "judge_agreement",
    "krippendorff_alpha",
    "position_consistency",
    "rubric_score",
    "self_preference_bias",
    "verbosity_bias",
    "win_rate",
]

_C = "text"
_REF_BT = "Bradley RA, Terry ME. Rank analysis of incomplete block designs. Biometrika. 1952;39(3/4):324-345."
_REF_HUNTER = "Hunter DR. MM algorithms for generalized Bradley-Terry models. Ann Stat. 2004;32(1):384-406."
_REF_ELO = "Elo AE. The Rating of Chessplayers, Past and Present. Arco; 1978."
_REF_ARENA = (
    "Chiang WL, Zheng L, Sheng Y, et al. Chatbot Arena: an open platform for evaluating LLMs by human "
    "preference. ICML. 2024."
)
_REF_KRIPP = "Krippendorff K. Content Analysis: An Introduction to Its Methodology. 4th ed. Sage; 2018."
_REF_FLEISS = "Fleiss JL. Measuring nominal scale agreement among many raters. Psychol Bull. 1971;76(5):378-382."
_REF_JUDGE = (
    "Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS "
    "Datasets and Benchmarks. 2023."
)
_REF_SELF = (
    "Panickssery A, Bowman SR, Feng S. LLM evaluators recognize and favor their own generations. NeurIPS. 2024."
)
_REF_WILSON = (
    "Wilson EB. Probable inference, the law of succession, and statistical inference. JASA. 1927;22:209-212."
)


def _outcome(v: Any) -> float:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("win", "w", "a", "1"):
            return 1.0
        if s in ("loss", "lose", "l", "b", "0"):
            return 0.0
        if s in ("tie", "draw", "t", "0.5"):
            return 0.5
    elif isinstance(v, (bool, np.bool_)) or (
        isinstance(v, (int, float, np.integer, np.floating)) and float(v) in (0.0, 0.5, 1.0)
    ):
        return float(v)
    raise InputValidationError(f"Outcome must be 'win', 'loss', 'tie' (or 1, 0, 0.5); got {v!r}.")


def _wilson(k: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# ---------------------------------------------------------------- pairwise preferences
@register(
    category=_C,
    task="preference",
    name="Pairwise win rate",
    definition="Share of pairwise comparisons in which a system's response is preferred to the baseline; ties "
    "count half (or are excluded). A Wilson interval is reported.",
    formula="(wins + ½ ties) / comparisons",
    range="[0, 1]",
    input_requirements=("outcomes",),
    references=(_REF_JUDGE, _REF_WILSON),
)
@per_item
def win_rate(outcomes: Any, *, ties: str = "half") -> MetricResult:
    """Win rate from outcomes ``"win" / "loss" / "tie"`` (or 1 / 0 / 0.5) of the system against a baseline.
    ``ties="half"`` counts a tie as half a win; ``"exclude"`` drops ties."""
    vals = np.array([_outcome(v) for v in outcomes], dtype=float)
    if vals.size == 0:
        raise InputValidationError("outcomes is empty.")
    if ties == "exclude":
        vals = vals[vals != 0.5]
        if vals.size == 0:
            raise InputValidationError("Every comparison is a tie; the win rate excluding ties is undefined.")
    elif ties != "half":
        raise InputValidationError("ties must be 'half' or 'exclude'.")
    wins = float(vals.sum())
    low, high = _wilson(wins, vals.size)
    return MetricResult(
        "win_rate",
        "Win rate",
        wins / vals.size,
        {"ties": ties, "n": int(vals.size), "n_ties": int(np.sum(vals == 0.5)), "ci_low": low, "ci_high": high},
    )


def _comparisons(comparisons: Any) -> tuple[list[Any], list[tuple[int, int, float]]]:
    rows = list(comparisons)
    if not rows:
        raise InputValidationError("comparisons is empty.")
    names: dict[Any, int] = {}
    out = []
    for i, row in enumerate(rows):
        if len(row) != 3:
            raise InputValidationError(f"comparisons[{i}] must be (model_a, model_b, outcome for model_a).")
        a, b, o = row
        if a == b:
            raise InputValidationError(f"comparisons[{i}] compares {a!r} with itself.")
        ia, ib = names.setdefault(a, len(names)), names.setdefault(b, len(names))
        out.append((ia, ib, _outcome(o)))
    return list(names), out


@register(
    category=_C,
    task="preference",
    name="Bradley–Terry scores",
    definition="Maximum-likelihood strengths of a Bradley–Terry model fitted to pairwise preferences (ties as "
    "half a win for each side), by Hunter's MM algorithm; reported as log-strengths centred at 0 or on the "
    "Elo-like Chatbot Arena scale.",
    formula="P(i beats j) = π_i / (π_i + π_j)",
    range="(−∞, ∞)",
    input_requirements=("comparisons",),
    references=(_REF_BT, _REF_HUNTER, _REF_ARENA),
)
def bradley_terry(
    comparisons: Any, *, scale: str = "log", max_iter: int = 10_000, tol: float = 1e-10
) -> MetricResult:
    """Bradley–Terry ratings from ``(model_a, model_b, outcome)`` rows, outcome for ``model_a`` being
    ``"win" / "loss" / "tie"``. ``scale="log"`` returns natural-log strengths with mean 0 (as ``choix``);
    ``"elo"`` maps them to ``1000 + 400·log10(π)``. The comparison graph must be connected and no model may win
    or lose every comparison, otherwise the MLE does not exist."""
    names, rows = _comparisons(comparisons)
    m = len(names)
    wins = np.zeros((m, m))
    for a, b, o in rows:
        wins[a, b] += o
        wins[b, a] += 1 - o
    games = wins + wins.T
    total_w = wins.sum(axis=1)
    if np.any(total_w == 0) or np.any(total_w == games.sum(axis=1)):
        raise InputValidationError(
            "The Bradley–Terry MLE does not exist: some model wins or loses every comparison. Add comparisons or "
            "use a prior."
        )
    reach = (games > 0).astype(int)
    seen, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in np.flatnonzero(reach[i]):
            if j not in seen:
                seen.add(int(j))
                frontier.append(int(j))
    if len(seen) != m:
        raise InputValidationError(
            "The comparison graph is not connected; ratings are not comparable across parts."
        )
    p = np.ones(m)
    for _ in range(max_iter):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = total_w / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            break
        p = new
    logp = np.log(p) - np.mean(np.log(p))
    if scale == "log":
        value = logp
    elif scale == "elo":
        value = 1000 + 400 * logp / math.log(10)
    else:
        raise InputValidationError("scale must be 'log' or 'elo'.")
    return MetricResult(
        "bradley_terry",
        "Bradley–Terry score",
        np.asarray(value),
        {"scale": scale, "n_comparisons": len(rows)},
        labels=tuple(names),
    )


@register(
    category=_C,
    task="preference",
    name="Elo ratings",
    definition="Sequential Elo ratings from pairwise outcomes in the given order (expected score "
    "1/(1 + 10^((R_b − R_a)/400)), update K·(outcome − expected)); order-dependent, unlike Bradley–Terry.",
    formula="R_a ← R_a + K (S_a − E_a)",
    range="(−∞, ∞), starting at the initial rating",
    input_requirements=("comparisons",),
    references=(_REF_ELO, _REF_ARENA),
)
def elo_ratings(
    comparisons: Any, *, k: float = 4.0, initial: float = 1000.0, scale: float = 400.0, base: float = 10.0
) -> MetricResult:
    """Online Elo with Chatbot Arena's defaults (K = 4, initial 1000). Shuffle and average over several
    orders, or prefer ``bradley_terry``, when the order of comparisons is arbitrary."""
    names, rows = _comparisons(comparisons)
    if k <= 0 or scale <= 0 or base <= 1:
        raise InputValidationError("k and scale must be positive and base > 1.")
    r = np.full(len(names), float(initial))
    for a, b, o in rows:
        ea = 1 / (1 + base ** ((r[b] - r[a]) / scale))
        r[a] += k * (o - ea)
        r[b] += k * ((1 - o) - (1 - ea))
    return MetricResult(
        "elo_ratings",
        "Elo rating",
        r,
        {"k": k, "initial": initial, "n_comparisons": len(rows)},
        labels=tuple(names),
    )


# ---------------------------------------------------------------- agreement
def _reliability_matrix(ratings: Any) -> np.ndarray:
    a = np.array(ratings, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 1:
        raise InputValidationError(
            "ratings must be a 2-D array: one row per rater, one column per item (NaN = missing)."
        )
    return a


@register(
    category=_C,
    task="agreement",
    name="Krippendorff's alpha",
    definition="Chance-corrected agreement among any number of raters with missing ratings, for nominal, "
    "ordinal, interval or ratio data.",
    formula="α = 1 − D_observed / D_expected",
    range="(−∞, 1]",
    input_requirements=("ratings",),
    references=(_REF_KRIPP,),
)
def krippendorff_alpha(ratings: Any, *, level: str = "nominal") -> MetricResult:
    """Krippendorff's alpha from a raters × items matrix (``np.nan`` for missing), identical to the
    ``krippendorff`` package. ``level`` is ``"nominal"``, ``"ordinal"``, ``"interval"`` or ``"ratio"``."""
    a = _reliability_matrix(ratings)
    if level not in ("nominal", "ordinal", "interval", "ratio"):
        raise InputValidationError("level must be 'nominal', 'ordinal', 'interval' or 'ratio'.")
    values = np.unique(a[~np.isnan(a)])
    if values.size < 2:
        raise InputValidationError("Krippendorff's alpha needs at least two distinct rating values.")
    v = values.size
    # counts[u, c]: how many raters gave item u the value c (vectorised coincidence matrix)
    mask = ~np.isnan(a)
    items = np.broadcast_to(np.arange(a.shape[1]), a.shape)[mask]
    codes = np.searchsorted(values, a[mask])
    counts = np.zeros((a.shape[1], v))
    np.add.at(counts, (items, codes), 1)
    m_u = counts.sum(axis=1)
    pairable = m_u >= 2
    paired = counts[pairable]
    scaled = paired / (m_u[pairable] - 1)[:, None]
    coinc = scaled.T @ paired - np.diag(scaled.sum(axis=0))
    n_c = coinc.sum(axis=1)
    n = n_c.sum()
    if n == 0:
        raise InputValidationError("No item has ratings from two or more raters.")
    if level == "nominal":
        delta = 1.0 - np.eye(v)
    elif level == "interval":
        delta = (values[:, None] - values[None, :]) ** 2
    elif level == "ratio":
        s = values[:, None] + values[None, :]
        delta = np.where(s == 0, 0.0, ((values[:, None] - values[None, :]) / np.where(s == 0, 1, s)) ** 2)
    else:
        cum = np.cumsum(n_c)
        delta = np.zeros((v, v))
        for c in range(v):
            for k_ in range(v):
                lo, hi = min(c, k_), max(c, k_)
                delta[c, k_] = (cum[hi] - (cum[lo - 1] if lo > 0 else 0) - (n_c[c] + n_c[k_]) / 2) ** 2
    d_o = float((coinc * delta).sum() / n)
    d_e = float((np.outer(n_c, n_c) * delta).sum() / (n * (n - 1)))
    if d_e == 0:
        raise InputValidationError("Expected disagreement is zero; alpha is undefined.")
    return MetricResult(
        "krippendorff_alpha",
        "Krippendorff's alpha",
        1 - d_o / d_e,
        {"level": level, "n_raters": int(a.shape[0]), "n_items": int(a.shape[1])},
    )


@register(
    category=_C,
    task="agreement",
    name="Fleiss' kappa",
    definition="Chance-corrected agreement of a fixed number of raters assigning items to nominal categories.",
    formula="κ = (P̄ − P̄_e) / (1 − P̄_e)",
    range="(−∞, 1]",
    input_requirements=("ratings",),
    references=(_REF_FLEISS,),
)
def fleiss_kappa(ratings: Any) -> MetricResult:
    """Fleiss' kappa from an items × raters matrix of category labels (every item rated by the same number of
    raters), identical to ``statsmodels``' ``fleiss_kappa`` on the aggregated table."""
    a = np.asarray(ratings, dtype=object)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2:
        raise InputValidationError("ratings must be a 2-D array: one row per item, one column per rater (>= 2).")
    cats = sorted({str(x) for x in a.ravel()})
    table = np.array([[sum(str(x) == c for x in row) for c in cats] for row in a], dtype=float)
    n_raters = a.shape[1]
    p_j = table.sum(axis=0) / table.sum()
    p_i = ((table * table).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), float((p_j**2).sum())
    if p_e == 1:
        raise InputValidationError("All ratings fall in one category; Fleiss' kappa is undefined.")
    return MetricResult(
        "fleiss_kappa",
        "Fleiss' kappa",
        (p_bar - p_e) / (1 - p_e),
        {"n_items": int(a.shape[0]), "n_raters": int(n_raters), "categories": cats},
    )


@register(
    category=_C,
    task="agreement",
    name="Judge–human agreement",
    definition="Agreement between an automatic judge and human ratings: Cohen's kappa for categories, "
    "quadratic-weighted kappa for ordinal scores (with Spearman's ρ), Pearson / Spearman correlation for "
    "continuous scores; raw agreement is reported alongside.",
    formula="kappa or correlation, by data type",
    range="[-1, 1]",
    input_requirements=("judge", "human"),
    references=(_REF_JUDGE,),
)
def judge_agreement(judge: Any, human: Any, *, kind: str = "categorical") -> MetricResult:
    """Agreement between judge and human labels or scores. ``kind`` is ``"categorical"``, ``"ordinal"`` or
    ``"continuous"``."""
    j, h = list(judge), list(human)
    if not j or len(j) != len(h):
        raise InputValidationError("judge and human must be non-empty and the same length.")
    from ..classification.metrics import cohen_kappa

    if kind == "categorical":
        k = float(cohen_kappa(h, j))
        return MetricResult(
            "judge_agreement",
            "Judge–human Cohen's kappa",
            k,
            {"kind": kind, "raw_agreement": float(np.mean([a == b for a, b in zip(j, h)])), "n": len(j)},
        )
    ja, ha = np.asarray(j, dtype=float), np.asarray(h, dtype=float)
    if kind == "ordinal":
        k = float(cohen_kappa(ha, ja, weights="quadratic"))
        rho = float(stats.spearmanr(ja, ha).statistic)
        return MetricResult(
            "judge_agreement",
            "Judge–human weighted kappa",
            k,
            {"kind": kind, "spearman": rho, "raw_agreement": float(np.mean(ja == ha)), "n": len(j)},
        )
    if kind == "continuous":
        r = float(stats.pearsonr(ja, ha).statistic)
        return MetricResult(
            "judge_agreement",
            "Judge–human Pearson r",
            r,
            {"kind": kind, "spearman": float(stats.spearmanr(ja, ha).statistic), "n": len(j)},
        )
    raise InputValidationError("kind must be 'categorical', 'ordinal' or 'continuous'.")


# ---------------------------------------------------------------- judge reliability and bias
def _ab(v: Any) -> str:
    s = str(v).strip().lower()
    if s in ("a", "1", "first"):
        return "a"
    if s in ("b", "2", "second"):
        return "b"
    if s in ("tie", "draw", "t", "0.5"):
        return "tie"
    raise InputValidationError(f"Verdict must be 'A', 'B' or 'tie'; got {v!r}.")


@register(
    category=_C,
    task="judge-reliability",
    name="Position consistency",
    definition="Share of pairwise judgements that stay the same when the two responses are presented in swapped "
    "order; the inconsistent cases that favour whichever response was shown first measure position bias.",
    formula="mean[verdict(A, B) = verdict(B, A)]",
    range="[0, 1]",
    input_requirements=("verdicts_original", "verdicts_swapped"),
    references=(_REF_JUDGE,),
)
def position_consistency(verdicts_original: Any, verdicts_swapped: Any) -> MetricResult:
    """Both verdict lists name the winner by identity (``"A"`` is always the same response, whether it was
    shown first or second). ``params["first_position_rate"]`` is the share of inconsistent pairs where the judge
    chose the response shown first both times."""
    o = [_ab(v) for v in verdicts_original]
    s = [_ab(v) for v in verdicts_swapped]
    if not o or len(o) != len(s):
        raise InputValidationError("verdicts_original and verdicts_swapped must be non-empty and the same length.")
    same = [a == b for a, b in zip(o, s)]
    inconsistent = [(a, b) for a, b in zip(o, s) if a != b]
    first = sum(a == "a" and b == "b" for a, b in inconsistent)
    second = sum(a == "b" and b == "a" for a, b in inconsistent)
    return MetricResult(
        "position_consistency",
        "Position consistency",
        float(np.mean(same)),
        {
            "n": len(o),
            "n_inconsistent": len(inconsistent),
            "first_position_rate": first / len(inconsistent) if inconsistent else float("nan"),
            "second_position_rate": second / len(inconsistent) if inconsistent else float("nan"),
        },
    )


@register(
    category=_C,
    task="judge-reliability",
    name="Verbosity bias",
    definition="Share of decisive judgements (no ties, different lengths) won by the longer response, with a "
    "two-sided binomial test against 0.5; well above 0.5 suggests a preference for length.",
    formula="wins of longer response / decisive comparisons",
    range="[0, 1]",
    input_requirements=("winners", "length_a", "length_b"),
    references=(_REF_JUDGE,),
)
def verbosity_bias(winners: Any, length_a: Any, length_b: Any) -> MetricResult:
    """``winners`` holds ``"A" / "B" / "tie"`` per comparison; lengths in tokens or characters. Pair it with
    human labels on the same comparisons before calling a length preference a bias."""
    w = [_ab(v) for v in winners]
    la, lb = np.asarray(length_a, float).ravel(), np.asarray(length_b, float).ravel()
    if not w or la.size != len(w) or lb.size != len(w):
        raise InputValidationError("winners, length_a and length_b must be non-empty and the same length.")
    longer_wins = [(x == "a") == (a > b) for x, a, b in zip(w, la, lb) if x != "tie" and a != b]
    if not longer_wins:
        raise InputValidationError("No decisive comparison between responses of different length.")
    k, n = int(sum(longer_wins)), len(longer_wins)
    p = float(stats.binomtest(k, n, 0.5).pvalue)
    return MetricResult("verbosity_bias", "Longer-response win rate", k / n, {"n_decisive": n, "p_value": p})


@register(
    category=_C,
    task="judge-reliability",
    name="Self-preference bias",
    definition="How much more often a judge prefers its own model's outputs than human raters do on the same "
    "comparisons.",
    formula="judge win rate of own outputs − human win rate of own outputs",
    range="[-1, 1]",
    input_requirements=("judge_prefers_own", "human_prefers_own"),
    references=(_REF_SELF,),
    higher_is_better=None,
)
def self_preference_bias(judge_prefers_own: Any, human_prefers_own: Any) -> MetricResult:
    """Difference in win rate of the judge's own model, judge versus humans; McNemar's exact p-value on the
    paired verdicts is reported in ``params``."""
    j = np.asarray(judge_prefers_own, dtype=bool).ravel()
    h = np.asarray(human_prefers_own, dtype=bool).ravel()
    if j.size == 0 or j.shape != h.shape:
        raise InputValidationError(
            "judge_prefers_own and human_prefers_own must be non-empty and the same length."
        )
    b, c = int(np.sum(j & ~h)), int(np.sum(~j & h))
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return MetricResult(
        "self_preference_bias",
        "Self-preference bias",
        float(j.mean() - h.mean()),
        {"judge_rate": float(j.mean()), "human_rate": float(h.mean()), "p_value": p, "n": int(j.size)},
    )


# ---------------------------------------------------------------- rubric scores
@register(
    category=_C,
    task="rubric",
    name="Rubric score",
    definition="Mean of rubric ratings (correctness, helpfulness, relevance, coherence, fluency, completeness, "
    "clarity, conciseness, tone, instruction adherence, reasoning quality, ...) rescaled to [0, 1], with the "
    "mean per criterion.",
    formula="mean((score − min) / (max − min))",
    range="[0, 1]",
    input_requirements=("scores",),
    references=(_REF_JUDGE,),
)
@per_item
def rubric_score(
    scores: Any, *, scale: tuple[float, float] = (1, 5), criteria: Optional[Sequence[str]] = None
) -> MetricResult:
    """Rubric ratings as an ``(n, n_criteria)`` array (or one column). Returns the overall normalised mean;
    ``params["per_criterion"]`` maps each criterion to its normalised mean."""
    a = np.asarray(scores, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.size == 0:
        raise InputValidationError("scores must be a 2-D array: one row per response, one column per criterion.")
    lo, hi = scale
    if hi <= lo:
        raise InputValidationError("scale must be (min, max) with max > min.")
    if np.any(np.isnan(a)) or a.min() < lo or a.max() > hi:
        raise InputValidationError(f"scores must lie within the scale {scale} without missing values.")
    names = list(criteria) if criteria is not None else [f"criterion_{i + 1}" for i in range(a.shape[1])]
    if len(names) != a.shape[1]:
        raise InputValidationError("criteria must name every column.")
    norm = (a - lo) / (hi - lo)
    return MetricResult(
        "rubric_score",
        "Rubric score",
        float(norm.mean()),
        {"scale": scale, "per_criterion": dict(zip(names, norm.mean(axis=0).tolist())), "n": int(a.shape[0])},
    )
