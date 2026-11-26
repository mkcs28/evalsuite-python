"""Study planning: how many test items are needed for a given precision or to detect a given difference.

* :func:`sample_size_proportion` — items so that a proportion (accuracy, sensitivity, ...) has a confidence
  interval no wider than ``± half_width``.
* :func:`sample_size_two_proportions` — items to detect a difference between two models' accuracies, on the same
  items (paired, McNemar; Connor 1987) or on separate samples (Fleiss, Levin & Paik 2003).
* :func:`sample_size_auc` — positives and negatives for an ROC AUC interval of given width, or to show an AUC
  above a reference value (Hanley & McNeil 1982; Obuchowski 1998).
* :func:`sample_size_mean_difference` — pairs (or per-group observations) to detect a difference in a mean
  score or loss (paired or two-sample t approximation).
* ``power_*`` counterparts give the power achieved with a fixed sample.

All formulas are the standard normal-approximation ones and are documented with their references; sample sizes
are rounded up.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from scipy import optimize, stats

from ..core.exceptions import InputValidationError

__all__ = [
    "SampleSize",
    "power_auc",
    "power_mean_difference",
    "power_two_proportions",
    "sample_size_auc",
    "sample_size_mean_difference",
    "sample_size_proportion",
    "sample_size_two_proportions",
]


@dataclass(frozen=True)
class SampleSize:
    """A planned sample size with the assumptions behind it."""

    n: int
    what: str
    method: str
    assumptions: Mapping[str, Any] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __int__(self) -> int:
        return self.n

    def __index__(self) -> int:
        return self.n

    def summary(self) -> str:
        a = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.assumptions.items())
        extra = "".join(f"\n  {k}: {v}" for k, v in self.details.items())
        return f"n = {self.n} {self.what} ({self.method}; {a}){extra}"

    def __repr__(self) -> str:
        return self.summary()


def _check_prob(name: str, p: float, *, open_: bool = True) -> float:
    p = float(p)
    lo_ok = p > 0 if open_ else p >= 0
    hi_ok = p < 1 if open_ else p <= 1
    if not (math.isfinite(p) and lo_ok and hi_ok):
        raise InputValidationError(f"{name} must be strictly between 0 and 1; got {p}.")
    return p


def _z_alpha(alpha: float, two_sided: bool) -> float:
    _check_prob("alpha", alpha)
    return float(stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha))


def _z_power(power: float) -> float:
    _check_prob("power", power)
    return float(stats.norm.ppf(power))


def _ceil(x: float) -> int:
    # guard against 30.000000000000004 → 31
    return math.ceil(x - 1e-9)


# --------------------------------------------------------------------------------------------- precision


def sample_size_proportion(p: float, half_width: float, *, level: float = 0.95) -> SampleSize:
    """Items so that a proportion near ``p`` is estimated within ``± half_width`` (normal approximation,
    n = z² p (1 − p) / w²). Use p = 0.5 when nothing is known (the most conservative choice)."""
    p = _check_prob("p", p)
    w = _check_prob("half_width", half_width)
    z = float(stats.norm.ppf(1 - (1 - _check_prob("level", level)) / 2))
    n = z * z * p * (1 - p) / (w * w)
    return SampleSize(
        _ceil(n),
        "items",
        "normal approximation to the binomial",
        {"p": p, "half_width": w, "level": level},
    )


# --------------------------------------------------------------------------------------------- two proportions


def _n_two_props(
    p1: float, p2: float, alpha: float, power: float, paired: bool, discordance: Optional[float], two_sided: bool
) -> tuple[float, dict[str, Any]]:
    d = p2 - p1
    if d == 0:
        raise InputValidationError("p1 and p2 must differ: there is no difference to detect.")
    za, zb = _z_alpha(alpha, two_sided), _z_power(power)
    if paired:
        # Connor (1987): ψ = P(discordant) = p10 + p01, δ = p01 − p10 = p2 − p1.
        psi = discordance if discordance is not None else p1 * (1 - p2) + p2 * (1 - p1)
        psi = _check_prob("discordance", psi)
        if psi <= abs(d):
            raise InputValidationError(
                f"discordance ({psi:g}) must exceed the difference |p2 − p1| ({abs(d):g}): at least that share "
                "of items must be answered differently by the two models."
            )
        n = (za * math.sqrt(psi) + zb * math.sqrt(psi - d * d)) ** 2 / (d * d)
        return n, {"discordance": psi}
    # Fleiss, Levin & Paik (2003), equal group sizes, no continuity correction.
    pbar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (d * d)
    return n, {}


def sample_size_two_proportions(
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = True,
    discordance: Optional[float] = None,
    two_sided: bool = True,
) -> SampleSize:
    """Items needed to detect accuracy ``p1`` vs ``p2``.

    ``paired=True`` (default; both models answer the same items, McNemar's test, Connor 1987):
    n = (z_α √ψ + z_β √(ψ − δ²))² / δ², with ψ the share of items the models disagree on (``discordance``).
    Without ``discordance``, ψ assumes independent errors, p1(1 − p2) + p2(1 − p1), which is conservative when
    the models are correlated (they usually are; a pilot estimate of ψ gives a smaller n).
    ``paired=False``: two separate samples, n per group (Fleiss, Levin & Paik 2003)."""
    p1, p2 = _check_prob("p1", p1), _check_prob("p2", p2)
    n, extra = _n_two_props(p1, p2, alpha, power, paired, discordance, two_sided)
    return SampleSize(
        _ceil(n),
        "items" if paired else "items per group",
        "McNemar, Connor (1987)" if paired else "two independent proportions, Fleiss et al. (2003)",
        {"p1": p1, "p2": p2, "alpha": alpha, "power": power, "two_sided": two_sided, **extra},
    )


def power_two_proportions(
    n: int,
    p1: float,
    p2: float,
    *,
    alpha: float = 0.05,
    paired: bool = True,
    discordance: Optional[float] = None,
    two_sided: bool = True,
) -> float:
    """Power to detect ``p1`` vs ``p2`` with ``n`` items (paired) or ``n`` per group (unpaired)."""
    p1, p2 = _check_prob("p1", p1), _check_prob("p2", p2)
    if n < 1:
        raise InputValidationError("n must be a positive integer.")
    d = abs(p2 - p1)
    za = _z_alpha(alpha, two_sided)
    if paired:
        psi = _check_prob("discordance", discordance if discordance is not None else p1 * (1 - p2) + p2 * (1 - p1))
        if psi <= d:
            raise InputValidationError("discordance must exceed |p2 − p1|.")
        z = (d * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - d * d)
    else:
        pbar = (p1 + p2) / 2
        z = (d * math.sqrt(n) - za * math.sqrt(2 * pbar * (1 - pbar))) / math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------------------------- ROC AUC


def _auc_var(auc: float, n_pos: float, n_neg: float) -> float:
    """Hanley & McNeil (1982) variance of the AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    return (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)


def sample_size_auc(
    auc: float,
    *,
    prevalence: float,
    half_width: Optional[float] = None,
    auc0: Optional[float] = None,
    level: float = 0.95,
    alpha: float = 0.05,
    power: float = 0.8,
) -> SampleSize:
    """Total items (positives + negatives at ``prevalence``) for an ROC AUC near ``auc``.

    Give ``half_width`` for an interval no wider than ``± half_width``, or ``auc0`` to show that the AUC exceeds a
    reference value (one-sided, at ``alpha`` with ``power``; variance under both hypotheses, Obuchowski 1998).
    Variances follow Hanley & McNeil (1982)."""
    auc = _check_prob("auc", auc)
    prev = _check_prob("prevalence", prevalence)
    if (half_width is None) == (auc0 is None):
        raise InputValidationError("Give exactly one of half_width (precision) or auc0 (a test against it).")

    def counts(n: float) -> tuple[float, float]:
        return max(n * prev, 1.0), max(n * (1 - prev), 1.0)

    if half_width is not None:
        w = _check_prob("half_width", half_width)
        z = float(stats.norm.ppf(1 - (1 - _check_prob("level", level)) / 2))

        def gap(n: float) -> float:
            return z * math.sqrt(_auc_var(auc, *counts(n))) - w

        method = "precision of the AUC, Hanley & McNeil (1982)"
        assumptions: dict[str, Any] = {"auc": auc, "prevalence": prev, "half_width": w, "level": level}
    else:
        a0 = _check_prob("auc0", float(auc0))  # type: ignore[arg-type]
        if a0 >= auc:
            raise InputValidationError("auc0 must be below the expected auc.")
        za, zb = _z_alpha(alpha, False), _z_power(power)

        def gap(n: float) -> float:
            npos, nneg = counts(n)
            return (
                za * math.sqrt(_auc_var(a0, npos, nneg)) + zb * math.sqrt(_auc_var(auc, npos, nneg)) - (auc - a0)
            )

        method = "one-sided test of AUC > auc0, Obuchowski (1998)"
        assumptions = {"auc": auc, "auc0": a0, "prevalence": prev, "alpha": alpha, "power": power}
    lo, hi = 2.0, 10.0
    while gap(hi) > 0:
        hi *= 2
        if hi > 1e12:
            raise InputValidationError("The requested precision needs an unrealistically large sample.")
    n = optimize.brentq(gap, lo, hi) if gap(lo) > 0 else lo
    total = _ceil(n)
    npos = _ceil(total * prev)
    return SampleSize(
        total,
        "items",
        method,
        assumptions,
        {"positives": npos, "negatives": total - npos},
    )


def power_auc(n: int, auc: float, auc0: float, *, prevalence: float, alpha: float = 0.05) -> float:
    """Power of a one-sided test that the AUC exceeds ``auc0`` with ``n`` items at ``prevalence``."""
    auc, a0, prev = _check_prob("auc", auc), _check_prob("auc0", auc0), _check_prob("prevalence", prevalence)
    npos, nneg = max(n * prev, 1.0), max(n * (1 - prev), 1.0)
    za = _z_alpha(alpha, False)
    z = ((auc - a0) - za * math.sqrt(_auc_var(a0, npos, nneg))) / math.sqrt(_auc_var(auc, npos, nneg))
    return float(stats.norm.cdf(z))


# --------------------------------------------------------------------------------------------- mean scores


def sample_size_mean_difference(
    delta: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    paired: bool = True,
    two_sided: bool = True,
) -> SampleSize:
    """Items to detect a difference ``delta`` in a mean score (a loss, a rating, a per-item metric).

    ``paired=True``: ``sd`` is the standard deviation of the per-item differences; n = ((z_α + z_β) σ / δ)²,
    refined with the t distribution. ``paired=False``: ``sd`` is the per-group standard deviation; n per group =
    2 ((z_α + z_β) σ / δ)², refined likewise."""
    delta, sd = float(delta), float(sd)
    if not (math.isfinite(delta) and delta != 0 and math.isfinite(sd) and sd > 0):
        raise InputValidationError("delta must be a non-zero number and sd a positive number.")
    za, zb = _z_alpha(alpha, two_sided), _z_power(power)
    factor = 1.0 if paired else 2.0
    n = factor * ((za + zb) * sd / delta) ** 2
    for _ in range(50):  # t refinement: degrees of freedom from the current n
        df = (n - 1) if paired else 2 * (n - 1)
        df = max(df, 1.0)
        ta = float(stats.t.ppf(1 - alpha / 2 if two_sided else 1 - alpha, df))
        tb = float(stats.t.ppf(power, df))
        new = factor * ((ta + tb) * sd / delta) ** 2
        if abs(new - n) < 1e-6:
            break
        n = new
    return SampleSize(
        _ceil(n),
        "items" if paired else "items per group",
        "paired t" if paired else "two-sample t",
        {"delta": delta, "sd": sd, "alpha": alpha, "power": power, "two_sided": two_sided},
    )


def power_mean_difference(
    n: int, delta: float, sd: float, *, alpha: float = 0.05, paired: bool = True, two_sided: bool = True
) -> float:
    """Power of a paired (or two-sample, ``n`` per group) t-test to detect ``delta`` (noncentral t)."""
    if n < 2:
        raise InputValidationError("n must be at least 2.")
    if not (sd > 0 and delta != 0):
        raise InputValidationError("delta must be non-zero and sd positive.")
    df = (n - 1) if paired else 2 * (n - 1)
    nc = abs(delta) / sd * math.sqrt(n if paired else n / 2)
    crit = float(stats.t.ppf(1 - alpha / 2 if two_sided else 1 - alpha, df))
    p = 1 - stats.nct.cdf(crit, df, nc)
    if two_sided:
        p += stats.nct.cdf(-crit, df, nc)
    return float(p)
