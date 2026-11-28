"""Agreement studies: how well raters, instruments or methods agree on the same subjects.

>>> import evalsuite as es
>>> icc = es.icc(ratings)                    # subjects × raters matrix → all six ICC forms with CIs
>>> icc["ICC2"].value, icc["ICC2"].low, icc["ICC2"].high
>>> ba = es.bland_altman(method_a, method_b) # bias, limits of agreement (with CIs), proportional bias
>>> ccc = es.lins_ccc(method_a, method_b)    # Lin's concordance correlation with a CI

ICC forms follow Shrout & Fleiss (1979) and McGraw & Wong (1996), with the exact F-based confidence intervals
(ICC2 by Satterthwaite's approximation), as in pingouin and R's ``psych::ICC``. Bland–Altman limits use the
Bland & Altman (1999) standard errors; Lin's CCC uses the Fisher-z interval of Lin (1989).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError
from .core.validation import numeric

__all__ = ["ICCResult", "ICCForm", "BlandAltman", "icc", "bland_altman", "lins_ccc"]

_ICC_DESCRIPTIONS = {
    "ICC1": "One-way random effects, single rater (absolute agreement)",
    "ICC2": "Two-way random effects, single rater (absolute agreement)",
    "ICC3": "Two-way mixed effects, single rater (consistency)",
    "ICC1k": "One-way random effects, mean of k raters",
    "ICC2k": "Two-way random effects, mean of k raters (absolute agreement)",
    "ICC3k": "Two-way mixed effects, mean of k raters (consistency)",
}


def _level(level: float) -> float:
    if not 0 < level < 1:
        raise InputValidationError(f"level must be in (0, 1); got {level}.")
    return float(level)


@dataclass(frozen=True)
class ICCForm:
    """One intraclass correlation: estimate, F test against zero and confidence interval."""

    name: str
    description: str
    value: float
    f: float
    df1: float
    df2: float
    p_value: float
    low: float
    high: float


@dataclass(frozen=True)
class ICCResult:
    """All six ICC forms for a subjects × raters table. Index by form name (``result["ICC2"]``)."""

    forms: tuple[ICCForm, ...]
    n_subjects: int
    n_raters: int
    level: float
    anova: dict[str, float] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def __getitem__(self, name: str) -> ICCForm:
        for f in self.forms:
            if f.name.lower() == name.lower():
                return f
        raise KeyError(f"{name!r}; available: {', '.join(f.name for f in self.forms)}")

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        head = (
            f"Intraclass correlation: {self.n_subjects} subjects × {self.n_raters} raters, "
            f"{self.level * 100:g}% CIs"
        )
        lines = [head]
        for f in self.forms:
            lines.append(
                f"  {f.name:<6} {f.value:.{d}f} ({f.low:.{d}f}–{f.high:.{d}f})  "
                f"F={f.f:.{d}f}  p={f.p_value:.{d}g}  {f.description}"
            )
        lines.append("  Koo & Li (2016): <0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, >0.9 excellent.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "forms": [f.__dict__ for f in self.forms],
                "n_subjects": self.n_subjects,
                "n_raters": self.n_raters,
                "level": self.level,
                "anova": self.anova,
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Form | ICC | CI | F | p | Meaning |", "| --- | ---: | --- | ---: | ---: | --- |"]
        rows += [
            f"| {f.name} | {f.value:.{d}f} | {f.low:.{d}f}–{f.high:.{d}f} | {f.f:.{d}f} | {f.p_value:.{d}g} | "
            f"{f.description} |"
            for f in self.forms
        ]
        return "\n".join(rows) + "\n"


def _table(ratings: Any, subjects: Any, raters: Any) -> np.ndarray:
    if subjects is None and raters is None:
        try:
            x = np.asarray(ratings, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise InputValidationError("ratings must be numeric.") from exc
        if x.ndim != 2:
            raise InputValidationError(
                "ratings must be a subjects × raters table (2-D), or long format with subjects= and raters=."
            )
        return x
    if subjects is None or raters is None:
        raise InputValidationError("Long format needs both subjects= and raters=.")
    vals = numeric(ratings, "ratings").ravel()
    s = np.asarray(subjects).ravel()
    r = np.asarray(raters).ravel()
    if not (vals.size == s.size == r.size):
        raise InputValidationError(
            f"ratings, subjects and raters must have the same length; got {vals.size}, {s.size} and {r.size}."
        )
    su, si = np.unique(s, return_inverse=True)
    ru, ri = np.unique(r, return_inverse=True)
    x = np.full((su.size, ru.size), np.nan)
    seen = np.zeros_like(x, dtype=bool)
    for a, b, v in zip(si, ri, vals):
        if seen[a, b]:
            raise InputValidationError(f"Subject {su[a]!r} has more than one rating from rater {ru[b]!r}.")
        seen[a, b] = True
        x[a, b] = v
    return x


def icc(
    ratings: Any,
    *,
    subjects: Any = None,
    raters: Any = None,
    level: float = 0.95,
    missing: str = "drop",
) -> ICCResult:
    """Intraclass correlation coefficients (all six Shrout–Fleiss forms) with F tests and confidence intervals.

    ``ratings`` is a subjects × raters table, or one rating per row with ``subjects=`` and ``raters=``
    (long format). Subjects with a missing rating are dropped (``missing="drop"``) or rejected
    (``missing="raise"``). Choose the form by design: ICC2 / ICC2k when the raters are a sample of possible raters
    and absolute agreement matters, ICC3 / ICC3k for these fixed raters' consistency, ICC1 when each subject has
    different raters; "k" forms describe the mean of the k ratings."""
    from scipy import stats

    lv = _level(level)
    x = _table(ratings, subjects, raters)
    bad = ~np.all(np.isfinite(x), axis=1)
    if bad.any():
        if missing != "drop":
            raise InputValidationError(f"{int(bad.sum())} subject(s) have missing or non-finite ratings.")
        x = x[~bad]
    n, k = x.shape
    if n < 2 or k < 2:
        raise InputValidationError(f"ICC needs at least 2 complete subjects and 2 raters; got {n} × {k}.")
    gm = x.mean()
    ss_r = k * float(np.sum((x.mean(axis=1) - gm) ** 2))
    ss_c = n * float(np.sum((x.mean(axis=0) - gm) ** 2))
    ss_t = float(np.sum((x - gm) ** 2))
    ss_e = ss_t - ss_r - ss_c
    msr, msc = ss_r / (n - 1), ss_c / (k - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    msw = (ss_c + ss_e) / (n * (k - 1))
    if msr == 0 and mse == 0 and msw == 0:
        raise InputValidationError("All ratings are identical; the ICC is undefined.")
    if msr == 0:
        raise InputValidationError("Subjects do not differ in their mean rating; the ICC is undefined.")
    a = (1 - lv) / 2
    df1, df2_1, df2_3 = n - 1, n * (k - 1), (n - 1) * (k - 1)

    def ratio(num: float, den: float) -> float:
        if den == 0:
            return math.inf if num > 0 else math.nan
        return num / den

    with np.errstate(divide="ignore", invalid="ignore"):
        icc1 = ratio(msr - msw, msr + (k - 1) * msw)
        icc2 = ratio(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n)
        icc3 = ratio(msr - mse, msr + (k - 1) * mse)
        icc1k = ratio(msr - msw, msr)
        icc2k = ratio(msr - mse, msr + (msc - mse) / n)
        icc3k = ratio(msr - mse, msr)
        f1 = ratio(msr, msw)
        f3 = ratio(msr, mse)
        f1l = f1 / stats.f.ppf(1 - a, df1, df2_1)
        f1u = f1 * stats.f.ppf(1 - a, df2_1, df1)
        f3l = f3 / stats.f.ppf(1 - a, df1, df2_3)
        f3u = f3 * stats.f.ppf(1 - a, df2_3, df1)
        l1, u1 = (f1l - 1) / (f1l + k - 1), (f1u - 1) / (f1u + k - 1)
        l3, u3 = (f3l - 1) / (f3l + k - 1), (f3u - 1) / (f3u + k - 1)
        fj = ratio(msc, mse)
        vn = (k - 1) * (n - 1) * (k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2) ** 2
        vd = (n - 1) * k**2 * icc2**2 * fj**2 + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2
        v = vn / vd
        f2u = stats.f.ppf(1 - a, n - 1, v)
        f2l = stats.f.ppf(1 - a, v, n - 1)
        l2 = n * (msr - f2u * mse) / (f2u * (k * msc + (k * n - k - n) * mse) + n * msr)
        u2 = n * (f2l * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f2l * msr)
    p1 = float(stats.f.sf(f1, df1, df2_1))
    p3 = float(stats.f.sf(f3, df1, df2_3))
    rows = [
        ("ICC1", icc1, f1, df1, df2_1, p1, l1, u1),
        ("ICC2", icc2, f3, df1, df2_3, p3, l2, u2),
        ("ICC3", icc3, f3, df1, df2_3, p3, l3, u3),
        ("ICC1k", icc1k, f1, df1, df2_1, p1, 1 - 1 / f1l, 1 - 1 / f1u),
        ("ICC2k", icc2k, f3, df1, df2_3, p3, l2 * k / (1 + l2 * (k - 1)), u2 * k / (1 + u2 * (k - 1))),
        ("ICC3k", icc3k, f3, df1, df2_3, p3, 1 - 1 / f3l, 1 - 1 / f3u),
    ]
    forms = tuple(
        ICCForm(nm, _ICC_DESCRIPTIONS[nm], float(val), float(f), float(d1), float(d2), p, float(lo), float(hi))
        for nm, val, f, d1, d2, p, lo, hi in rows
    )
    anova = {"ms_subjects": msr, "ms_raters": msc, "ms_error": mse, "ms_within": msw}
    return ICCResult(forms, n, k, lv, anova)


@dataclass(frozen=True)
class BlandAltman:
    """Bland–Altman analysis of two measurement methods on the same subjects."""

    n: int
    bias: float
    bias_low: float
    bias_high: float
    sd: float
    lower_loa: float
    upper_loa: float
    lower_loa_ci: tuple[float, float]
    upper_loa_ci: tuple[float, float]
    proportional_bias_slope: float
    proportional_bias_p: float
    level: float
    loa_level: float
    within_loa: float
    means: np.ndarray = field(repr=False)
    differences: np.ndarray = field(repr=False)

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        pct = f"{self.level * 100:g}%"
        loa = f"{self.loa_level * 100:g}%"
        return "\n".join(
            [
                f"Bland–Altman analysis (n={self.n}; {pct} CIs; {loa} limits of agreement)",
                f"  Bias (mean difference)  {self.bias:.{d}f}  ({self.bias_low:.{d}f} to {self.bias_high:.{d}f})",
                f"  Lower limit            {self.lower_loa:.{d}f}  "
                f"({self.lower_loa_ci[0]:.{d}f} to {self.lower_loa_ci[1]:.{d}f})",
                f"  Upper limit            {self.upper_loa:.{d}f}  "
                f"({self.upper_loa_ci[0]:.{d}f} to {self.upper_loa_ci[1]:.{d}f})",
                f"  Differences within limits: {self.within_loa * 100:.1f}%",
                f"  Proportional bias: slope {self.proportional_bias_slope:.{d}f} of difference on mean "
                f"(p={self.proportional_bias_p:.{d}g})",
            ]
        )

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {k: v for k, v in self.__dict__.items() if k not in ("means", "differences")}
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Quantity | Estimate | CI |", "| --- | ---: | --- |"]
        rows.append(f"| Bias | {self.bias:.{d}f} | {self.bias_low:.{d}f} to {self.bias_high:.{d}f} |")
        lo, hi = self.lower_loa_ci, self.upper_loa_ci
        rows.append(f"| Lower limit of agreement | {self.lower_loa:.{d}f} | {lo[0]:.{d}f} to {lo[1]:.{d}f} |")
        rows.append(f"| Upper limit of agreement | {self.upper_loa:.{d}f} | {hi[0]:.{d}f} to {hi[1]:.{d}f} |")
        return "\n".join(rows) + "\n"

    def plot(self, ax: Any = None) -> Any:
        """Difference against mean with bias and limits of agreement (Matplotlib)."""
        from .core.exceptions import OptionalDependencyError

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise OptionalDependencyError("matplotlib", "plot", "BlandAltman.plot") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(self.means, self.differences, s=12, alpha=0.7)
        for y, style in ((self.bias, "-"), (self.lower_loa, "--"), (self.upper_loa, "--")):
            ax.axhline(y, linestyle=style, color="0.3", linewidth=1)
        ax.set_xlabel("Mean of the two methods")
        ax.set_ylabel("Difference (A − B)")
        ax.set_title("Bland–Altman")
        return ax


def _pair(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        x = np.asarray(a, dtype=np.float64).ravel()
        y = np.asarray(b, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise InputValidationError("Both methods' measurements must be numeric.") from exc
    if x.size != y.size:
        raise InputValidationError(f"The two methods must measure the same subjects; got {x.size} and {y.size}.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise InputValidationError("Measurements must be finite (drop missing pairs first).")
    if x.size < 3:
        raise InputValidationError("Agreement needs at least 3 paired measurements.")
    return x, y


def bland_altman(method_a: Any, method_b: Any, *, level: float = 0.95, loa_level: float = 0.95) -> BlandAltman:
    """Bias and limits of agreement between two methods (Bland & Altman 1986), with confidence intervals for the
    bias (t) and for each limit (Bland & Altman 1999), and a test for proportional bias (regression of the
    difference on the mean)."""
    from scipy import stats

    lv, ll = _level(level), _level(loa_level)
    a, b = _pair(method_a, method_b)
    d = a - b
    m = (a + b) / 2
    n = d.size
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    z = float(stats.norm.ppf(1 - (1 - ll) / 2))
    t = float(stats.t.ppf(1 - (1 - lv) / 2, n - 1))
    se_bias = sd / math.sqrt(n)
    se_loa = sd * math.sqrt(1 / n + z**2 / (2 * (n - 1)))
    lo, hi = bias - z * sd, bias + z * sd
    if np.ptp(m) > 0:
        reg = stats.linregress(m, d)
        slope, p = float(reg.slope), float(reg.pvalue)
    else:
        slope, p = math.nan, math.nan
    return BlandAltman(
        n=n,
        bias=bias,
        bias_low=bias - t * se_bias,
        bias_high=bias + t * se_bias,
        sd=sd,
        lower_loa=lo,
        upper_loa=hi,
        lower_loa_ci=(lo - t * se_loa, lo + t * se_loa),
        upper_loa_ci=(hi - t * se_loa, hi + t * se_loa),
        proportional_bias_slope=slope,
        proportional_bias_p=p,
        level=lv,
        loa_level=ll,
        within_loa=float(np.mean((d >= lo) & (d <= hi))),
        means=m,
        differences=d,
    )


def lins_ccc(method_a: Any, method_b: Any, *, level: float = 0.95) -> Any:
    """Lin's concordance correlation coefficient (agreement with the identity line) with the Fisher-z confidence
    interval of Lin (1989); returns a :class:`~evalsuite.MetricResult`."""
    from scipy import stats

    from .core.result import MetricResult

    lv = _level(level)
    x, y = _pair(method_a, method_b)
    n = x.size
    sx2, sy2 = x.var(), y.var()  # maximum-likelihood variances, as in Lin (1989)
    sxy = float(np.mean((x - x.mean()) * (y - y.mean())))
    denom = sx2 + sy2 + (x.mean() - y.mean()) ** 2
    if denom == 0:
        raise InputValidationError("Both methods are constant and equal; the CCC is undefined.")
    ccc = 2 * sxy / denom
    r = sxy / math.sqrt(sx2 * sy2) if sx2 > 0 and sy2 > 0 else math.nan
    params: dict[str, Any] = {"pearson_r": r, "n": n, "ci_level": lv}
    if abs(ccc) < 1 and np.isfinite(r) and abs(r) < 1:
        u = (x.mean() - y.mean()) / (sx2 * sy2) ** 0.25
        var_z = (
            (1 - r**2) * ccc**2 / ((1 - ccc**2) * r**2)
            + 2 * ccc**3 * (1 - ccc) * u**2 / (r * (1 - ccc**2) ** 2)
            - ccc**4 * u**4 / (2 * r**2 * (1 - ccc**2) ** 2)
        ) / (n - 2)
        zc = math.atanh(ccc)
        half = float(stats.norm.ppf(1 - (1 - lv) / 2)) * math.sqrt(max(var_z, 0.0))
        params.update(ci_low=math.tanh(zc - half), ci_high=math.tanh(zc + half), ci_method="Fisher z (Lin 1989)")
    return MetricResult(name="Lin's concordance correlation", value=float(ccc), metric="lins_ccc", params=params)
