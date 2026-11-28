"""Clinical model updates: does a new risk model reclassify patients better, and does a model transport to a new
site?

>>> import evalsuite as es
>>> es.reclassification(y, p_old, p_new, cutoffs=[0.05, 0.2])   # NRI (categorical and continuous) and IDI
>>> es.external_validation(y_dev, p_dev, y_ext, p_ext)          # development vs external site, and recalibration

Net reclassification improvement (Pencina et al., Stat Med 2008): among events, the share moved to a higher risk
category minus the share moved lower; among non-events, the share moved lower minus higher; the NRI is their sum.
The continuous (category-free) NRI counts any increase or decrease (Pencina et al. 2011). The integrated
discrimination improvement (IDI) is the gain in mean predicted risk among events minus that among non-events.
P-values use Pencina's asymptotic z tests; intervals are percentile bootstrap (paired, stratified by outcome).

``external_validation`` reports discrimination (DeLong CIs and the unpaired difference), calibration-in-the-large,
calibration slope, observed / expected ratio and Brier score at both sites, and how much intercept-only and
logistic recalibration on the external data would recover (Steyerberg, Clinical Prediction Models, ch. 19).
"""

from __future__ import annotations

# ruff: noqa: E501  (long f-strings of report text)
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError
from .core.validation import numeric

__all__ = ["ReclassificationResult", "reclassification", "ExternalValidation", "external_validation"]


def _binary(y: Any, *ps: Any) -> tuple[np.ndarray, list[np.ndarray]]:
    yy = np.asarray(y).ravel()
    labels = np.unique(yy)
    if not set(labels.tolist()) <= {0, 1}:
        raise InputValidationError("y must be a binary outcome coded 0 / 1.")
    out = []
    for k, p in enumerate(ps):
        a = numeric(p, f"risk {k + 1}").ravel()
        if a.size != yy.size:
            raise InputValidationError(
                f"Predicted risks must have one value per patient ({yy.size}); got {a.size}."
            )
        if not np.all(np.isfinite(a)) or np.any((a < 0) | (a > 1)):
            raise InputValidationError("Predicted risks must be probabilities in [0, 1].")
        out.append(a)
    y01 = yy.astype(int)
    if y01.min() == y01.max():
        raise InputValidationError("Both events and non-events are needed.")
    return y01, out


def _nri_parts(y: np.ndarray, old: np.ndarray, new: np.ndarray, cut: Optional[np.ndarray]) -> dict[str, float]:
    if cut is None:
        up, down = new > old, new < old
    else:
        co, cn = np.searchsorted(cut, old, side="right"), np.searchsorted(cut, new, side="right")
        up, down = cn > co, cn < co
    e, ne = y == 1, y == 0
    ne_n, e_n = ne.sum(), e.sum()
    pu_e, pd_e = up[e].mean(), down[e].mean()
    pu_n, pd_n = up[ne].mean(), down[ne].mean()
    return {
        "events": float(pu_e - pd_e),
        "nonevents": float(pd_n - pu_n),
        "nri": float(pu_e - pd_e + pd_n - pu_n),
        "pu_e": float(pu_e),
        "pd_e": float(pd_e),
        "pu_n": float(pu_n),
        "pd_n": float(pd_n),
        "n_e": float(e_n),
        "n_n": float(ne_n),
    }


def _idi(y: np.ndarray, old: np.ndarray, new: np.ndarray) -> float:
    e, ne = y == 1, y == 0
    return float((new[e].mean() - old[e].mean()) - (new[ne].mean() - old[ne].mean()))


@dataclass(frozen=True)
class ReclassificationResult:
    """NRI (categorical and continuous) and IDI with p-values and bootstrap intervals."""

    rows: tuple[dict[str, Any], ...]
    table: Optional[dict[str, Any]]
    settings: dict[str, Any]

    def __getitem__(self, name: str) -> dict[str, Any]:
        for r in self.rows:
            if r["name"] == name:
                return r
        raise KeyError(f"{name!r}; available: {', '.join(r['name'] for r in self.rows)}")

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        lines = [
            f"Reclassification: new vs old model (n={s['n']}, events={s['events']}, {s['level'] * 100:g}% "
            "bootstrap CIs)"
        ]
        for r in self.rows:
            lines.append(
                f"  {r['name']:<28} {r['value']:+.{d}f} ({r['low']:+.{d}f} to {r['high']:+.{d}f})  p={r['p_value']:.{d}g}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe({"rows": list(self.rows), "table": self.table, "settings": self.settings})
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Measure | Estimate | CI | p |", "| --- | ---: | --- | ---: |"]
        rows += [
            f"| {r['name']} | {r['value']:+.{d}f} | {r['low']:+.{d}f} to {r['high']:+.{d}f} | {r['p_value']:.{d}g} |"
            for r in self.rows
        ]
        if self.table:
            t = self.table
            for grp in ("events", "nonevents"):
                rows += ["", f"**Reclassification table, {grp}** (rows: old category, columns: new)", ""]
                rows.append("| Old \\ New | " + " | ".join(t["labels"]) + " |")
                rows.append("| --- |" + " ---: |" * len(t["labels"]))
                for lab, line in zip(t["labels"], t[grp]):
                    rows.append(f"| {lab} | " + " | ".join(str(v) for v in line) + " |")
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def reclassification(
    y: Any,
    p_old: Any,
    p_new: Any,
    *,
    cutoffs: Optional[Sequence[float]] = None,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: Optional[int] = None,
) -> ReclassificationResult:
    """NRI and IDI of a new risk model against an old one on the same patients (see the module docstring).
    ``cutoffs`` (e.g. ``[0.05, 0.2]``) define risk categories for the categorical NRI and the reclassification
    tables; the continuous NRI and the IDI are always reported."""
    from scipy import stats

    yy, (old, new) = _binary(y, p_old, p_new)
    cut = None
    if cutoffs is not None:
        cut = np.sort(numeric(list(cutoffs), "cutoffs"))
        if cut.size == 0 or np.any((cut <= 0) | (cut >= 1)):
            raise InputValidationError("cutoffs must be risks strictly between 0 and 1.")
    if not 0 < level < 1 or n_resamples < 100:
        raise InputValidationError("level must be in (0, 1) and n_resamples at least 100.")
    measures: list[tuple[str, Any]] = []
    if cut is not None:
        measures.append(("NRI (categorical)", lambda yv, o, n_: _nri_parts(yv, o, n_, cut)["nri"]))
    measures += [
        ("NRI (continuous)", lambda yv, o, n_: _nri_parts(yv, o, n_, None)["nri"]),
        ("IDI", _idi),
    ]
    rng = np.random.default_rng(random_state)
    ev, ne = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    draws = np.empty((n_resamples, len(measures)))
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(ev, ev.size), rng.choice(ne, ne.size)])
        for j, (_, fn) in enumerate(measures):
            draws[b, j] = fn(yy[idx], old[idx], new[idx])
    a = (1 - level) / 2
    rows = []
    for j, (name, fn) in enumerate(measures):
        value = float(fn(yy, old, new))
        if name.startswith("NRI"):
            parts = _nri_parts(yy, old, new, cut if "categorical" in name else None)
            var = (parts["pu_e"] + parts["pd_e"] - (parts["pu_e"] - parts["pd_e"]) ** 2) / parts["n_e"] + (
                parts["pu_n"] + parts["pd_n"] - (parts["pu_n"] - parts["pd_n"]) ** 2
            ) / parts["n_n"]
            extra = {"events_component": parts["events"], "nonevents_component": parts["nonevents"]}
        else:
            e, n0 = yy == 1, yy == 0
            de, dn = new[e] - old[e], new[n0] - old[n0]
            var = de.var(ddof=1) / de.size + dn.var(ddof=1) / dn.size
            extra = {}
        z = value / math.sqrt(var) if var > 0 else math.inf if value else 0.0
        p = float(2 * stats.norm.sf(abs(z))) if math.isfinite(z) else 0.0
        lo, hi = np.quantile(draws[:, j], [a, 1 - a])
        rows.append(
            {"name": name, "value": value, "low": float(lo), "high": float(hi), "p_value": p, "z": z, **extra}
        )
    table = None
    if cut is not None:
        labels = [f"<{cut[0]:g}"] + [f"{cut[i]:g}–{cut[i + 1]:g}" for i in range(cut.size - 1)] + [f"≥{cut[-1]:g}"]
        co, cn = np.searchsorted(cut, old, side="right"), np.searchsorted(cut, new, side="right")
        m = cut.size + 1

        def tab(mask: np.ndarray) -> list[list[int]]:
            t = np.zeros((m, m), dtype=int)
            np.add.at(t, (co[mask], cn[mask]), 1)
            return t.tolist()

        table = {"labels": labels, "events": tab(yy == 1), "nonevents": tab(yy == 0)}
    settings = {
        "n": int(yy.size),
        "events": int(yy.sum()),
        "cutoffs": None if cut is None else cut.tolist(),
        "level": level,
        "n_resamples": n_resamples,
    }
    return ReclassificationResult(tuple(rows), table, settings)


# ---------------------------------------------------------------------------------------- external validation


@dataclass(frozen=True)
class ExternalValidation:
    """Performance at the development and the external site, and what recalibration would recover."""

    rows: tuple[dict[str, Any], ...]
    recalibration: dict[str, Any]
    settings: dict[str, Any]

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        lines = [
            f"External validation: development n={s['n_dev']} ({s['events_dev']} events), external "
            f"n={s['n_ext']} ({s['events_ext']} events)",
            f"  {'Measure':<30} {'Development':>22} {'External':>22}",
        ]
        for r in self.rows:
            lines.append(f"  {r['name']:<30} {_fmt(r['dev'], d):>22} {_fmt(r['ext'], d):>22}")
        rc = self.recalibration
        lines.append(
            f"  Recalibration on the external data: intercept update → calibration-in-the-large "
            f"{rc['intercept_update']['calibration_in_the_large']:+.{d}f}; logistic (intercept {rc['logistic']['a']:.{d}f}, "
            f"slope {rc['logistic']['b']:.{d}f}) → Brier {rc['logistic']['brier']:.{d}f} (from {rc['original_brier']:.{d}f})"
        )
        if "auc_difference_p" in s:
            lines.append(
                f"  ROC AUC difference (external − development): p={s['auc_difference_p']:.{d}g} (unpaired DeLong)"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {"rows": list(self.rows), "recalibration": self.recalibration, "settings": self.settings}
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Measure | Development | External |", "| --- | --- | --- |"]
        rows += [f"| {r['name']} | {_fmt(r['dev'], d)} | {_fmt(r['ext'], d)} |" for r in self.rows]
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def _fmt(v: Any, d: int) -> str:
    if isinstance(v, (tuple, list)):
        est, lo, hi = v
        return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"
    return f"{v:.{d}f}"


def _site(y: np.ndarray, p: np.ndarray, level: float) -> dict[str, Any]:
    from .calibration import brier_score, calibration_intercept, calibration_slope
    from .stats.intervals import roc_auc_ci

    auc = roc_auc_ci(y, p, level=level)
    return {
        "roc_auc": (auc.estimate, auc.low, auc.high),
        "auc_se": (auc.high - auc.low) / (2 * 1.959963984540054),
        "citl": float(calibration_intercept(y, p).value),
        "slope": float(calibration_slope(y, p).value),
        "oe": float(y.mean() / p.mean()) if p.mean() > 0 else math.inf,
        "brier": float(brier_score(y, p).value),
        "prevalence": float(y.mean()),
        "mean_risk": float(p.mean()),
    }


def external_validation(
    y_dev: Any, p_dev: Any, y_ext: Any, p_ext: Any, *, level: float = 0.95
) -> ExternalValidation:
    """Compare a model's performance at development and at an external site (see the module docstring)."""
    from scipy import stats

    from .calibration import logistic_fit

    yd, (pd_,) = _binary(y_dev, p_dev)
    ye, (pe,) = _binary(y_ext, p_ext)
    dev, ext = _site(yd, pd_, level), _site(ye, pe, level)
    rows = [
        {"name": "ROC AUC", "dev": dev["roc_auc"], "ext": ext["roc_auc"]},
        {"name": "Calibration-in-the-large", "dev": dev["citl"], "ext": ext["citl"]},
        {"name": "Calibration slope", "dev": dev["slope"], "ext": ext["slope"]},
        {"name": "Observed / expected", "dev": dev["oe"], "ext": ext["oe"]},
        {"name": "Brier score", "dev": dev["brier"], "ext": ext["brier"]},
        {"name": "Outcome prevalence", "dev": dev["prevalence"], "ext": ext["prevalence"]},
        {"name": "Mean predicted risk", "dev": dev["mean_risk"], "ext": ext["mean_risk"]},
    ]
    q = np.clip(pe, 1e-12, 1 - 1e-12)
    lp = np.log(q / (1 - q))
    w = np.ones_like(lp)
    (a0,) = logistic_fit(ye.astype(float), lp, w, offset=True)
    p_int = 1 / (1 + np.exp(-(a0 + lp)))
    a1, b1 = logistic_fit(ye.astype(float), lp, w, offset=False)
    p_log = 1 / (1 + np.exp(-(a1 + b1 * lp)))
    recal = {
        "original_brier": ext["brier"],
        "intercept_update": {
            "a": float(a0),
            "brier": float(np.mean((p_int - ye) ** 2)),
            "calibration_in_the_large": float(ye.mean() - p_int.mean()),
        },
        "logistic": {"a": float(a1), "b": float(b1), "brier": float(np.mean((p_log - ye) ** 2))},
        "note": "Fitted and assessed on the same external data: an optimistic estimate of what updating recovers.",
    }
    diff = ext["roc_auc"][0] - dev["roc_auc"][0]
    se = math.hypot(dev["auc_se"], ext["auc_se"])
    settings = {
        "n_dev": int(yd.size),
        "events_dev": int(yd.sum()),
        "n_ext": int(ye.size),
        "events_ext": int(ye.sum()),
        "level": level,
        "auc_difference": diff,
        "auc_difference_p": float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else math.nan,
    }
    return ExternalValidation(tuple(rows), recal, settings)
