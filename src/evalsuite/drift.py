"""Data drift between a reference set (training or validation data) and current data, feature by feature.

>>> import evalsuite as es
>>> rep = es.drift_report(train_df, live_df)              # DataFrames, dicts of columns or 2-D arrays
>>> rep.drifted()                                         # features that changed significantly and materially
>>> print(rep.summary()); rep.to_html()

Numeric features: two-sample Kolmogorov–Smirnov test, population stability index (PSI, reference-quantile bins)
and Wasserstein distance in reference standard deviations. Categorical features (strings, booleans, or
``categorical=[...]``): chi-square test of homogeneity, PSI over categories and Jensen–Shannon distance.
Missing-value rates are compared with a two-proportion z-test. P-values are Holm-adjusted across features. A
feature is flagged when its adjusted p-value is below ``alpha`` *and* the shift is material (PSI ≥
``psi_threshold`` or distance ≥ ``distance_threshold``: 0.1 reference SDs, or a Jensen–Shannon distance of 0.1),
so a large data set does not flag trivial shifts; a significant change of at least one point in the missing-value
rate is flagged too. Model scores and labels can be checked the
same way (``es.drift_report({"score": ref_scores}, {"score": live_scores})``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError

__all__ = ["DriftReport", "drift_report"]


def _columns(data: Any, names: Optional[Sequence[str]]) -> dict[str, np.ndarray]:
    if hasattr(data, "columns") and hasattr(data, "__getitem__") and not isinstance(data, np.ndarray):
        cols = {str(c): np.asarray(data[c]) for c in data.columns}  # pandas / polars
    elif isinstance(data, Mapping):
        cols = {str(k): np.asarray(v) for k, v in data.items()}
    else:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise InputValidationError("Data must be a DataFrame, a mapping of columns, or a 1-D / 2-D array.")
        labels = list(names) if names is not None else [f"x{i}" for i in range(arr.shape[1])]
        if len(labels) != arr.shape[1]:
            raise InputValidationError(f"{len(labels)} feature names for {arr.shape[1]} columns.")
        cols = {str(labels[i]): arr[:, i] for i in range(arr.shape[1])}
    for k, v in cols.items():
        if v.ndim != 1:
            raise InputValidationError(f"Feature {k!r} must be one-dimensional.")
    return cols


def _missing(v: np.ndarray) -> np.ndarray:
    if v.dtype.kind in "fc":
        return ~np.isfinite(v)
    if v.dtype.kind == "O":
        return np.array([x is None or (isinstance(x, float) and math.isnan(x)) for x in v], dtype=bool)
    return np.zeros(v.size, dtype=bool)


def _is_categorical(v: np.ndarray) -> bool:
    return v.dtype.kind in "OUSb"


def _psi(rp: np.ndarray, cp: np.ndarray, eps: float) -> float:
    rp, cp = np.maximum(rp, eps), np.maximum(cp, eps)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def _numeric(r: np.ndarray, c: np.ndarray, bins: int, eps: float) -> dict[str, Any]:
    from scipy import stats

    from .monitor import population_stability_index

    ks = stats.ks_2samp(r, c)
    sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    w = float(stats.wasserstein_distance(r, c))
    return {
        "type": "numeric",
        "test": "Kolmogorov–Smirnov",
        "statistic": float(ks.statistic),
        "p_value": float(ks.pvalue),
        "psi": population_stability_index(r, c, bins=bins, eps=eps),
        "distance": w / sd if sd > 0 else (0.0 if w == 0 else math.inf),
        "distance_name": "Wasserstein / reference SD",
        "reference_mean": float(np.mean(r)),
        "current_mean": float(np.mean(c)),
    }


def _categorical(r: np.ndarray, c: np.ndarray, eps: float) -> dict[str, Any]:
    from scipy import stats
    from scipy.spatial.distance import jensenshannon

    rs, cs = r.astype(str), c.astype(str)
    cats = np.unique(np.concatenate([rs, cs]))
    rn = np.array([np.sum(rs == k) for k in cats], dtype=float)
    cn = np.array([np.sum(cs == k) for k in cats], dtype=float)
    table = np.vstack([rn, cn])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        stat, p = 0.0, 1.0
    else:
        res = stats.chi2_contingency(table, correction=False)
        stat, p = float(res.statistic), float(res.pvalue)
    rp, cp = rn / rn.sum(), cn / cn.sum()
    top = cats[np.argmax(np.abs(cp - rp))]
    return {
        "type": "categorical",
        "test": "chi-square",
        "statistic": stat,
        "p_value": p,
        "psi": _psi(rp, cp, eps),
        "distance": float(jensenshannon(rp, cp, base=2)),
        "distance_name": "Jensen–Shannon",
        "categories": int(cats.size),
        "largest_change": str(top),
        "new_categories": sorted(set(cs.tolist()) - set(rs.tolist()))[:10],
    }


def _two_prop_p(x1: int, n1: int, x2: int, n2: int) -> float:
    from scipy import stats

    p = (x1 + x2) / (n1 + n2)
    if p in (0.0, 1.0):
        return 1.0
    z = (x1 / n1 - x2 / n2) / math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float(2 * stats.norm.sf(abs(z)))


@dataclass(frozen=True)
class DriftReport:
    """Per-feature drift results. ``features`` rows carry the test, statistic, raw and Holm-adjusted p-values,
    PSI, a distance, missing-rate change and whether the feature drifted."""

    features: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def drifted(self) -> list[str]:
        """Names of the drifted features, largest PSI first."""
        rows = sorted((f for f in self.features if f["drifted"]), key=lambda f: -f["psi"])
        return [f["feature"] for f in rows]

    def __getitem__(self, feature: str) -> dict[str, Any]:
        for f in self.features:
            if f["feature"] == feature:
                return f
        raise KeyError(feature)

    @property
    def share_drifted(self) -> float:
        return sum(f["drifted"] for f in self.features) / len(self.features)

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        head = (
            f"Drift report: {len(self.features)} features, "
            f"reference n={s['n_reference']}, current n={s['n_current']}"
            f" — {len(self.drifted())} drifted (Holm-adjusted p < {s['alpha']:g} and PSI ≥ {s['psi_threshold']:g} "
            f"or distance ≥ {s['distance_threshold']:g})"
        )
        lines = [head]
        for f in sorted(self.features, key=lambda f: -f["psi"]):
            flag = "DRIFT" if f["drifted"] else "ok"
            lines.append(
                f"  {flag:<5} {f['feature']:<24} {f['type']:<11} PSI {f['psi']:.{d}f}  "
                f"{f['distance_name']} {f['distance']:.{d}f}  p_adj {f['p_adjusted']:.{d}g}"
                + (
                    f"  missing {f['missing_reference']:.1%} → {f['missing_current']:.1%}"
                    if f["missing_shift"]
                    else ""
                )
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe({"features": list(self.features), "settings": dict(self.settings)})
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_dataframe(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self.to_dict()["features"])

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = [
            "| Feature | Type | Drifted | PSI | Distance | Statistic | p (Holm) | Missing ref → cur |",
            "| --- | --- | :---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for f in sorted(self.features, key=lambda f: -f["psi"]):
            rows.append(
                f"| {f['feature']} | {f['type']} | {'yes' if f['drifted'] else 'no'} | {f['psi']:.{d}f} | "
                f"{f['distance']:.{d}f} | {f['statistic']:.{d}f} | {f['p_adjusted']:.{d}g} | "
                f"{f['missing_reference']:.1%} → {f['missing_current']:.1%} |"
            )
        return "\n".join(rows) + "\n"

    def to_html(self) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown())


def drift_report(
    reference: Any,
    current: Any,
    *,
    features: Optional[Sequence[str]] = None,
    categorical: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    psi_threshold: float = 0.1,
    distance_threshold: float = 0.1,
    bins: int = 10,
    eps: float = 1e-4,
) -> DriftReport:
    """Compare every shared feature of ``reference`` and ``current`` (see the module docstring). ``features``
    selects and orders features (or names the columns of 2-D arrays); ``categorical`` forces features to be
    treated as categories (for example integer codes)."""
    from .stats.effect import adjust_pvalues

    if not 0 < alpha < 1:
        raise InputValidationError("alpha must be in (0, 1).")
    ref = _columns(reference, features if not hasattr(reference, "columns") else None)
    cur = _columns(current, features if not hasattr(current, "columns") else None)
    names = [f for f in (features or list(ref)) if f in ref]
    missing_cols = [f for f in names if f not in cur]
    if missing_cols:
        raise InputValidationError(f"Current data lacks feature(s): {', '.join(missing_cols[:5])}.")
    if not names:
        raise InputValidationError("No features to compare.")
    cat = set(categorical or ())
    rows: list[dict[str, Any]] = []
    for name in names:
        r_all, c_all = ref[name], cur[name]
        mr, mc = _missing(r_all), _missing(c_all)
        r, c = r_all[~mr], c_all[~mc]
        if r.size < 2 or c.size < 1:
            raise InputValidationError(f"Feature {name!r} has too few non-missing values to compare.")
        if name in cat or _is_categorical(r) or _is_categorical(c):
            row = _categorical(r, c, eps)
        else:
            row = _numeric(r.astype(float), c.astype(float), bins, eps)
        mp = _two_prop_p(int(mr.sum()), mr.size, int(mc.sum()), mc.size)
        row.update(
            feature=name,
            n_reference=int(r.size),
            n_current=int(c.size),
            missing_reference=float(mr.mean()),
            missing_current=float(mc.mean()),
            missing_p=mp,
        )
        rows.append(row)
    adj = adjust_pvalues([r["p_value"] for r in rows], method="holm")
    madj = adjust_pvalues([r["missing_p"] for r in rows], method="holm")
    for r, a, m in zip(rows, adj, madj):
        r["p_adjusted"] = float(a)
        r["missing_shift"] = bool(m < alpha and abs(r["missing_current"] - r["missing_reference"]) >= 0.01)
        material = r["psi"] >= psi_threshold or r["distance"] >= distance_threshold
        r["drifted"] = bool((a < alpha and material) or r["missing_shift"])
    n_ref = len(next(iter(ref.values())))
    n_cur = len(next(iter(cur.values())))
    settings = {
        "alpha": alpha,
        "psi_threshold": psi_threshold,
        "distance_threshold": distance_threshold,
        "bins": bins,
        "n_reference": n_ref,
        "n_current": n_cur,
        "correction": "holm",
    }
    return DriftReport(tuple(rows), settings)
