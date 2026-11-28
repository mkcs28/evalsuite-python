"""Conformal prediction: prediction sets and intervals with a finite-sample coverage guarantee.

>>> import evalsuite as es
>>> sets = es.conformal_classification(y_cal, p_cal, p_test, alpha=0.1)            # 90% prediction sets
>>> sets.sets[0], sets.mean_size
>>> es.conformal_classification(y_cal, p_cal, p_test, y_test=y_test).coverage      # check on labelled data
>>> iv = es.conformal_regression(y_cal, yhat_cal, yhat_test, alpha=0.1)            # 90% intervals
>>> iv.lower, iv.upper

Split conformal prediction (Vovk et al. 2005; Lei et al. 2018; Angelopoulos & Bates 2023): a held-out calibration
set gives the score threshold q̂ = the ⌈(n + 1)(1 − α)⌉-th smallest calibration score, and for a new exchangeable
example P(y ∈ set) ≥ 1 − α, whatever the model. Classification scores: ``"lac"`` (1 − probability of the true
class; smallest sets), ``"aps"`` (adaptive prediction sets, Romano et al. 2020; sets that grow with uncertainty)
and ``"raps"`` (regularised APS, Angelopoulos et al. 2021). Regression: ``"absolute"`` residuals, ``"normalized"``
residuals divided by a per-example scale (e.g. a predicted SD) and ``"cqr"`` (conformalized quantile regression,
Romano et al. 2019) from lower / upper quantile predictions. ``by=`` gives Mondrian (group-conditional) conformal
prediction: a threshold per group, so coverage holds within every group (classes, sites, subgroups).

With labelled test data, the result reports empirical coverage with a Wilson interval, the coverage per group and
per set size, and set sizes / interval widths — the checks reviewers ask for.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError
from .core.validation import numeric

__all__ = [
    "ConformalSets",
    "ConformalIntervals",
    "conformal_classification",
    "conformal_regression",
    "conformal_quantile",
]


def conformal_quantile(scores: Any, alpha: float) -> float:
    """The split-conformal threshold: the ⌈(n + 1)(1 − α)⌉-th smallest score (infinity when that exceeds n, i.e.
    the calibration set is too small for this α)."""
    s = numeric(scores, "scores").ravel()
    if s.size == 0 or not np.all(np.isfinite(s)):
        raise InputValidationError("Calibration scores must be finite and non-empty.")
    if not 0 < alpha < 1:
        raise InputValidationError(f"alpha must be in (0, 1); got {alpha}.")
    k = math.ceil((s.size + 1) * (1 - alpha))
    if k > s.size:
        return math.inf
    return float(np.partition(s, k - 1)[k - 1])


def _groups(by: Any, n: int, what: str) -> np.ndarray:
    g = np.asarray(by)
    if g.ndim != 1 or g.size != n:
        raise InputValidationError(f"{what} must have one group per example ({n}); got {g.size}.")
    return g


def _cov_text(c: Mapping[str, Any], d: int) -> str:
    return f"{c['coverage']:.{d}f} ({c['low']:.{d}f}–{c['high']:.{d}f}), n={c['n']}"


def _coverage_block(hit: np.ndarray, level: float) -> dict[str, Any]:
    from .stats.intervals import proportion_ci

    k, n = int(hit.sum()), int(hit.size)
    ci = proportion_ci(k, n, level=level)
    return {"coverage": k / n, "low": ci.low, "high": ci.high, "n": n}


# ---------------------------------------------------------------------------------------------- classification


def _class_scores(p: np.ndarray, method: str, lam: float, k_reg: int) -> np.ndarray:
    """Non-conformity score of every class for every row (n × k): the set at threshold q is {j : score ≤ q}."""
    if method == "lac":
        return 1.0 - p
    order = np.argsort(-p, axis=1, kind="stable")
    sorted_p = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    if method == "raps":
        ranks = np.arange(1, p.shape[1] + 1)
        cum = cum + lam * np.maximum(0, ranks - k_reg)[None, :]
    out = np.empty_like(p)
    np.put_along_axis(out, order, cum, axis=1)
    return out


@dataclass(frozen=True)
class ConformalSets:
    """Conformal prediction sets for classification."""

    sets: np.ndarray = field(repr=False)  # n_test × k booleans
    classes: tuple[Any, ...]
    threshold: Any
    alpha: float
    method: str
    n_calibration: int
    coverage: Optional[dict[str, Any]] = None
    by_group: Optional[dict[str, dict[str, Any]]] = None
    by_size: Optional[dict[int, dict[str, Any]]] = None

    @property
    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def sizes(self) -> np.ndarray:
        return np.asarray(self.sets.sum(axis=1))

    @property
    def mean_size(self) -> float:
        return float(self.sizes.mean())

    @property
    def singleton_rate(self) -> float:
        return float(np.mean(self.sizes == 1))

    @property
    def empty_rate(self) -> float:
        return float(np.mean(self.sizes == 0))

    def labels(self, i: int) -> list[Any]:
        """The labels in the prediction set of test example ``i``."""
        return [c for c, keep in zip(self.classes, self.sets[i]) if keep]

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        lines = [
            f"Conformal prediction sets ({self.method}, target coverage {1 - self.alpha:.{d}g}, "
            f"{self.n_calibration} calibration examples)",
            f"  Mean set size {self.mean_size:.{d}f}; singletons {self.singleton_rate:.1%}; empty "
            f"{self.empty_rate:.1%}",
        ]
        if self.coverage:
            c = self.coverage
            lines.append(
                f"  Empirical coverage {c['coverage']:.{d}f} ({c['low']:.{d}f}–{c['high']:.{d}f}), n={c['n']}"
            )
        for title, table in (("group", self.by_group), ("set size", self.by_size)):
            if table:
                lines.append(f"  Coverage by {title}:")
                for k, c in table.items():
                    lines.append(
                        f"    {k!s:<12} {c['coverage']:.{d}f} ({c['low']:.{d}f}–{c['high']:.{d}f}), n={c['n']}"
                    )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "method": self.method,
                "alpha": self.alpha,
                "classes": list(self.classes),
                "threshold": self.threshold,
                "n_calibration": self.n_calibration,
                "mean_size": self.mean_size,
                "singleton_rate": self.singleton_rate,
                "empty_rate": self.empty_rate,
                "coverage": self.coverage,
                "by_group": self.by_group,
                "by_size": self.by_size,
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Quantity | Value |", "| --- | ---: |"]
        rows += [
            f"| Target coverage | {1 - self.alpha:.{d}g} |",
            f"| Mean set size | {self.mean_size:.{d}f} |",
            f"| Singleton sets | {self.singleton_rate:.1%} |",
            f"| Empty sets | {self.empty_rate:.1%} |",
        ]
        if self.coverage:
            c = self.coverage
            rows.append(f"| Empirical coverage | {c['coverage']:.{d}f} ({c['low']:.{d}f}–{c['high']:.{d}f}) |")
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def _probs(p: Any, what: str) -> np.ndarray:
    a = numeric(p, what)
    if a.ndim != 2 or a.shape[1] < 2:
        raise InputValidationError(f"{what} must be an n × classes probability matrix.")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1 + 1e-9):
        raise InputValidationError(f"{what} must hold probabilities in [0, 1].")
    return a


def _label_index(y: Any, classes: Optional[Any], k: int, what: str) -> tuple[np.ndarray, tuple[Any, ...]]:
    arr = np.asarray(y)
    if classes is not None:
        order = list(np.asarray(classes).tolist())
    elif np.issubdtype(arr.dtype, np.integer):
        order = list(range(k))
    else:
        order = sorted(set(arr.tolist()))
    if len(order) != k:
        raise InputValidationError(f"{len(order)} classes for {k} probability columns; pass classes=.")
    pos = {c: i for i, c in enumerate(order)}
    try:
        idx = np.array([pos[v] for v in arr.tolist()], dtype=np.int64)
    except KeyError as exc:
        raise InputValidationError(
            f"{what} has a label that is not a probability column: {exc.args[0]!r}."
        ) from None
    return idx, tuple(order)


def conformal_classification(
    y_cal: Any,
    p_cal: Any,
    p_test: Any,
    *,
    alpha: float = 0.1,
    method: str = "aps",
    y_test: Any = None,
    classes: Any = None,
    by: Any = None,
    by_test: Any = None,
    lam: float = 0.01,
    k_reg: int = 1,
    level: float = 0.95,
) -> ConformalSets:
    """Split-conformal prediction sets from class probabilities (see the module docstring).

    ``y_cal`` / ``p_cal`` are held-out calibration labels and probabilities (n × k); ``p_test`` the probabilities
    to build sets for. ``by`` / ``by_test`` (or ``by="class"``) give group-conditional thresholds. With ``y_test``
    the result also reports coverage overall, per group and per set size."""
    if method not in ("lac", "aps", "raps"):
        raise InputValidationError("method must be 'lac', 'aps' or 'raps'.")
    pc, pt = _probs(p_cal, "p_cal"), _probs(p_test, "p_test")
    if pc.shape[1] != pt.shape[1]:
        raise InputValidationError("p_cal and p_test must have the same number of classes.")
    k = pc.shape[1]
    yc, order = _label_index(y_cal, classes, k, "y_cal")
    if yc.size != pc.shape[0]:
        raise InputValidationError(f"y_cal has {yc.size} labels for {pc.shape[0]} probability rows.")
    cal_scores = _class_scores(pc, method, lam, k_reg)[np.arange(yc.size), yc]
    test_scores = _class_scores(pt, method, lam, k_reg)
    class_conditional = isinstance(by, str) and by == "class"
    if class_conditional:
        thresholds = {}
        sets = np.zeros_like(test_scores, dtype=bool)
        for j in range(k):
            sc = cal_scores[yc == j]
            q = conformal_quantile(sc, alpha) if sc.size else math.inf
            thresholds[order[j]] = q
            sets[:, j] = test_scores[:, j] <= q
        threshold: Any = thresholds
    elif by is not None:
        gc = _groups(by, yc.size, "by")
        if by_test is None:
            raise InputValidationError("Group-conditional conformal prediction needs by_test for the test rows.")
        gt = _groups(by_test, pt.shape[0], "by_test")
        thresholds = {}
        sets = np.zeros_like(test_scores, dtype=bool)
        for g in np.unique(np.concatenate([gc, gt])):
            sc = cal_scores[gc == g]
            q = conformal_quantile(sc, alpha) if sc.size else math.inf
            thresholds[g.item() if hasattr(g, "item") else g] = q
            rows = gt == g
            sets[rows] = test_scores[rows] <= q
        threshold = thresholds
    else:
        threshold = conformal_quantile(cal_scores, alpha)
        sets = test_scores <= threshold
    if method in ("aps", "raps"):
        # the top class is always included (as in the deterministic APS of Romano et al.), so no set is empty
        top = np.argmax(pt, axis=1)
        sets[np.arange(pt.shape[0]), top] = True
    coverage = by_group = by_size = None
    if y_test is not None:
        yt, _ = _label_index(y_test, list(order), k, "y_test")
        if yt.size != pt.shape[0]:
            raise InputValidationError(f"y_test has {yt.size} labels for {pt.shape[0]} probability rows.")
        hit = sets[np.arange(yt.size), yt]
        coverage = _coverage_block(hit, level)
        groups = (
            np.array([order[i] for i in yt], dtype=object)
            if class_conditional or by is None
            else _groups(by_test, yt.size, "by_test")
        )
        by_group = {
            str(g): _coverage_block(hit[groups == g], level) for g in sorted(set(groups.tolist()), key=str)
        }
        sizes = sets.sum(axis=1)
        by_size = {int(s): _coverage_block(hit[sizes == s], level) for s in np.unique(sizes)}
    return ConformalSets(
        sets=sets,
        classes=order,
        threshold=threshold,
        alpha=alpha,
        method=method,
        n_calibration=int(yc.size),
        coverage=coverage,
        by_group=by_group,
        by_size=by_size,
    )


# -------------------------------------------------------------------------------------------------- regression


@dataclass(frozen=True)
class ConformalIntervals:
    """Conformal prediction intervals for regression."""

    lower: np.ndarray = field(repr=False)
    upper: np.ndarray = field(repr=False)
    threshold: Any
    alpha: float
    method: str
    n_calibration: int
    coverage: Optional[dict[str, Any]] = None
    by_group: Optional[dict[str, dict[str, Any]]] = None

    @property
    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def widths(self) -> np.ndarray:
        return np.asarray(self.upper - self.lower)

    @property
    def mean_width(self) -> float:
        return float(np.mean(self.widths))

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        lines = [
            f"Conformal prediction intervals ({self.method}, target coverage {1 - self.alpha:.{d}g}, "
            f"{self.n_calibration} calibration examples)",
            f"  Mean width {self.mean_width:.{d}f}; median width {float(np.median(self.widths)):.{d}f}",
        ]
        if self.coverage:
            c = self.coverage
            lines.append(
                f"  Empirical coverage {c['coverage']:.{d}f} ({c['low']:.{d}f}–{c['high']:.{d}f}), n={c['n']}"
            )
        if self.by_group:
            lines.append("  Coverage by group:")
            for k, c in self.by_group.items():
                lines.append(f"    {k:<12} {_cov_text(c, d)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "method": self.method,
                "alpha": self.alpha,
                "threshold": self.threshold,
                "n_calibration": self.n_calibration,
                "mean_width": self.mean_width,
                "coverage": self.coverage,
                "by_group": self.by_group,
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Quantity | Value |", "| --- | ---: |"]
        rows += [f"| Target coverage | {1 - self.alpha:.{d}g} |", f"| Mean width | {self.mean_width:.{d}f} |"]
        if self.coverage:
            c = self.coverage
            rows.append(f"| Empirical coverage | {c['coverage']:.{d}f} ({c['low']:.{d}f}–{c['high']:.{d}f}) |")
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def _vec(x: Any, what: str) -> np.ndarray:
    a = numeric(x, what).ravel()
    if not np.all(np.isfinite(a)):
        raise InputValidationError(f"{what} must be finite.")
    return a


def conformal_regression(
    y_cal: Any,
    pred_cal: Any,
    pred_test: Any,
    *,
    alpha: float = 0.1,
    method: str = "absolute",
    scale_cal: Any = None,
    scale_test: Any = None,
    y_test: Any = None,
    by: Any = None,
    by_test: Any = None,
    level: float = 0.95,
) -> ConformalIntervals:
    """Split-conformal prediction intervals (see the module docstring).

    ``"absolute"``: ``pred_*`` are point predictions; interval pred ± q̂. ``"normalized"``: also give
    ``scale_cal`` / ``scale_test`` (positive per-example scales such as predicted SDs); interval pred ± q̂·scale.
    ``"cqr"``: ``pred_*`` are n × 2 arrays of lower and upper quantile predictions (e.g. the α/2 and 1 − α/2
    quantiles); interval [lo − q̂, hi + q̂]. ``by`` / ``by_test`` give a threshold per group."""
    if method not in ("absolute", "normalized", "cqr"):
        raise InputValidationError("method must be 'absolute', 'normalized' or 'cqr'.")
    yc = _vec(y_cal, "y_cal")
    if method == "cqr":
        pc, pt = numeric(pred_cal, "pred_cal"), numeric(pred_test, "pred_test")
        if pc.ndim != 2 or pc.shape[1] != 2 or pt.ndim != 2 or pt.shape[1] != 2:
            raise InputValidationError("For cqr, pred_cal and pred_test must be n × 2 (lower, upper) quantiles.")
        if not (np.all(np.isfinite(pc)) and np.all(np.isfinite(pt))):
            raise InputValidationError("Quantile predictions must be finite.")
        if pc.shape[0] != yc.size:
            raise InputValidationError(f"pred_cal has {pc.shape[0]} rows for {yc.size} labels.")
        cal_scores = np.maximum(pc[:, 0] - yc, yc - pc[:, 1])
        n_test = pt.shape[0]
    else:
        pc, pt = _vec(pred_cal, "pred_cal"), _vec(pred_test, "pred_test")
        if pc.size != yc.size:
            raise InputValidationError(f"pred_cal has {pc.size} values for {yc.size} labels.")
        cal_scores = np.abs(yc - pc)
        n_test = pt.size
        if method == "normalized":
            if scale_cal is None or scale_test is None:
                raise InputValidationError("method='normalized' needs scale_cal and scale_test.")
            sc, st = _vec(scale_cal, "scale_cal"), _vec(scale_test, "scale_test")
            if sc.size != yc.size or st.size != n_test or np.any(sc <= 0) or np.any(st <= 0):
                raise InputValidationError("scale_cal / scale_test must be positive, one per example.")
            cal_scores = cal_scores / sc

    def bounds(q: float, rows: Any = slice(None)) -> tuple[np.ndarray, np.ndarray]:
        if method == "cqr":
            return pt[rows, 0] - q, pt[rows, 1] + q
        half = q * (_vec(scale_test, "scale_test")[rows] if method == "normalized" else 1.0)
        return pt[rows] - half, pt[rows] + half

    lower = np.empty(n_test)
    upper = np.empty(n_test)
    if by is not None:
        gc = _groups(by, yc.size, "by")
        if by_test is None:
            raise InputValidationError("Group-conditional conformal prediction needs by_test for the test rows.")
        gt = _groups(by_test, n_test, "by_test")
        threshold: Any = {}
        for g in np.unique(np.concatenate([gc, gt])):
            sc_g = cal_scores[gc == g]
            q = conformal_quantile(sc_g, alpha) if sc_g.size else math.inf
            threshold[g.item() if hasattr(g, "item") else g] = q
            rows = gt == g
            lower[rows], upper[rows] = bounds(q, rows)
    else:
        threshold = conformal_quantile(cal_scores, alpha)
        lower, upper = bounds(threshold)
    coverage = by_group = None
    if y_test is not None:
        yt = _vec(y_test, "y_test")
        if yt.size != n_test:
            raise InputValidationError(f"y_test has {yt.size} values for {n_test} predictions.")
        hit = (yt >= lower) & (yt <= upper)
        coverage = _coverage_block(hit, level)
        if by_test is not None:
            gt2 = _groups(by_test, n_test, "by_test")
            by_group = {str(g): _coverage_block(hit[gt2 == g], level) for g in sorted(set(gt2.tolist()), key=str)}
    return ConformalIntervals(
        lower=np.asarray(lower),
        upper=np.asarray(upper),
        threshold=threshold,
        alpha=alpha,
        method=method,
        n_calibration=int(yc.size),
        coverage=coverage,
        by_group=by_group,
    )
