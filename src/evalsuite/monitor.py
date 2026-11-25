"""Monitoring mode: metrics over a sliding window of live traffic, drift against a reference, and alerts.

>>> import evalsuite as es
>>> mon = es.Monitor(window=500, alerts={"accuracy": ("<", 0.8)})
>>> mon.update([1, 0, 1, 1], y_pred=[1, 0, 0, 1])
[Alert(accuracy=0.75 < 0.8, window n=4)]

Feed batches as labels arrive (``update``); every call returns the alerts that fire on the current window.
``snapshot()`` evaluates the window with :func:`evalsuite.evaluate` (intervals included), so a dip can be told
apart from noise. With ``reference_scores`` (e.g. validation-set probabilities), the window's scores are compared
with the reference distribution for drift: the population stability index (PSI) and the two-sample
Kolmogorov–Smirnov test, which also work before any label arrives (``observe``).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .core.exceptions import InputValidationError

_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def population_stability_index(reference: Any, current: Any, *, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI between two samples of scores, with bins at the reference's quantiles:
    Σ (c − r) · ln(c / r) over bins, shares floored at ``eps``. Rule of thumb: < 0.1 stable, 0.1–0.25 moderate
    shift, > 0.25 major shift."""
    r = np.asarray(reference, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()
    if r.size < 2 or c.size < 1 or not (np.all(np.isfinite(r)) and np.all(np.isfinite(c))):
        raise InputValidationError("PSI needs finite reference (>= 2) and current (>= 1) values.")
    if bins < 2:
        raise InputValidationError("bins must be at least 2.")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        edges = np.array([r[0] - 0.5, r[0] + 0.5])
    inner = edges[1:-1]
    rp = np.bincount(np.searchsorted(inner, r, side="right"), minlength=edges.size - 1) / r.size
    cp = np.bincount(np.searchsorted(inner, c, side="right"), minlength=edges.size - 1) / c.size
    rp, cp = np.maximum(rp, eps), np.maximum(cp, eps)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def ks_drift(reference: Any, current: Any) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov statistic and p-value (SciPy's exact / asymptotic choice)."""
    from scipy import stats

    r = np.asarray(reference, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()
    if r.size < 1 or c.size < 1:
        raise InputValidationError("ks_drift needs non-empty samples.")
    res = stats.ks_2samp(r, c)
    return float(res.statistic), float(res.pvalue)


@dataclass(frozen=True)
class Alert:
    """A threshold crossed by a monitored quantity."""

    name: str
    value: float
    op: str
    threshold: float
    n: int

    def __repr__(self) -> str:
        return f"Alert({self.name}={self.value:.4g} {self.op} {self.threshold:g}, window n={self.n})"


class Monitor:
    """Sliding-window metrics, drift and alerts for a deployed classifier or regressor.

    ``metrics``: names for :func:`evalsuite.evaluate` (default: its standard set). ``window``: number of most
    recent labelled observations kept. ``alerts``: ``{name: (op, threshold)}`` with op one of ``<``, ``<=``,
    ``>``, ``>=``; names are metrics, ``"psi"``, ``"ks"`` (statistic) or ``"ks_p"``. ``min_n``: no metric alert
    before this many labelled observations. ``on_alert``: called with each alert (e.g. to page someone).
    """

    def __init__(
        self,
        *,
        metrics: Optional[Sequence[str]] = None,
        window: int = 1000,
        alerts: Optional[Mapping[str, tuple[str, float]]] = None,
        reference_scores: Any = None,
        drift_window: Optional[int] = None,
        min_n: int = 1,
        ci: Any = "analytic",
        on_alert: Optional[Callable[[Alert], Any]] = None,
    ) -> None:
        if window < 1:
            raise InputValidationError("window must be a positive integer.")
        for name, rule in (alerts or {}).items():
            if not (isinstance(rule, tuple) and len(rule) == 2 and rule[0] in _OPS):
                raise InputValidationError(f"alerts[{name!r}] must be (op, threshold) with op in {sorted(_OPS)}.")
        self.metrics = list(metrics) if metrics is not None else None
        self.window = window
        self.alerts = dict(alerts or {})
        self.min_n = max(1, int(min_n))
        self.ci = ci
        self.on_alert = on_alert
        self.reference = None if reference_scores is None else np.asarray(reference_scores, dtype=float).ravel()
        self._true: deque[Any] = deque(maxlen=window)
        self._pred: deque[Any] = deque(maxlen=window)
        self._prob: deque[Any] = deque(maxlen=window)
        self._scores: deque[float] = deque(maxlen=drift_window or window)
        self.n_seen = 0
        self.history: list[dict[str, Any]] = []
        self._kinds: Optional[tuple[bool, bool]] = None

    # -------------------------------------------------------------------------------------------------- data

    def observe(self, scores: Any) -> list[Alert]:
        """Record model scores without labels (drift only); returns the drift alerts that fire."""
        s = np.asarray(scores, dtype=float).ravel()
        self._scores.extend(s.tolist())
        return self._fire(self._drift_values())

    def update(self, y_true: Any, *, y_pred: Any = None, y_prob: Any = None) -> list[Alert]:
        """Add a batch of labelled observations; returns the alerts that fire on the current window."""
        yt = list(np.asarray(y_true).tolist())
        if y_pred is None and y_prob is None:
            raise InputValidationError("update() needs y_pred, y_prob, or both.")
        for name, arr in (("y_pred", y_pred), ("y_prob", y_prob)):
            if arr is not None and len(arr) != len(yt):
                raise InputValidationError(f"{name} must have one value per observation in y_true.")
        kinds = (y_pred is not None, y_prob is not None)
        if self._kinds is None:
            self._kinds = kinds
        elif kinds != self._kinds:
            raise InputValidationError(
                "Every update() must pass the same inputs (y_pred and/or y_prob) as the first."
            )
        self._true.extend(yt)
        if y_pred is not None:
            self._pred.extend(np.asarray(y_pred).tolist())
        if y_prob is not None:
            p = np.asarray(y_prob, dtype=float)
            self._prob.extend(p.tolist())
            self._scores.extend((p if p.ndim == 1 else p.max(axis=1)).tolist())
        self.n_seen += len(yt)
        values = {} if len(self._true) < self.min_n else self._metric_values()
        values.update(self._drift_values())
        self.history.append({"n_seen": self.n_seen, **values})
        return self._fire(values)

    # -------------------------------------------------------------------------------------------- results

    def snapshot(self) -> Any:
        """:func:`evalsuite.evaluate` on the current window (intervals included)."""
        from .api import evaluate

        if not self._true:
            raise InputValidationError("No labelled observations yet.")
        pred = list(self._pred) if len(self._pred) == len(self._true) else None
        prob = list(self._prob) if len(self._prob) == len(self._true) else None
        return evaluate(list(self._true), pred, y_prob=prob, metrics=self.metrics, ci=self.ci)

    def drift(self) -> dict[str, float]:
        """PSI and KS of the recent scores against the reference (empty without a reference)."""
        return self._drift_values()

    def _metric_values(self) -> dict[str, float]:
        try:
            r = self.snapshot()
        except Exception:  # a window with one class, etc.: no metric this round
            return {}
        return {k: float(m.value) for k, m in r.metrics.items() if not isinstance(m.value, np.ndarray)}

    def _drift_values(self) -> dict[str, float]:
        if self.reference is None or len(self._scores) < 2:
            return {}
        cur = np.fromiter(self._scores, dtype=float)
        ks, p = ks_drift(self.reference, cur)
        return {"psi": population_stability_index(self.reference, cur), "ks": ks, "ks_p": p}

    def _fire(self, values: Mapping[str, float]) -> list[Alert]:
        fired = []
        for name, (op, thr) in self.alerts.items():
            v = values.get(name)
            if v is not None and not math.isnan(v) and _OPS[op](v, thr):
                a = Alert(name, float(v), op, float(thr), len(self._true) or len(self._scores))
                fired.append(a)
                if self.on_alert is not None:
                    self.on_alert(a)
        return fired
