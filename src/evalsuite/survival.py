"""Time-to-event (survival) evaluation with right-censored data.

* :func:`concordance_index` — Harrell's C (``ipcw=False``) or Uno's IPCW C (``ipcw=True``), with a bootstrap
  interval on request.
* :func:`time_dependent_auc` — cumulative/dynamic ROC AUC at each time point (Uno et al. 2007; Hung & Chiang
  2010), and its mean over time.
* :func:`brier_score_survival` / :func:`integrated_brier_score` — Graf et al. (1999) IPCW Brier score at each
  time point and integrated over a time range.
* :func:`kaplan_meier` — the Kaplan–Meier estimate (and of the censoring distribution, ``reverse=True``).

Conventions follow scikit-survival, and the results agree with it (tests in ``tests/test_survival.py``):
``event`` is True / 1 for an observed event and False / 0 for censoring; ``risk`` is a risk score (higher means
earlier events); survival-probability ``estimate`` columns correspond to ``times``. Inverse-probability-of-
censoring weights come from the Kaplan–Meier estimate of the censoring distribution fitted on
``train_time`` / ``train_event`` (default: the evaluation data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError
from .core.result import MetricResult

__all__ = [
    "SurvivalCurve",
    "brier_score_survival",
    "concordance_index",
    "integrated_brier_score",
    "kaplan_meier",
    "time_dependent_auc",
]


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal rule (np.trapezoid on NumPy 2, written out for NumPy 1.x)."""
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


def _survival_inputs(time: Any, event: Any) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float).ravel()
    e = np.asarray(event).ravel()
    if t.size == 0 or t.shape != e.shape:
        raise InputValidationError("time and event must be non-empty and the same length.")
    if not np.all(np.isfinite(t)):
        raise InputValidationError("time contains NaN or infinite values.")
    if e.dtype != bool:
        if not np.all(np.isin(e, (0, 1))):
            raise InputValidationError("event must be boolean or 0 / 1 (1 = event observed, 0 = censored).")
        e = e.astype(bool)
    return t, e


@dataclass(frozen=True)
class SurvivalCurve:
    """A step function: ``prob[k]`` holds from ``time[k]`` until the next time point."""

    time: np.ndarray
    prob: np.ndarray

    def __call__(self, t: Any) -> np.ndarray:
        idx = np.searchsorted(self.time, np.asarray(t, dtype=float), side="right") - 1
        out = np.where(idx >= 0, self.prob[np.clip(idx, 0, None)], 1.0)
        return np.asarray(out, dtype=float)


def kaplan_meier(time: Any, event: Any, *, reverse: bool = False) -> SurvivalCurve:
    """Kaplan–Meier survival curve; ``reverse=True`` estimates the censoring distribution G(t) (at tied times,
    events leave the risk set before censorings, as in scikit-survival)."""
    t, e = _survival_inputs(time, event)
    uniq, inv = np.unique(t, return_inverse=True)
    n_events = np.bincount(inv, weights=e.astype(float), minlength=uniq.size)
    n_total = np.bincount(inv, minlength=uniq.size).astype(float)
    at_risk = n_total[::-1].cumsum()[::-1]
    if reverse:
        if e.all():
            return SurvivalCurve(uniq, np.ones(uniq.size))
        d = n_total - n_events  # censorings
        r = at_risk - n_events
    else:
        d, r = n_events, at_risk
    ratio = np.divide(d, r, out=np.zeros_like(d), where=d != 0)
    return SurvivalCurve(uniq, np.cumprod(1.0 - ratio))


def _censoring(time: np.ndarray, event: np.ndarray, train_time: Any, train_event: Any) -> SurvivalCurve:
    if (train_time is None) != (train_event is None):
        raise InputValidationError("Give both train_time and train_event, or neither.")
    if train_time is None:
        return kaplan_meier(time, event, reverse=True)
    return kaplan_meier(train_time, train_event, reverse=True)


class _Fenwick:
    def __init__(self, n: int) -> None:
        self.n = n
        self.tree = np.zeros(n + 1)

    def add(self, i: int, v: float) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += v
            i += i & -i

    def prefix(self, i: int) -> float:  # sum of positions [0, i)
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return float(s)


def _concordance(t: np.ndarray, e: np.ndarray, r: np.ndarray, w: np.ndarray, tol: float) -> tuple[float, int]:
    """Σ_i w_i (concordant_i + ties_i / 2) / Σ_i w_i comparable_i over events i; comparable j: t_j > t_i, or
    t_j == t_i with j censored. O(n log n)."""
    order_r = np.sort(r)
    fen = _Fenwick(r.size)
    # Each subject sits at the first sorted position of its risk value, so a prefix sum up to position p counts
    # the inserted subjects whose risk is below order_r[p].
    slot = np.searchsorted(order_r, r, side="left")
    num = den = 0.0
    pairs = 0
    order_t = np.argsort(-t, kind="stable")
    i = 0
    n = t.size
    inserted = 0
    while i < n:
        j = i
        ti = t[order_t[i]]
        while j < n and t[order_t[j]] == ti:
            j += 1
        group = order_t[i:j]
        cens = group[~e[group]]
        evts = group[e[group]]
        # censored subjects at this time are comparable to events at the same time: insert them first
        for k in cens:
            fen.add(int(slot[k]), 1.0)
            inserted += 1
        for k in evts:
            lo = int(np.searchsorted(order_r, r[k] - tol, side="left"))
            hi = int(np.searchsorted(order_r, r[k] + tol, side="right"))
            less = fen.prefix(lo)
            ties = fen.prefix(hi) - less
            if inserted:
                num += w[k] * (less + 0.5 * ties)
                den += w[k] * inserted
                pairs += inserted
        for k in evts:
            fen.add(int(slot[k]), 1.0)
            inserted += 1
        i = j
    if den == 0:
        raise InputValidationError(
            "No comparable pairs: the concordance index needs at least one observed event "
            "before another subject's time."
        )
    return num / den, pairs


def concordance_index(
    time: Any,
    event: Any,
    risk: Any,
    *,
    ipcw: bool = False,
    tau: Optional[float] = None,
    train_time: Any = None,
    train_event: Any = None,
    tied_tol: float = 1e-8,
    ci: Optional[str] = None,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: Optional[int] = 0,
) -> MetricResult:
    """Concordance index for risk scores: the share of comparable pairs whose higher-risk member fails first
    (ties in risk count 1/2). ``ipcw=False``: Harrell's C. ``ipcw=True``: Uno's C, pairs weighted by 1 / G(t_i)²
    (censoring KM from ``train_*``), optionally truncated at ``tau``. ``ci="bootstrap"`` adds a percentile
    interval over resampled subjects."""
    t, e = _survival_inputs(time, event)
    r = np.asarray(risk, dtype=float).ravel()
    if r.shape != t.shape or not np.all(np.isfinite(r)):
        raise InputValidationError("risk must be finite and have one value per subject.")

    def weights(tt: np.ndarray, ee: np.ndarray) -> np.ndarray:
        if not ipcw:
            return np.ones(tt.size)
        g = _censoring(tt, ee, train_time, train_event)(tt)
        if np.any(g[ee] == 0):
            raise InputValidationError("The censoring survival function is zero at an event time.")
        w = np.zeros(tt.size)
        w[ee] = 1.0 / g[ee] ** 2
        if tau is not None:
            w[tt >= tau] = 0.0
        return w

    value, pairs = _concordance(t, e, r, weights(t, e), tied_tol)
    params: dict[str, Any] = {"method": "uno" if ipcw else "harrell", "comparable_pairs": pairs}
    if tau is not None:
        params["tau"] = float(tau)
    if ci == "bootstrap":
        rng = np.random.default_rng(random_state)
        vals = []
        for _ in range(n_resamples):
            idx = rng.integers(0, t.size, t.size)
            try:
                vals.append(_concordance(t[idx], e[idx], r[idx], weights(t[idx], e[idx]), tied_tol)[0])
            except InputValidationError:
                continue
        a = (1 - level) / 2
        params.update(
            ci_low=float(np.quantile(vals, a)),
            ci_high=float(np.quantile(vals, 1 - a)),
            ci_level=level,
            ci_method="percentile bootstrap",
        )
    elif ci is not None:
        raise InputValidationError("ci must be None or 'bootstrap'.")
    return MetricResult(name="Concordance index", value=float(value), metric="concordance_index", params=params)


def _times(t: np.ndarray, times: Any) -> np.ndarray:
    ts = np.unique(np.atleast_1d(np.asarray(times, dtype=float)))
    if ts.size == 0 or ts.max() >= t.max() or ts.min() < t.min():
        raise InputValidationError(
            f"times must lie within the follow-up of the evaluation data: [{t.min():g}, {t.max():g})."
        )
    return ts


def time_dependent_auc(
    time: Any,
    event: Any,
    risk: Any,
    times: Any,
    *,
    train_time: Any = None,
    train_event: Any = None,
    tied_tol: float = 1e-8,
) -> MetricResult:
    """Cumulative/dynamic AUC at each of ``times``: how well the risk scores separate subjects with an event by
    time t (cases, IPCW-weighted) from those still event-free after t (controls). ``risk`` is one score per subject
    or one column per time point. ``value`` is the mean AUC weighted by the Kaplan–Meier drop over the times;
    ``params["auc"]`` the per-time values."""
    t, e = _survival_inputs(time, event)
    ts = _times(t, times)
    est = np.asarray(risk, dtype=float)
    n, m = t.size, ts.size
    if est.ndim == 1:
        if est.size != n:
            raise InputValidationError("risk must have one value per subject.")
        est = np.broadcast_to(est[:, None], (n, m))
    elif est.shape != (n, m):
        raise InputValidationError(f"risk must be (n_subjects,) or (n_subjects, {m}).")
    g = _censoring(t, e, train_time, train_event)(t)
    ipcw = np.zeros(n)
    if np.any(g[e] == 0):
        raise InputValidationError("The censoring survival function is zero at an event time.")
    ipcw[e] = 1.0 / g[e]
    scores = np.empty(m)
    for k, tk in enumerate(ts):
        o = np.argsort(-est[:, k], kind="quicksort")
        s, tt, ee, w = est[o, k], t[o], e[o], ipcw[o]
        case = (tt <= tk) & ee
        control = tt > tk
        if not case.any() or not control.any():
            raise InputValidationError(f"No cases or no controls at time {tk:g}.")
        tp = np.cumsum(case * w)
        fp = np.cumsum(control).astype(float)
        tp, fp = tp / tp[-1], fp / control.sum()
        tied = np.abs(np.diff(np.r_[np.inf, s])) <= tied_tol
        drop = np.flatnonzero(tied) - 1
        tp, fp = np.delete(tp, drop), np.delete(fp, drop)
        tp, fp = np.r_[0.0, tp], np.r_[0.0, fp]
        scores[k] = float(_trapezoid(tp, fp))
    if m == 1:
        mean = float(scores[0])
    else:
        s_t = kaplan_meier(t, e)(ts)
        d = -np.diff(np.r_[1.0, s_t])
        mean = float((scores * d).sum() / (1.0 - s_t[-1]))
    return MetricResult(
        name="Time-dependent AUC",
        value=mean,
        metric="time_dependent_auc",
        params={"times": ts.tolist(), "auc": scores.tolist(), "method": "cumulative/dynamic, IPCW"},
    )


def _brier(t: np.ndarray, e: np.ndarray, est: np.ndarray, ts: np.ndarray, g: SurvivalCurve) -> np.ndarray:
    g_t = g(ts)
    g_t[g_t == 0] = np.inf
    g_y = g(t)
    g_y[g_y == 0] = np.inf
    out = np.empty(ts.size)
    for k, tk in enumerate(ts):
        s = est[:, k]
        case = (t <= tk) & e
        control = t > tk
        out[k] = float(np.mean(s * s * case / g_y + (1.0 - s) ** 2 * control / g_t[k]))
    return out


def _estimate(est: Any, n: int, m: int) -> np.ndarray:
    a = np.asarray(est, dtype=float)
    if a.ndim == 1 and m == 1:
        a = a.reshape(-1, 1)
    if a.shape != (n, m):
        raise InputValidationError(f"estimate must hold survival probabilities of shape (n_subjects, {m}).")
    if not np.all((a >= 0) & (a <= 1)):
        raise InputValidationError("estimate must hold survival probabilities in [0, 1].")
    return a


def brier_score_survival(
    time: Any, event: Any, estimate: Any, times: Any, *, train_time: Any = None, train_event: Any = None
) -> MetricResult:
    """IPCW Brier score (Graf et al. 1999) of predicted survival probabilities ``estimate[:, k]`` = P(T > times[k])
    at each time point; ``value`` is the mean over times, ``params["brier"]`` the per-time values."""
    t, e = _survival_inputs(time, event)
    ts = _times(t, times)
    est = _estimate(estimate, t.size, ts.size)
    b = _brier(t, e, est, ts, _censoring(t, e, train_time, train_event))
    return MetricResult(
        name="Brier score (survival)",
        value=float(b.mean()),
        metric="brier_score_survival",
        params={"times": ts.tolist(), "brier": b.tolist(), "method": "IPCW, Graf et al. (1999)"},
    )


def integrated_brier_score(
    time: Any, event: Any, estimate: Any, times: Any, *, train_time: Any = None, train_event: Any = None
) -> MetricResult:
    """Integrated Brier score: the IPCW Brier score integrated over ``times`` (trapezoidal rule) and divided by
    the time range. Needs at least two time points. Lower is better; 0.25 is the score of predicting 0.5."""
    t, e = _survival_inputs(time, event)
    ts = _times(t, times)
    if ts.size < 2:
        raise InputValidationError("integrated_brier_score needs at least two time points.")
    est = _estimate(estimate, t.size, ts.size)
    b = _brier(t, e, est, ts, _censoring(t, e, train_time, train_event))
    value = float(_trapezoid(b, ts) / (ts[-1] - ts[0]))
    if not math.isfinite(value):
        raise InputValidationError("The integrated Brier score is undefined for these times (censoring weights).")
    return MetricResult(
        name="Integrated Brier score",
        value=value,
        metric="integrated_brier_score",
        params={"times": ts.tolist(), "brier": b.tolist(), "range": [float(ts[0]), float(ts[-1])]},
    )
