"""Recalibration: fit a map from a model's probabilities to calibrated ones on held-out data, then apply it.

>>> import evalsuite as es
>>> cal = es.recalibrate(y_val, p_val, method="isotonic")   # "platt", "temperature", "beta", "histogram"
>>> p_test_cal = cal(p_test)                               # or cal.transform(p_test)
>>> es.calibration_report(y_test, p_test_cal, n_resamples=1000)  # check it on separate data, with CIs

Methods: Platt scaling (Platt 1999, with Platt's smoothed targets, on the logit of the probabilities),
isotonic regression (Zadrozny & Elkan 2002; pool-adjacent-violators, clipped outside the fitted range),
temperature scaling (Guo et al. 2017; one parameter, also for multiclass probability matrices), beta calibration
(Kull, Silva Filho & Flach 2017; three parameters) and histogram binning (equal-mass bins). Fit on data the model
was not trained on, and evaluate on different data again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core.exceptions import InputValidationError, StatisticalTestError

__all__ = ["Calibrator", "recalibrate"]

_EPS = 1e-12
_METHODS = ("platt", "isotonic", "temperature", "beta", "histogram")


def _clip(p: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(p, _EPS, 1 - _EPS), dtype=np.float64)


def _logit(p: np.ndarray) -> np.ndarray:
    q = _clip(p)
    return np.asarray(np.log(q / (1 - q)), dtype=np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -700, 700))), dtype=np.float64)


def _newton_logistic(x: np.ndarray, t: np.ndarray, w: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Weighted logistic regression of (possibly soft) targets ``t`` on design ``x``: Newton–Raphson with
    step halving (the log-likelihood never decreases) and a tiny ridge for stability; returns coefficients."""

    def loglik(b: np.ndarray) -> float:
        eta = x @ b
        return float(np.sum(w * (t * eta - np.logaddexp(0.0, eta))))

    beta = np.zeros(x.shape[1])
    ridge = 1e-10 * np.eye(x.shape[1])
    cur = loglik(beta)
    for _ in range(max_iter):
        mu = _sigmoid(x @ beta)
        grad = x.T @ (w * (t - mu))
        hess = (x * (w * mu * (1 - mu))[:, None]).T @ x + ridge
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            raise StatisticalTestError(
                "The recalibration model could not be fitted (singular information matrix)."
            ) from None
        scale = 1.0
        while scale > 1e-8:
            cand = beta + scale * step
            new = loglik(cand)
            if new >= cur - 1e-12:
                break
            scale /= 2
        beta, cur = cand, new
        if np.max(np.abs(scale * step)) < 1e-10:
            return beta
    raise StatisticalTestError("The recalibration model did not converge (the outcome may be separable).")


def _pava(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Increasing isotonic fit: returns (thresholds x, fitted values), ties in x pooled first."""
    order = np.argsort(x, kind="mergesort")
    xs, ys, ws = x[order], y[order], w[order]
    ux, start = np.unique(xs, return_index=True)
    wsum = np.add.reduceat(ws, start)
    ymean = np.add.reduceat(ws * ys, start) / wsum
    # pool adjacent violators on the unique-x means
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for v, wt in zip(ymean.tolist(), wsum.tolist()):
        vals.append(v)
        wts.append(wt)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, l2 = vals.pop(), wts.pop(), lens.pop()
            tot = wts[-1] + w2
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / tot
            wts[-1] = tot
            lens[-1] += l2
    fitted = np.repeat(np.asarray(vals), lens)
    return ux, fitted


@dataclass(frozen=True)
class Calibrator:
    """A fitted recalibration map. Call it (or ``transform``) on new probabilities."""

    method: str
    params: dict[str, Any]
    n_fit: int
    _state: dict[str, Any] = field(repr=False, default_factory=dict)

    def __call__(self, y_prob: Any) -> np.ndarray:
        return self.transform(y_prob)

    def transform(self, y_prob: Any) -> np.ndarray:
        p = np.asarray(y_prob, dtype=np.float64)
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise InputValidationError("Probabilities must be finite and within [0, 1].")
        m = self.method
        if m == "temperature" and p.ndim == 2:
            if p.shape[1] != self._state["n_classes"]:
                raise InputValidationError(f"Expected {self._state['n_classes']} probability columns.")
            z = np.log(_clip(p)) / self.params["temperature"]
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            return np.asarray(e / e.sum(axis=1, keepdims=True))
        if p.ndim != 1:
            raise InputValidationError("This calibrator maps one probability per example (a 1-D array).")
        if m == "platt":
            return _sigmoid(self.params["a"] * _logit(p) + self.params["b"])
        if m == "temperature":
            return _sigmoid(_logit(p) / self.params["temperature"])
        if m == "beta":
            q = _clip(p)
            return _sigmoid(self.params["a"] * np.log(q) - self.params["b"] * np.log(1 - q) + self.params["c"])
        if m == "isotonic":
            xs, fy = self._state["x"], self._state["y"]
            if xs.size == 1:
                return np.full(p.shape, float(fy[0]))
            return np.asarray(np.interp(p, xs, fy))  # constant beyond the fitted range
        # histogram
        edges, rates = self._state["edges"], self._state["rates"]
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, rates.size - 1)
        return np.asarray(rates[idx])

    def summary(self) -> str:
        ps = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"Calibrator({self.method}; {ps}; fitted on {self.n_fit} examples)"

    def __repr__(self) -> str:
        return self.summary()


def _binary_inputs(y_true: Any, y_prob: Any, sample_weight: Any, pos_label: Any) -> tuple[np.ndarray, ...]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.size != p.size:
        raise InputValidationError(f"y_true and y_prob must be 1-D of equal length; got {y.shape} and {p.shape}.")
    if y.size < 2:
        raise InputValidationError("At least 2 examples are needed to recalibrate.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise InputValidationError("y_prob must be finite probabilities in [0, 1].")
    labels = np.unique(y)
    if labels.size > 2:
        raise InputValidationError(
            "Binary methods need two classes; use method='temperature' with a probability matrix."
        )
    pos = pos_label if pos_label is not None else (labels.max() if labels.size else 1)
    t = (y == pos).astype(np.float64)
    if t.min() == t.max():
        raise InputValidationError("Recalibration needs both outcomes in y_true.")
    w = np.ones(y.size) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if w.shape != t.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise InputValidationError("sample_weight must be finite, non-negative and one per example.")
    return t, p, w


def recalibrate(
    y_true: Any,
    y_prob: Any,
    *,
    method: str = "isotonic",
    n_bins: int = 10,
    pos_label: Any = None,
    sample_weight: Any = None,
) -> Calibrator:
    """Fit a recalibration map on held-out labels and probabilities (see the module docstring for methods).

    For ``method="temperature"`` ``y_prob`` may be an n × k probability matrix with class labels ``y_true``
    (classes sorted, as columns)."""
    from scipy import optimize

    if method not in _METHODS:
        raise InputValidationError(f"method must be one of {', '.join(_METHODS)}; got {method!r}.")
    p_all = np.asarray(y_prob, dtype=np.float64)
    if method == "temperature" and p_all.ndim == 2:
        y = np.asarray(y_true)
        if p_all.shape[0] != y.size or not np.all(np.isfinite(p_all)):
            raise InputValidationError("y_prob must be a finite n × k matrix with one row per label.")
        classes = np.unique(y)
        k = p_all.shape[1]
        if classes.size > k:
            raise InputValidationError(f"{classes.size} classes in y_true but {k} probability columns.")
        cols = np.searchsorted(np.arange(k) if np.issubdtype(y.dtype, np.integer) else classes, y)
        logp = np.log(_clip(p_all))

        def nll_k(log_t: float) -> float:
            z = logp / math.exp(log_t)
            z -= z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1))
            return float(np.mean(lse - z[np.arange(y.size), cols]))

        res = optimize.minimize_scalar(nll_k, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10})
        temp = math.exp(float(res.x))
        return Calibrator("temperature", {"temperature": temp}, int(y.size), {"n_classes": k})
    t, p, w = _binary_inputs(y_true, y_prob, sample_weight, pos_label)
    n = t.size
    if method == "platt":
        n_pos, n_neg = float(w[t == 1].sum()), float(w[t == 0].sum())
        hi, lo = (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)
        target = np.where(t == 1, hi, lo)
        beta = _newton_logistic(np.column_stack([_logit(p), np.ones(n)]), target, w)
        return Calibrator("platt", {"a": float(beta[0]), "b": float(beta[1])}, n)
    if method == "temperature":
        z = _logit(p)

        def nll(log_t: float) -> float:
            q = _clip(_sigmoid(z / math.exp(log_t)))
            return float(-np.sum(w * (t * np.log(q) + (1 - t) * np.log(1 - q))))

        res = optimize.minimize_scalar(nll, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10})
        return Calibrator("temperature", {"temperature": math.exp(float(res.x))}, n, {"n_classes": 2})
    if method == "beta":
        q = _clip(p)
        beta = _newton_logistic(np.column_stack([np.log(q), -np.log(1 - q), np.ones(n)]), t, w)
        a, b, c = (float(v) for v in beta)
        if a < 0 or b < 0:  # Kull et al.: refit without the offending term to keep the map monotone
            if a < 0:
                beta2 = _newton_logistic(np.column_stack([-np.log(1 - q), np.ones(n)]), t, w)
                a, b, c = 0.0, float(beta2[0]), float(beta2[1])
            else:
                beta2 = _newton_logistic(np.column_stack([np.log(q), np.ones(n)]), t, w)
                a, b, c = float(beta2[0]), 0.0, float(beta2[1])
        return Calibrator("beta", {"a": a, "b": b, "c": c}, n)
    if method == "isotonic":
        xs, fy = _pava(p, t, w)
        return Calibrator("isotonic", {"steps": int(np.unique(fy).size)}, n, {"x": xs, "y": fy})
    # histogram binning, equal-mass bins
    if n_bins < 1:
        raise InputValidationError("n_bins must be at least 1.")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = 0.0, 1.0 + 1e-12
    if edges.size < 2:
        edges = np.array([0.0, 1.0 + 1e-12])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    wsum = np.bincount(idx, weights=w, minlength=edges.size - 1)
    rates = np.divide(
        np.bincount(idx, weights=w * t, minlength=edges.size - 1),
        wsum,
        out=np.full(edges.size - 1, np.nan),
        where=wsum > 0,
    )
    overall = float(np.sum(w * t) / np.sum(w))
    rates = np.where(np.isnan(rates), overall, rates)
    return Calibrator("histogram", {"bins": int(rates.size)}, n, {"edges": edges, "rates": rates})
