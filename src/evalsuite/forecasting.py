"""Forecast evaluation: scaled errors, quantile and interval scores, CRPS, and the Diebold–Mariano test.

>>> import evalsuite as es
>>> es.mase(y_test, y_hat, y_train=y_train, m=12)          # < 1: better than the seasonal naive forecast
>>> es.interval_score(y_test, lower, upper, level=0.9)     # width plus penalties for misses
>>> es.crps(y_test, ensemble=samples)                      # or crps(y_test, mean=mu, sd=sigma)
>>> es.diebold_mariano(y_test, y_hat_a, y_hat_b, h=1)      # are A's forecasts more accurate than B's?
>>> es.forecast_report(y_test, {"ets": a, "arima": b}, y_train=y_train, m=12)

Definitions: MASE and RMSSE follow Hyndman & Koehler (2006) and the M5 competition, scaled by the in-sample
(seasonal) naive forecast; the interval score and MSIS follow Gneiting & Raftery (2007) and the M4 competition;
CRPS for ensembles uses the energy form (as ``properscoring``) and for normal forecasts the closed form; the
Diebold–Mariano (1995) test uses the Harvey, Leybourne & Newbold (1997) small-sample correction with a t
reference distribution.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError
from .core.result import MetricResult

__all__ = [
    "mase",
    "rmsse",
    "pinball_loss",
    "interval_score",
    "msis",
    "interval_coverage",
    "crps",
    "diebold_mariano",
    "ForecastReport",
    "forecast_report",
]


def _vec(x: Any, what: str) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise InputValidationError(f"{what} must be numeric.") from exc
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise InputValidationError(f"{what} must be non-empty and finite.")
    return a


def _pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    y, p = _vec(y_true, "y_true"), _vec(y_pred, "y_pred")
    if y.size != p.size:
        raise InputValidationError(f"y_true and y_pred must have the same length; got {y.size} and {p.size}.")
    return y, p


def _naive_scale(y_train: Any, m: int, power: int) -> float:
    if m < 1:
        raise InputValidationError("m (seasonal period) must be at least 1.")
    tr = _vec(y_train, "y_train")
    if tr.size <= m:
        raise InputValidationError(f"y_train needs more than m={m} observations; got {tr.size}.")
    d = np.abs(tr[m:] - tr[:-m]) ** power
    s = float(d.mean())
    if s == 0:
        raise InputValidationError(
            "The naive forecast is perfect on y_train (constant series); scaling undefined."
        )
    return s


def mase(y_true: Any, y_pred: Any, *, y_train: Any, m: int = 1) -> MetricResult:
    """Mean absolute scaled error: MAE divided by the in-sample MAE of the seasonal naive forecast with period
    ``m`` (1 for the plain random walk). Below 1 beats the naive forecast."""
    y, p = _pair(y_true, y_pred)
    v = float(np.mean(np.abs(y - p))) / _naive_scale(y_train, m, 1)
    return MetricResult(name="MASE", value=v, metric="mase", params={"m": m, "n": y.size})


def rmsse(y_true: Any, y_pred: Any, *, y_train: Any, m: int = 1) -> MetricResult:
    """Root mean squared scaled error (M5 competition): sqrt(MSE / in-sample MSE of the seasonal naive
    forecast)."""
    y, p = _pair(y_true, y_pred)
    v = math.sqrt(float(np.mean((y - p) ** 2)) / _naive_scale(y_train, m, 2))
    return MetricResult(name="RMSSE", value=v, metric="rmsse", params={"m": m, "n": y.size})


def pinball_loss(y_true: Any, quantile_preds: Any, *, quantiles: Any) -> MetricResult:
    """Mean pinball (quantile) loss. ``quantile_preds`` is n × q (one column per quantile in ``quantiles``) or a
    vector for a single quantile. ``params["per_quantile"]`` holds each quantile's loss (as scikit-learn's
    ``mean_pinball_loss``); the value is their mean."""
    y = _vec(y_true, "y_true")
    qs = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
    if np.any((qs <= 0) | (qs >= 1)):
        raise InputValidationError("quantiles must lie strictly between 0 and 1.")
    q = np.asarray(quantile_preds, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    if q.shape != (y.size, qs.size) or not np.all(np.isfinite(q)):
        raise InputValidationError(
            f"quantile_preds must be finite with shape ({y.size}, {qs.size}); got {q.shape}."
        )
    diff = y[:, None] - q
    loss = np.maximum(qs * diff, (qs - 1) * diff).mean(axis=0)
    per = {float(a): float(b) for a, b in zip(qs, loss)}
    return MetricResult(
        name="Pinball loss", value=float(loss.mean()), metric="pinball_loss", params={"per_quantile": per}
    )


def _interval(y_true: Any, lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, lo, hi = _vec(y_true, "y_true"), _vec(lower, "lower"), _vec(upper, "upper")
    if not (y.size == lo.size == hi.size):
        raise InputValidationError("y_true, lower and upper must have the same length.")
    if np.any(lo > hi):
        raise InputValidationError("Every lower bound must be at most its upper bound.")
    return y, lo, hi


def interval_score(y_true: Any, lower: Any, upper: Any, *, level: float = 0.95) -> MetricResult:
    """Mean interval (Winkler) score of central ``level`` prediction intervals (Gneiting & Raftery 2007):
    width + (2/α)·(lower − y) when y < lower + (2/α)·(y − upper) when y > upper, α = 1 − level. Lower is better."""
    if not 0 < level < 1:
        raise InputValidationError("level must be in (0, 1).")
    y, lo, hi = _interval(y_true, lower, upper)
    a = 1 - level
    s = (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)
    return MetricResult(
        name="Interval score",
        value=float(s.mean()),
        metric="interval_score",
        params={
            "level": level,
            "coverage": float(np.mean((y >= lo) & (y <= hi))),
            "mean_width": float(np.mean(hi - lo)),
        },
    )


def msis(y_true: Any, lower: Any, upper: Any, *, y_train: Any, m: int = 1, level: float = 0.95) -> MetricResult:
    """Mean scaled interval score (M4 competition): the interval score divided by the in-sample MAE of the
    seasonal naive forecast."""
    raw = interval_score(y_true, lower, upper, level=level)
    v = raw.value / _naive_scale(y_train, m, 1)
    return MetricResult(name="MSIS", value=v, metric="msis", params={**raw.params, "m": m})


def interval_coverage(y_true: Any, lower: Any, upper: Any, *, level: Optional[float] = None) -> MetricResult:
    """Share of observations inside their prediction intervals, with a Wilson interval; with the nominal
    ``level`` the binomial test of coverage = level is added (``params["p_value"]``)."""
    from scipy import stats

    from .stats.intervals import proportion_ci

    y, lo, hi = _interval(y_true, lower, upper)
    k, n = int(np.sum((y >= lo) & (y <= hi))), y.size
    ci = proportion_ci(k, n)
    params: dict[str, Any] = {"ci_low": ci.low, "ci_high": ci.high, "ci_method": "wilson", "n": n}
    if level is not None:
        params["nominal"] = level
        params["p_value"] = float(stats.binomtest(k, n, level).pvalue)
    return MetricResult(name="Interval coverage", value=k / n, metric="interval_coverage", params=params)


def crps(
    y_true: Any,
    *,
    ensemble: Any = None,
    mean: Any = None,
    sd: Any = None,
) -> MetricResult:
    """Mean continuous ranked probability score. Give ``ensemble`` (n × m sample paths / members) for the
    empirical CRPS E|X − y| − ½E|X − X'|, or ``mean`` and ``sd`` for normal forecasts (closed form). Lower is
    better; for a point forecast it reduces to the absolute error."""
    from scipy import stats

    y = _vec(y_true, "y_true")
    if ensemble is not None:
        e = np.asarray(ensemble, dtype=np.float64)
        if e.ndim == 1:
            e = e[:, None]
        if e.shape[0] != y.size or not np.all(np.isfinite(e)):
            raise InputValidationError(f"ensemble must be finite with {y.size} rows; got shape {e.shape}.")
        m = e.shape[1]
        s = np.sort(e, axis=1)
        term1 = np.abs(e - y[:, None]).mean(axis=1)
        # E|X − X'| from sorted members: (2/m²) Σ_i (2i − m − 1) x_(i)
        idx = np.arange(1, m + 1)
        term2 = (2.0 / m**2) * ((2 * idx - m - 1) * s).sum(axis=1)
        v = term1 - 0.5 * term2
        params: dict[str, Any] = {"form": "ensemble", "members": m}
    elif mean is not None and sd is not None:
        mu = np.broadcast_to(np.asarray(mean, dtype=np.float64), y.shape)
        sg = np.broadcast_to(np.asarray(sd, dtype=np.float64), y.shape)
        if np.any(sg < 0) or not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sg))):
            raise InputValidationError("mean and sd must be finite, with sd ≥ 0.")
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (y - mu) / sg
            v = sg * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))
        v = np.where(sg == 0, np.abs(y - mu), v)
        params = {"form": "normal"}
    else:
        raise InputValidationError("crps needs ensemble=, or mean= and sd=.")
    return MetricResult(name="CRPS", value=float(np.mean(v)), metric="crps", params=params)


def diebold_mariano(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    *,
    h: int = 1,
    loss: str = "squared",
    alternative: str = "two-sided",
    harvey: bool = True,
) -> MetricResult:
    """Diebold–Mariano test of equal predictive accuracy of two forecasts of the same series. ``loss`` is
    ``"squared"`` or ``"absolute"``; ``h`` is the forecast horizon (the loss differential's autocorrelation is
    allowed up to lag h − 1). The value is the DM statistic (negative: A's losses are smaller); ``params`` holds
    the p-value and the mean loss difference. ``alternative="less"`` tests that A is more accurate."""
    from scipy import stats

    y, a = _pair(y_true, pred_a)
    _, b = _pair(y_true, pred_b)
    if h < 1:
        raise InputValidationError("h must be at least 1.")
    if loss == "squared":
        with np.errstate(over="ignore", invalid="ignore"):
            d = (y - a) ** 2 - (y - b) ** 2
    elif loss == "absolute":
        d = np.abs(y - a) - np.abs(y - b)
    else:
        raise InputValidationError("loss must be 'squared' or 'absolute'.")
    if not np.all(np.isfinite(d)):
        raise InputValidationError("The loss differential overflows; rescale the series before testing.")
    n = d.size
    if n < 3 or h >= n:
        raise InputValidationError("The Diebold–Mariano test needs more observations than the horizon (and ≥ 3).")
    dbar = float(d.mean())
    dc = d - dbar
    gamma = [float(np.dot(dc[k:], dc[: n - k]) / n) for k in range(h)]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        raise InputValidationError(
            "The loss differential's long-run variance estimate is not positive (identical losses, or h too large "
            "for a short series); the Diebold–Mariano statistic is undefined."
        )
    dm = dbar / math.sqrt(lrv / n)
    if harvey:
        dm *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dist: Any = stats.t(df=n - 1)
    else:
        dist = stats.norm()
    if alternative == "two-sided":
        p = float(2 * dist.sf(abs(dm)))
    elif alternative == "less":
        p = float(dist.cdf(dm))
    elif alternative == "greater":
        p = float(dist.sf(dm))
    else:
        raise InputValidationError("alternative must be 'two-sided', 'less' or 'greater'.")
    return MetricResult(
        name="Diebold–Mariano statistic",
        value=float(dm),
        metric="diebold_mariano",
        params={
            "p_value": p,
            "mean_loss_difference": dbar,
            "h": h,
            "loss": loss,
            "alternative": alternative,
            "correction": "Harvey-Leybourne-Newbold" if harvey else None,
            "n": n,
        },
    )


@dataclass(frozen=True)
class ForecastReport:
    """Accuracy of several forecasts of one series, with Diebold–Mariano tests against a baseline."""

    rows: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def best(self, metric: str = "mase") -> str:
        key = metric if any(metric in r for r in self.rows) else "mae"
        return str(min(self.rows, key=lambda r: r[key])["model"])

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        lines = [f"Forecast report (n={s['n']}, baseline {s['baseline']!r}, {s['loss']} loss, h={s['h']})"]
        for r in self.rows:
            extra = f"  MASE {r['mase']:.{d}f}" if "mase" in r else ""
            dm = (
                f"  DM vs baseline {r['dm']:.{d}f} (p_holm={r['dm_p_adjusted']:.{d}g})"
                if r.get("dm") is not None
                else ""
            )
            lines.append(
                f"  {r['model']:<16} MAE {r['mae']:.{d}f}  RMSE {r['rmse']:.{d}f}  "
                f"sMAPE {r['smape']:.{d}f}{extra}{dm}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe({"rows": list(self.rows), "settings": dict(self.settings)})
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        has_mase = any("mase" in r for r in self.rows)
        head = "| Model | MAE | RMSE | sMAPE |" + (" MASE |" if has_mase else "") + " DM vs baseline | p (Holm) |"
        sep = "| --- | ---: | ---: | ---: |" + (" ---: |" if has_mase else "") + " ---: | ---: |"
        out = [head, sep]
        for r in self.rows:
            m = f" {r['mase']:.{d}f} |" if has_mase else ""
            dm = f"{r['dm']:.{d}f}" if r.get("dm") is not None else "—"
            p = f"{r['dm_p_adjusted']:.{d}g}" if r.get("dm") is not None else "—"
            out.append(
                f"| {r['model']} | {r['mae']:.{d}f} | {r['rmse']:.{d}f} | {r['smape']:.{d}f} |{m} {dm} | {p} |"
            )
        return "\n".join(out) + "\n"


def forecast_report(
    y_true: Any,
    forecasts: Mapping[str, Any],
    *,
    y_train: Any = None,
    m: int = 1,
    baseline: Optional[str] = None,
    h: int = 1,
    loss: str = "squared",
) -> ForecastReport:
    """MAE, RMSE, sMAPE (and MASE with ``y_train``) for each forecast, and a Holm-adjusted Diebold–Mariano test
    of each against ``baseline`` (the first forecast by default)."""
    from .stats.effect import adjust_pvalues

    if not isinstance(forecasts, Mapping) or not forecasts:
        raise InputValidationError("forecasts must be a non-empty mapping of name → predictions.")
    names: Sequence[str] = list(forecasts)
    base = baseline if baseline is not None else names[0]
    if base not in forecasts:
        raise InputValidationError(f"baseline {base!r} is not one of the forecasts.")
    y = _vec(y_true, "y_true")
    rows: list[dict[str, Any]] = []
    pvals: list[float] = []
    for name in names:
        _, p = _pair(y, forecasts[name])
        e = y - p
        den = np.abs(y) + np.abs(p)
        smape = float(np.mean(np.where(den == 0, 0.0, 2 * np.abs(e) / np.where(den == 0, 1.0, den))))
        row: dict[str, Any] = {
            "model": name,
            "mae": float(np.mean(np.abs(e))),
            "rmse": math.sqrt(float(np.mean(e**2))),
            "smape": smape,
        }
        if y_train is not None:
            row["mase"] = mase(y, p, y_train=y_train, m=m).value
        if name != base:
            try:
                t = diebold_mariano(y, p, forecasts[base], h=h, loss=loss)
                row["dm"], row["dm_p"] = t.value, t.params["p_value"]
                pvals.append(t.params["p_value"])
            except InputValidationError:
                row["dm"], row["dm_p"] = None, None
        rows.append(row)
    adj = iter(adjust_pvalues(pvals, method="holm")) if pvals else iter(())
    for row in rows:
        if row.get("dm") is not None:
            row["dm_p_adjusted"] = float(next(adj))
    return ForecastReport(tuple(rows), {"n": y.size, "baseline": base, "h": h, "loss": loss, "m": m})
