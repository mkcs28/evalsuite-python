"""Confidence intervals: bootstrap (percentile, basic, BCa), Wilson and Clopper-Pearson, DeLong."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scipy import special, stats
else:
    from .._lazy import lazy_module

    special = lazy_module("scipy.special")
    stats = lazy_module("scipy.stats")

from .._native import array_view, core
from ..core.exceptions import EvalSuiteError, InputValidationError, StatisticalTestError
from ..core.types import ArrayLike
from ..core.validation import to_numpy
from ._resolve import MetricCall, resolve_metric
from .results import ConfidenceInterval

__all__ = ["accuracy_ci", "bootstrap_ci", "proportion_ci", "roc_auc_ci"]

BootstrapMethod = Literal["percentile", "basic", "bca"]
MAX_FAILED_FRACTION = 0.1


def _check_level(level: float) -> float:
    if not (isinstance(level, (int, float)) and 0 < level < 1):
        raise InputValidationError("level must be between 0 and 1, for example 0.95.")
    return float(level)


def _check_resamples(n_resamples: int) -> int:
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 100:
        raise InputValidationError("n_resamples must be an integer of at least 100 (1000 or more recommended).")
    return int(n_resamples)


RESAMPLING = ("numpy", "native")


def check_resampling(resampling: str) -> str:
    if resampling not in RESAMPLING:
        raise InputValidationError("resampling must be 'numpy' or 'native'.")
    if resampling == "native" and core is None:
        raise InputValidationError(
            "resampling='native' needs the compiled core, which is not available (pure-Python install or "
            "EVALSUITE_PURE_PYTHON=1)."
        )
    return resampling


def _strata_groups(n: int, strata: NDArray[Any]) -> list[NDArray[np.intp]]:
    values = np.unique(strata, axis=0) if strata.ndim > 1 else np.unique(strata)
    return [
        np.flatnonzero(np.all(strata == value, axis=1) if strata.ndim > 1 else strata == value) for value in values
    ]


def resample_indices(
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    strata: Optional[NDArray[Any]] = None,
    resampling: str = "numpy",
) -> NDArray[np.int64]:
    """(n_resamples, n) indices drawn with replacement, optionally within strata (keeps class balance).

    ``resampling="native"`` draws them in the compiled core (xoshiro256++, in parallel) from one seed taken from
    ``rng``: reproducible for a given ``random_state`` and thread count–independent, but a different stream from
    NumPy's, so intervals differ (statistically equivalently) from the default."""
    if resampling == "native":
        c = core
        if c is None:  # pragma: no cover - checked by check_resampling
            raise InputValidationError("resampling='native' needs the compiled core.")
        seed = int(rng.integers(0, 2**63))
        groups = None
        if strata is not None:
            parts = _strata_groups(n, strata)
            members, view = array_view(np.concatenate(parts), np.int64)
            groups = (view, [0, *np.cumsum([p.shape[0] for p in parts]).tolist()])
        out = np.empty((n_resamples, n), dtype=np.int64)
        c.bootstrap_indices((int(out.ctypes.data), int(out.size)), n, seed, groups)
        return out
    if strata is None:
        return rng.integers(0, n, size=(n_resamples, n))
    out = np.empty((n_resamples, n), dtype=np.int64)
    pos = 0
    for members in _strata_groups(n, strata):
        k = members.shape[0]
        out[:, pos : pos + k] = members[rng.integers(0, k, size=(n_resamples, k))]
        pos += k
    return out


def bootstrap_distribution(call: MetricCall, indices: NDArray[np.int64]) -> tuple[NDArray[np.float64], int]:
    """Metric value on each resample; resamples where the metric is undefined become NaN."""
    # Binary ROC AUC / average precision: every resample's curve in one parallel compiled call.
    curves = call.batch_curves(indices)
    if curves is not None:
        return curves, int(np.isnan(curves).sum())
    values = np.empty(indices.shape[0])
    failed = 0
    # Confusion-matrix metrics: count every resample's (true, predicted) pairs in one compiled pass.
    counts = call.batch_code_counts(indices)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-division warnings inside resamples are expected noise
        for b, idx in enumerate(indices):
            try:
                values[b] = call(idx) if counts is None else call.from_counts(counts[b], idx)
            except EvalSuiteError:
                values[b] = np.nan
                failed += 1
    return values, failed


def _jackknife(call: MetricCall, n: int, rng: np.random.Generator, groups: int) -> NDArray[np.float64]:
    """Leave-one-out (or leave-one-group-out for large n) estimates for the BCa acceleration."""
    blocks = [np.array([i]) for i in range(n)] if n <= groups else np.array_split(rng.permutation(n), groups)
    fast = call.leave_out_curves(blocks)
    if fast is not None:
        return fast
    everything = np.arange(n)
    out = np.empty(len(blocks))
    # Confusion-matrix metrics: the counts without a block are the full counts minus the block's.
    full = call.code_counts(everything) if call.compressible else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, block in enumerate(blocks):
            try:
                if full is not None:
                    counts = full - call.code_counts(block)
                    if np.count_nonzero(counts) >= 2:
                        out[j] = call.from_counts(counts)
                        continue
                keep = np.setdiff1d(everything, block, assume_unique=True)
                out[j] = call(keep, compress=False)
            except EvalSuiteError:
                out[j] = np.nan
    return out


def bootstrap_ci(
    metric: Any,
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    y_prob: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
    level: float = 0.95,
    method: BootstrapMethod = "bca",
    n_resamples: int = 2000,
    random_state: Optional[int] = None,
    stratify: Optional[bool] = None,
    jackknife_groups: int = 1000,
    resampling: str = "numpy",
    clusters: Optional[ArrayLike] = None,
    **metric_kwargs: Any,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for any metric.

        ``metric`` is a metric function (``es.f1``) or its name (``"f1"``). Pass ``y_pred`` for label/value
        metrics or ``y_prob`` for probability metrics; extra keyword arguments go to the metric (for example
        ``average="macro"``). Resampling is stratified by class for classification targets unless
        ``stratify=False``, so every resample contains every class. For classification metrics that accept
        ``labels``, the full label set is fixed across resamples.

        ``method``: ``"bca"`` (bias-corrected and accelerated; default, Efron 1987), ``"percentile"`` or
        ``"basic"``. Set ``random_state`` for reproducible intervals. Resamples on which the metric is undefined
        are counted in ``params["failed_resamples"]``; more than 10% raises :class:`StatisticalTestError`.
    ``resampling="native"`` (opt-in) draws the resample indices in the compiled core instead of NumPy: faster on
    large inputs and still reproducible with ``random_state``, but a different random stream from the default.

    ``clusters`` (one id per observation: patient, document, site, conversation) resamples whole clusters
    instead of observations (cluster bootstrap; Field & Welsh 2007), so correlated observations in a cluster are
    kept together. Ordinary intervals are too narrow when observations are clustered; the cluster interval is
    honest. BCa's acceleration then uses leave-one-cluster-out estimates.

        References: Efron B. Better bootstrap confidence intervals. JASA. 1987;82(397):171-185.
        Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall; 1993.
    """
    level = _check_level(level)
    n_resamples = _check_resamples(n_resamples)
    resampling = check_resampling(resampling)
    if method not in ("percentile", "basic", "bca"):
        raise InputValidationError("method must be 'bca', 'percentile' or 'basic'.")
    fn, name = resolve_metric(metric)
    call = MetricCall(fn, y_true, y_pred, y_prob, sample_weight, metric_kwargs)
    estimate = call()
    rng = np.random.default_rng(random_state)
    if clusters is not None:
        return _cluster_bootstrap(
            call,
            name,
            estimate,
            clusters,
            level,
            method,
            n_resamples,
            rng,
            random_state,
            jackknife_groups,
            metric_kwargs,
        )
    use_strata = call.categorical if stratify is None else bool(stratify)
    idx = resample_indices(call.n, n_resamples, rng, call.y_true if use_strata else None, resampling)
    boot, failed = bootstrap_distribution(call, idx)
    if failed > MAX_FAILED_FRACTION * n_resamples:
        raise StatisticalTestError(
            f"The metric was undefined on {failed} of {n_resamples} bootstrap resamples. The sample is probably "
            "too small or too imbalanced for a reliable bootstrap interval."
        )
    alpha = 1 - level
    valid = boot[~np.isnan(boot)]
    params: dict[str, Any] = {
        "n_resamples": n_resamples,
        "random_state": random_state,
        "stratified": use_strata,
        "failed_resamples": failed,
        "resampling": resampling,
        **{k: v for k, v in metric_kwargs.items()},
    }
    if method == "percentile":
        low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    elif method == "basic":
        q_low, q_high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
        low, high = 2 * estimate - q_high, 2 * estimate - q_low
    else:
        low, high, extra = _bca(call, valid, estimate, alpha, rng, jackknife_groups)
        params.update(extra)
    return ConfidenceInterval(estimate, float(low), float(high), level, f"bootstrap-{method}", name, params)


def _cluster_members(clusters: ArrayLike, n: int) -> list[NDArray[np.int64]]:
    c = np.asarray(clusters).ravel()
    if c.shape[0] != n:
        raise InputValidationError(f"clusters must have one id per observation ({n}); got {c.shape[0]}.")
    _, codes = np.unique(c, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return [m.astype(np.int64) for m in np.split(order, bounds)]


def _leave_blocks_out(call: MetricCall, blocks: list[NDArray[Any]]) -> NDArray[np.float64]:
    """The metric without each block of observations (jackknife), NaN where undefined; counts or the compiled
    core when the metric allows, identical to calling it on the kept observations."""
    if not call.compressible:
        fast = call.leave_out_curves(blocks)
        if fast is not None:
            return fast
    everything = np.arange(call.n)
    jack = np.empty(len(blocks))
    total = call.code_counts(everything) if call.compressible else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, block in enumerate(blocks):
            try:
                if total is not None:
                    counts = total - call.code_counts(block)
                    few = np.count_nonzero(counts) < 2
                    keep = np.setdiff1d(everything, block, assume_unique=True) if few else None
                    jack[j] = call.from_counts(counts, keep)
                else:
                    jack[j] = call(np.setdiff1d(everything, block, assume_unique=True))
            except EvalSuiteError:
                jack[j] = np.nan
    return jack


def _weighted_curves(
    is_auc: bool,
    target: NDArray[np.float64],
    score: NDArray[np.float64],
    weight: Optional[NDArray[np.float64]],
    cluster_of: NDArray[np.int64],
    mult: NDArray[Any],
) -> NDArray[np.float64]:
    """Binary ROC AUC or average precision of every cluster resample, from how often each cluster was drawn
    (``mult``, resamples × clusters): a drawn cluster's observations count with that multiplicity, which is the
    same as concatenating them. One sort for all resamples, then O(n) per resample. NaN where undefined."""
    order = np.argsort(-score if not is_auc else score, kind="stable")
    s = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]  # tie groups
    t = target[order]
    base = np.ones_like(t) if weight is None else weight[order]
    pos_w, neg_w = base * t, base * (1.0 - t)
    owner = cluster_of[order]
    ties = starts.shape[0] < s.shape[0]
    multf = mult.astype(np.float64)
    out = np.empty(mult.shape[0])
    step = max(1, int(4_000_000 // max(1, s.shape[0])))
    for a in range(0, mult.shape[0], step):
        f = multf[a : a + step][:, owner]  # resamples × observations
        pos, neg = f * pos_w, f * neg_w
        if ties:
            pos = np.add.reduceat(pos, starts, axis=1)
            neg = np.add.reduceat(neg, starts, axis=1)
        p_tot, n_tot = pos.sum(axis=1), neg.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            if is_auc:
                below = np.cumsum(neg, axis=1) - neg
                val = (pos * (below + 0.5 * neg)).sum(axis=1) / (p_tot * n_tot)
                val[(p_tot == 0) | (n_tot == 0)] = np.nan
            else:
                tp, fp = np.cumsum(pos, axis=1), np.cumsum(neg, axis=1)
                prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
                val = (pos * prec).sum(axis=1) / p_tot
                val[p_tot == 0] = np.nan
        out[a : a + step] = val
    return out


def _mean_terms(call: MetricCall) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64], bool]]:
    """(per-observation terms, weights, take square root) for metrics that are a weighted mean of
    per-observation terms (binary log loss and Brier score; MAE, MSE, RMSE and mean bias error of one output),
    else None. Terms are computed on the full data exactly as the metric computes them on any subset."""
    from ..classification import metrics as cm
    from ..regression import metrics as rm

    fn = call.fn
    if call.items or call.y_true.ndim != 1 or call.second.ndim != 1:
        return None
    try:
        if fn in (cm.log_loss, cm.brier_score):
            if set(call.kwargs) - {"labels", "pos_label"}:
                return None
            ctx = cm._ctx(
                call.y_true, None, y_prob=call.second, labels=call.kwargs.get("labels"), sample_weight=call.weight
            )
            if ctx.target_type != "binary":
                return None
            y = cm._binary_target(ctx, call.kwargs.get("pos_label"))
            if fn is cm.log_loss:
                eps = np.finfo(np.float64).eps
                p = np.clip(ctx.y_prob, eps, 1 - eps)
                terms = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            else:
                terms = (ctx.y_prob - y) ** 2
            w = np.ones(ctx.n) if ctx.weights is None else np.asarray(ctx.weights, dtype=np.float64)
            return np.asarray(terms, dtype=np.float64), w, False
        if fn in (rm.mae, rm.mse, rm.rmse, rm.mean_bias_error):
            if set(call.kwargs) - {"multioutput"}:
                return None
            inp = rm._Inputs(call.y_true, call.second, call.weight)
            e = inp.error[:, 0]
            terms = np.abs(e) if fn is rm.mae else e if fn is rm.mean_bias_error else e * e
            if not np.all(np.isfinite(terms)):
                return None
            return terms, inp.w, fn is rm.rmse
    except EvalSuiteError:
        return None
    return None


def cluster_distribution(
    call: MetricCall, members: list[NDArray[np.int64]], picks: list[NDArray[Any]]
) -> tuple[NDArray[np.float64], int]:
    """Metric value on each cluster resample (``picks[b]`` lists the drawn clusters), NaN where undefined.

    Equal to concatenating the drawn clusters' observations and calling the metric, but much faster:
    confusion-matrix metrics add up per-cluster code counts (one small matrix product for all resamples), and
    binary ROC AUC / average precision weight each observation by how often its cluster was drawn."""
    n_resamples = len(picks)
    boot = np.empty(n_resamples)
    failed = 0

    def rows(b: int) -> NDArray[np.int64]:
        return np.concatenate([members[k] for k in picks[b]])

    g = len(members)
    cluster_of = np.empty(call.n, dtype=np.int64)
    for k, m in enumerate(members):
        cluster_of[m] = k
    mult = np.zeros((n_resamples, g), dtype=np.int64)
    for b, pick in enumerate(picks):
        mult[b] = np.bincount(pick, minlength=g)
    codes = call._codes
    if codes is not None and n_resamples > 0:
        kk = call._k * call._k
        per = np.bincount(cluster_of * kk + codes, minlength=g * kk).reshape(g, kk)
        totals = mult @ per
    else:
        totals = None
        mean = _mean_terms(call) if n_resamples > 0 else None
        if mean is not None:
            terms, w, root = mean
            num = mult @ np.bincount(cluster_of, weights=w * terms, minlength=g)
            den = mult @ np.bincount(cluster_of, weights=w, minlength=g)
            with np.errstate(invalid="ignore", divide="ignore"):
                boot = np.where(den > 0, num / den, np.nan)
            if root:
                boot = np.sqrt(boot)
            return boot, int(np.isnan(boot).sum())
        inputs = call._curve_inputs() if n_resamples > 0 else None
        if inputs is not None:
            from ..classification.metrics import roc_auc

            _, target, score, w = inputs
            boot = _weighted_curves(call.fn is roc_auc, target, score, w, cluster_of, mult)
            return boot, int(np.isnan(boot).sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in range(n_resamples):
            try:
                if totals is not None:
                    counts = totals[b]
                    boot[b] = call.from_counts(counts, rows(b) if np.count_nonzero(counts) < 2 else None)
                else:
                    boot[b] = call(rows(b))
            except EvalSuiteError:
                boot[b] = np.nan
                failed += 1
    return boot, failed


def _cluster_bootstrap(
    call: MetricCall,
    name: str,
    estimate: float,
    clusters: ArrayLike,
    level: float,
    method: str,
    n_resamples: int,
    rng: np.random.Generator,
    random_state: Optional[int],
    groups: int,
    metric_kwargs: dict[str, Any],
) -> ConfidenceInterval:
    members = _cluster_members(clusters, call.n)
    g = len(members)
    if g < 2:
        raise InputValidationError("A cluster bootstrap needs at least two clusters.")
    boot, failed = cluster_distribution(call, members, [rng.integers(0, g, g) for _ in range(n_resamples)])
    if failed > MAX_FAILED_FRACTION * n_resamples:
        raise StatisticalTestError(
            f"The metric was undefined on {failed} of {n_resamples} cluster resamples; too few clusters?"
        )
    alpha = 1 - level
    valid = boot[~np.isnan(boot)]
    params: dict[str, Any] = {
        "n_resamples": n_resamples,
        "random_state": random_state,
        "clusters": g,
        "failed_resamples": failed,
        "resampling": "cluster",
        **metric_kwargs,
    }
    if method == "percentile":
        low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    elif method == "basic":
        q_low, q_high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
        low, high = 2 * estimate - q_high, 2 * estimate - q_low
    else:
        share = (np.sum(valid < estimate) + 0.5 * np.sum(valid == estimate)) / valid.shape[0]
        if share <= 0 or share >= 1:
            low, high = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
            params["bca_fallback"] = "percentile"
        else:
            z0 = special.ndtri(share)
            if g <= groups:
                blocks = members
            else:  # leave out groups of clusters
                parts = np.array_split(rng.permutation(g), groups)
                blocks = [np.concatenate([members[k] for k in part]) for part in parts]
            jack = _leave_blocks_out(call, blocks)
            jack = jack[~np.isnan(jack)]
            diffs = jack.mean() - jack
            denom = 6.0 * np.sum(diffs**2) ** 1.5
            accel = float(np.sum(diffs**3) / denom) if denom > 0 else 0.0
            probs = []
            for z_alpha in (special.ndtri(alpha / 2), special.ndtri(1 - alpha / 2)):
                num = z0 + z_alpha
                probs.append(special.ndtr(z0 + num / (1 - accel * num)))
            low, high = np.quantile(valid, probs)
            params.update(bias_correction=float(z0), acceleration=accel)
    return ConfidenceInterval(
        estimate, float(low), float(high), level, f"cluster-bootstrap-{method}", name, params
    )


def _bca(
    call: MetricCall,
    boot: NDArray[np.float64],
    estimate: float,
    alpha: float,
    rng: np.random.Generator,
    groups: int,
) -> tuple[float, float, dict[str, Any]]:
    # bias correction: share of the bootstrap distribution below the estimate (ties count half)
    share = (np.sum(boot < estimate) + 0.5 * np.sum(boot == estimate)) / boot.shape[0]
    if share <= 0 or share >= 1:
        warnings.warn(
            "BCa is undefined because the estimate lies outside the bootstrap distribution; "
            "returning the percentile interval.",
            RuntimeWarning,
            stacklevel=3,
        )
        low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return float(low), float(high), {"bca_fallback": "percentile"}
    z0 = special.ndtri(share)
    jack = _jackknife(call, call.n, rng, groups)
    jack = jack[~np.isnan(jack)]
    diffs = jack.mean() - jack
    denom = 6.0 * np.sum(diffs**2) ** 1.5
    accel = float(np.sum(diffs**3) / denom) if denom > 0 else 0.0
    probs = []
    for z_alpha in (special.ndtri(alpha / 2), special.ndtri(1 - alpha / 2)):
        num = z0 + z_alpha
        probs.append(special.ndtr(z0 + num / (1 - accel * num)))
    low, high = np.quantile(boot, probs)
    return float(low), float(high), {"bias_correction": float(z0), "acceleration": accel}


def proportion_ci(
    successes: int,
    n: int,
    *,
    level: float = 0.95,
    method: Literal["wilson", "clopper-pearson", "normal"] = "wilson",
) -> ConfidenceInterval:
    """Confidence interval for a proportion ``successes / n``.

    ``"wilson"`` (default; good coverage even for small n or extreme proportions), ``"clopper-pearson"``
    (exact, conservative) or ``"normal"`` (Wald; shown for comparison, not recommended).

    References: Wilson EB. JASA. 1927;22(158):209-212. Clopper CJ, Pearson ES. Biometrika. 1934;26(4):404-413.
    Brown LD, Cai TT, DasGupta A. Interval estimation for a binomial proportion. Stat Sci. 2001;16(2):101-133.
    """
    level = _check_level(level)
    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise InputValidationError("n must be a positive integer.")
    if not (0 <= successes <= n):
        raise InputValidationError("successes must be between 0 and n.")
    p = successes / n
    alpha = 1 - level
    z = float(special.ndtri(1 - alpha / 2))
    if method == "wilson":
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        # Rounding can leave a bound a hair past the estimate at 0 or n successes; the interval always holds it.
        low, high = max(0.0, min(p, centre - half)), min(1.0, max(p, centre + half))
    elif method == "clopper-pearson":
        # Beta quantiles via the inverse regularised incomplete beta function (what scipy.stats.beta.ppf uses).
        low = 0.0 if successes == 0 else special.betaincinv(successes, n - successes + 1, alpha / 2)
        high = 1.0 if successes == n else special.betaincinv(successes + 1, n - successes, 1 - alpha / 2)
    elif method == "normal":
        half = z * math.sqrt(p * (1 - p) / n)
        low, high = max(0.0, p - half), min(1.0, p + half)
    else:
        raise InputValidationError("method must be 'wilson', 'clopper-pearson' or 'normal'.")
    return ConfidenceInterval(
        p, float(low), float(high), level, method, "proportion", {"successes": int(successes), "n": int(n)}
    )


def accuracy_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    level: float = 0.95,
    method: Literal["wilson", "clopper-pearson", "normal"] = "wilson",
) -> ConfidenceInterval:
    """Analytic confidence interval for accuracy (a proportion of correct predictions)."""
    yt = to_numpy(y_true, "y_true", allow_2d=True)
    yp = to_numpy(y_pred, "y_pred", allow_2d=True)
    if yt.shape != yp.shape:
        raise InputValidationError(f"y_true has shape {yt.shape} but y_pred has shape {yp.shape}.")
    correct = (yt == yp).all(axis=1) if yt.ndim == 2 else yt == yp
    ci = proportion_ci(int(np.count_nonzero(correct)), int(correct.shape[0]), level=level, method=method)
    return ConfidenceInterval(ci.estimate, ci.low, ci.high, level, method, "accuracy", ci.params)


# ---- DeLong ---------------------------------------------------------------------------------------
def delong_placements(
    y: NDArray[np.float64], scores: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """AUCs and structural components for k score columns (Sun & Xu fast DeLong).

    Returns (auc[k], v10[k, m], v01[k, n]) where m/n are the numbers of positives/negatives.
    """
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m == 0 or n == 0:
        from ..core.exceptions import MetricInputError

        raise MetricInputError("DeLong's method needs at least one positive and one negative observation.")
    c = core
    if c is not None:
        # One compiled sort per column gives all three rankings (exact half-integer mid-ranks).
        keep = np.ascontiguousarray(y == 1)
        cols = [np.ascontiguousarray(row, dtype=np.float64) for row in scores]
        out = c.delong_ranks(array_view(keep, np.bool_)[1], [array_view(col, np.float64)[1] for col in cols])
        as_rows = [np.vstack([np.frombuffer(r[j], dtype=np.float64) for r in out]) for j in range(4)]
        tz = np.concatenate([as_rows[0], as_rows[1]], axis=1)
        tx, ty = as_rows[2], as_rows[3]
    else:
        tx = np.apply_along_axis(stats.rankdata, 1, pos)
        ty = np.apply_along_axis(stats.rankdata, 1, neg)
        tz = np.apply_along_axis(stats.rankdata, 1, np.concatenate([pos, neg], axis=1))
    auc = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v10, v01


def delong_covariance(v10: NDArray[np.float64], v01: NDArray[np.float64]) -> NDArray[np.float64]:
    m, n = v10.shape[1], v01.shape[1]
    s10 = np.atleast_2d(np.cov(v10)) if m > 1 else np.zeros((v10.shape[0],) * 2)
    s01 = np.atleast_2d(np.cov(v01)) if n > 1 else np.zeros((v01.shape[0],) * 2)
    cov: NDArray[np.float64] = s10 / m + s01 / n
    return cov


def binary_scores(
    y_true: ArrayLike, y_prob: ArrayLike, pos_label: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yt = to_numpy(y_true, "y_true")
    s = to_numpy(y_prob, "y_prob").astype(np.float64)
    if s.shape[0] != yt.shape[0]:
        raise InputValidationError(
            "y_true and y_prob must contain the same number of observations. "
            f"Received {yt.shape[0]} and {s.shape[0]}."
        )
    labels = np.unique(yt)
    if labels.shape[0] > 2:
        raise InputValidationError("DeLong's method is for binary targets.")
    if pos_label is None:
        if not set(labels.tolist()) <= {0, 1}:
            raise InputValidationError(f"Labels are {labels.tolist()}; specify pos_label=...")
        pos_label = 1
    return (yt == pos_label).astype(np.float64), s


def roc_auc_ci(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    level: float = 0.95,
    pos_label: Any = None,
) -> ConfidenceInterval:
    """DeLong confidence interval for a binary ROC AUC (normal approximation, clipped to [0, 1]).

    Reference: DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated
    receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845.
    Sun X, Xu W. Fast implementation of DeLong's algorithm. IEEE Signal Process Lett. 2014;21(11):1389-1393.
    """
    level = _check_level(level)
    y, s = binary_scores(y_true, y_prob, pos_label)
    auc, v10, v01 = delong_placements(y, s[None, :])
    var = float(delong_covariance(v10, v01)[0, 0])
    se = np.sqrt(var)
    z = special.ndtri(1 - (1 - level) / 2)
    a = float(auc[0])
    return ConfidenceInterval(
        a,
        max(0.0, a - z * se),
        min(1.0, a + z * se),
        level,
        "delong",
        "roc_auc",
        {"standard_error": float(se), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum())},
    )
