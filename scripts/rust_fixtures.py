"""Write parity fixtures for the Rust `evalsuite` crate from the Python package.

Each fixture file (rust/evalsuite/tests/fixtures/<group>.json) is a list of cases
``{"fn": name, "args": {...}, "expected": value | {"error": kind}}``; the Rust test
``tests/parity.rs`` replays every case and compares with a relative tolerance.

    python scripts/rust_fixtures.py            # all groups
    python scripts/rust_fixtures.py regression # one group
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

import evalsuite as es
from evalsuite.core.exceptions import EvalSuiteError

OUT = Path(__file__).resolve().parents[1] / "rust" / "evalsuite" / "tests" / "fixtures"
GROUPS: dict[str, Callable[[np.random.Generator], list[dict[str, Any]]]] = {}


def group(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        GROUPS[name] = fn
        return fn

    return deco


def plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else ("nan" if math.isnan(f) else ("inf" if f > 0 else "-inf"))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def case(fn: str, call: Callable[[], Any], **args: Any) -> dict[str, Any]:
    """Run ``call`` and record its value (``.value`` of a MetricResult) or the error kind."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = call()
            value = out.value if hasattr(out, "value") and hasattr(out, "metric") else out
            expected: Any = plain(value)
        except EvalSuiteError as e:
            expected = {"error": type(e).__name__}
    return {"fn": fn, "args": plain(args), "expected": expected}


# ---- classification ------------------------------------------------------------------------------------
@group("classification")
def classification(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    averages = [None, "micro", "macro", "weighted", "binary"]
    for t in range(40):
        n = int(rng.integers(5, 300))
        k = 2 if t % 3 == 0 else int(rng.integers(3, 6))
        y = rng.integers(0, k, n)
        p = np.where(rng.random(n) < 0.6, y, rng.integers(0, k, n))
        w = (rng.random(n) * 2).round(3) if t % 2 else None
        for avg in averages:
            if avg == "binary" and k != 2:
                continue
            for name in ("precision", "recall", "f1", "specificity", "npv", "jaccard"):
                fn = getattr(es, name)
                out.append(
                    case(
                        name,
                        lambda fn=fn, avg=avg: fn(y, p, average="auto" if avg == "binary" else avg, sample_weight=w),
                        y_true=y,
                        y_pred=p,
                        average=avg or "none",
                        sample_weight=w,
                    )
                )
            out.append(
                case("fbeta", lambda avg=avg: es.fbeta(y, p, beta=0.5, average="auto" if avg == "binary" else avg, sample_weight=w),
                     y_true=y, y_pred=p, beta=0.5, average=avg or "none", sample_weight=w)
            )
        for name in ("accuracy", "hamming_loss", "mcc"):
            fn = getattr(es, name)
            out.append(case(name, lambda fn=fn: fn(y, p, sample_weight=w), y_true=y, y_pred=p, sample_weight=w))
        out.append(case("balanced_accuracy", lambda: es.balanced_accuracy(y, p, adjusted=bool(t % 2), sample_weight=w),
                        y_true=y, y_pred=p, adjusted=bool(t % 2), sample_weight=w))
        for kw in (None, "linear", "quadratic"):
            out.append(case("cohen_kappa", lambda kw=kw: es.cohen_kappa(y, p, weights=kw, sample_weight=w),
                            y_true=y, y_pred=p, weights=kw or "none", sample_weight=w))
        out.append(case("confusion_matrix", lambda: es.confusion_matrix(y, p, sample_weight=w, normalize="true"),
                        y_true=y, y_pred=p, sample_weight=w, normalize="true"))
        # probabilities
        if k == 2:
            s = np.round(np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1), 2)
            for name in ("roc_auc", "average_precision", "log_loss", "brier_score"):
                fn = getattr(es, name)
                out.append(case("binary." + name, lambda fn=fn: fn(y, s, sample_weight=w), y_true=y, y_prob=s, sample_weight=w))
            out.append(case("binary.roc_curve", lambda: es.roc_curve(y, s, sample_weight=w), y_true=y, y_prob=s, sample_weight=w))
            out.append(case("binary.pr_curve", lambda: es.pr_curve(y, s, sample_weight=w), y_true=y, y_prob=s, sample_weight=w))
            for strategy in ("uniform", "quantile"):
                out.append(case("binary.expected_calibration_error",
                                lambda strategy=strategy: es.expected_calibration_error(y, s, n_bins=7, strategy=strategy, sample_weight=w),
                                y_true=y, y_prob=s, n_bins=7, strategy=strategy, sample_weight=w))
                out.append(case("binary.calibration_curve",
                                lambda strategy=strategy: es.calibration_curve(y, s, n_bins=7, strategy=strategy, sample_weight=w),
                                y_true=y, y_prob=s, n_bins=7, strategy=strategy, sample_weight=w))
        else:
            P = rng.dirichlet(np.ones(k), n).round(4)
            P = P / P.sum(1, keepdims=True)
            for avg in ("macro", "weighted", None):
                out.append(case("roc_auc", lambda avg=avg: es.roc_auc(y, P, average=avg, sample_weight=w),
                                y_true=y, y_prob=P, average=avg or "none", multi_class="ovr", sample_weight=w))
                out.append(case("average_precision", lambda avg=avg: es.average_precision(y, P, average=avg, sample_weight=w),
                                y_true=y, y_prob=P, average=avg or "none", sample_weight=w))
            out.append(case("roc_auc", lambda: es.roc_auc(y, P, multi_class="ovo", sample_weight=w),
                            y_true=y, y_prob=P, average="macro", multi_class="ovo", sample_weight=w))
            out.append(case("average_precision", lambda: es.average_precision(y, P, average="micro", sample_weight=w),
                            y_true=y, y_prob=P, average="micro", sample_weight=w))
            for name in ("log_loss", "brier_score"):
                fn = getattr(es, name)
                out.append(case(name, lambda fn=fn: fn(y, P, sample_weight=w), y_true=y, y_prob=P, sample_weight=w))
            out.append(case("top_k_accuracy", lambda: es.top_k_accuracy(y, P, k=2, sample_weight=w), y_true=y, y_prob=P, k=2, sample_weight=w))
    # multilabel
    for t in range(10):
        n, k = int(rng.integers(5, 200)), int(rng.integers(2, 5))
        Y = rng.integers(0, 2, (n, k))
        Pm = np.where(rng.random((n, k)) < 0.7, Y, 1 - Y)
        S = np.round(np.clip(Y * 0.4 + rng.random((n, k)) * 0.6, 0, 1), 2)
        w = rng.random(n).round(3) if t % 2 else None
        for avg in ("micro", "macro", "weighted", "samples", None):
            for name in ("precision", "recall", "f1", "jaccard"):
                fn = getattr(es, name)
                out.append(case("multilabel." + name, lambda fn=fn, avg=avg: fn(Y, Pm, average=avg, sample_weight=w),
                                y_true=Y, y_pred=Pm, average=avg or "none", sample_weight=w))
        for name in ("accuracy", "hamming_loss"):
            fn = getattr(es, name)
            out.append(case("multilabel." + name, lambda fn=fn: fn(Y, Pm, sample_weight=w), y_true=Y, y_pred=Pm, sample_weight=w))
        for avg in ("micro", "macro", "weighted", None):
            for name in ("roc_auc", "average_precision"):
                fn = getattr(es, name)
                out.append(case("multilabel." + name, lambda fn=fn, avg=avg: fn(Y, S, average=avg, sample_weight=w),
                                y_true=Y, y_prob=S, average=avg or "none", sample_weight=w))
    # string labels and errors
    ys = np.array(["cat", "dog", "cat", "cat", "dog"])
    ps = np.array(["cat", "cat", "cat", "dog", "dog"])
    out.append(case("f1", lambda: es.f1(ys, ps, pos_label="dog"), y_true=ys, y_pred=ps, average="binary", pos_label="dog"))
    out.append(case("f1", lambda: es.f1(ys, ps), y_true=ys, y_pred=ps, average="binary"))
    out.append(case("binary.roc_auc", lambda: es.roc_auc([1, 1, 1], [0.2, 0.4, 0.9]), y_true=[1, 1, 1], y_prob=[0.2, 0.4, 0.9]))
    return out


# ---- regression ----------------------------------------------------------------------------------------
@group("regression")
def regression(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    names = ["mae", "mse", "rmse", "r2", "mape", "smape", "msle", "rmsle", "median_absolute_error",
             "explained_variance", "mean_bias_error", "rae", "rse"]
    for t in range(40):
        n = int(rng.integers(2, 400))
        yt = rng.normal(5, 3, n).round(3)
        if t % 4 == 0:
            yt = np.abs(yt)
        yp = (yt + rng.normal(0, 1, n)).round(3)
        if t % 4 == 0:
            yp = np.abs(yp)
        if t % 9 == 0:
            yt[:] = yt[0]
        w = (rng.random(n) * 3).round(3) if t % 2 else None
        for name in names:
            fn = getattr(es, name)
            out.append(case(name, lambda fn=fn: fn(yt, yp, sample_weight=w), y_true=yt, y_pred=yp, sample_weight=w))
        out.append(case("max_error", lambda: es.max_error(yt, yp), y_true=yt, y_pred=yp))
        out.append(case("adjusted_r2", lambda: es.adjusted_r2(yt, yp, n_features=3, sample_weight=w),
                        y_true=yt, y_pred=yp, n_features=3, sample_weight=w))
        out.append(case("quantile_loss", lambda: es.quantile_loss(yt, yp, alpha=0.3, sample_weight=w),
                        y_true=yt, y_pred=yp, alpha=0.3, sample_weight=w))
        out.append(case("huber_loss", lambda: es.huber_loss(yt, yp, delta=1.5, sample_weight=w),
                        y_true=yt, y_pred=yp, delta=1.5, sample_weight=w))
    for t in range(8):
        n, k = int(rng.integers(3, 100)), int(rng.integers(2, 4))
        Yt = np.abs(rng.normal(5, 3, (n, k)).round(3))
        Yp = np.abs((Yt + rng.normal(0, 1, (n, k))).round(3))
        for name in ("mae", "rmse", "r2", "msle"):
            fn = getattr(es, name)
            for mo in ("uniform_average", "raw_values", [1.0, 2.0] + [1.0] * (k - 2)):
                out.append(case("multioutput." + name, lambda fn=fn, mo=mo: fn(Yt, Yp, multioutput=mo),
                                y_true=Yt, y_pred=Yp, multioutput=mo))
    return out


# ---- clinical, calibration, uncertainty ----------------------------------------------------------------
@group("clinical")
def clinical(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in range(40):
        n = int(rng.integers(4, 300))
        y = rng.integers(0, 2, n)
        p = np.where(rng.random(n) < 0.75, y, 1 - y)
        if t % 7 == 0:
            p = y.copy()  # perfect test: infinite / zero ratios
        w = (rng.random(n) * 2).round(3) if t % 2 else None
        for name in ("sensitivity", "ppv", "lr_positive", "lr_negative", "youden_j"):
            fn = getattr(es, name)
            out.append(case(name, lambda fn=fn: fn(y, p, sample_weight=w), y_true=y, y_pred=p, sample_weight=w))
        for corr in (0.0, 0.5):
            out.append(case("diagnostic_odds_ratio", lambda corr=corr: es.diagnostic_odds_ratio(y, p, sample_weight=w, correction=corr),
                            y_true=y, y_pred=p, sample_weight=w, correction=corr))
        r = np.round(np.clip(y * 0.3 + rng.random(n) * 0.7, 0.001, 0.999), 3)
        for thr in (0.1, 0.3, 0.6):
            out.append(case("net_benefit", lambda thr=thr: es.net_benefit(y, r, threshold=thr, sample_weight=w),
                            y_true=y, y_prob=r, threshold=thr, sample_weight=w))
        from evalsuite.clinical.metrics import _binary_risk, net_benefit_curve
        thr = np.array([0.05, 0.2, 0.5, 0.8])
        yy, pp, ww = _binary_risk(y, r, None, w)
        out.append(case("decision_curve", lambda: np.concatenate(net_benefit_curve(yy, pp, ww, thr)),
                        y_true=y, y_prob=r, thresholds=thr, sample_weight=w))
    return out


@group("calibration")
def calibration(rng: np.random.Generator) -> list[dict[str, Any]]:
    from evalsuite import calibration as cal

    out = []
    for t in range(30):
        n = int(rng.integers(12, 400))
        r = np.round(np.clip(rng.beta(2, 3, n), 0.001, 0.999), 3)
        y = (rng.random(n) < r ** (0.7 + t % 3 * 0.3)).astype(int)
        w = (rng.random(n) * 2 + 0.1).round(3) if t % 2 else None
        for strategy in ("uniform", "quantile"):
            out.append(case("maximum_calibration_error",
                            lambda strategy=strategy: cal.maximum_calibration_error(y, r, n_bins=8, strategy=strategy, sample_weight=w),
                            y_true=y, y_prob=r, n_bins=8, strategy=strategy, sample_weight=w))
        out.append(case("calibration_slope", lambda: (lambda m: [m.value, m.params["intercept"]])(cal.calibration_slope(y, r, sample_weight=w)),
                        y_true=y, y_prob=r, sample_weight=w))
        out.append(case("calibration_intercept", lambda: cal.calibration_intercept(y, r, sample_weight=w), y_true=y, y_prob=r, sample_weight=w))
        out.append(case("hosmer_lemeshow", lambda: (lambda h: [h.statistic, h.p_value])(cal.hosmer_lemeshow(y, r, n_groups=8)),
                        y_true=y, y_prob=r, n_groups=8))
    return out


@group("uncertainty")
def uncertainty(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in range(40):
        n = int(rng.integers(3, 300))
        conf = np.round(rng.random(n), int(rng.integers(1, 4)))
        c = rng.random(n) < conf
        if t % 9 == 0:
            c[:] = True
        out.append(case("adaptive_calibration_error", lambda: es.adaptive_calibration_error(c, conf, n_bins=10), correct=c, confidence=conf, n_bins=10))
        for cov in (0.3, 0.8, 1.0):
            out.append(case("selective_risk", lambda cov=cov: es.selective_risk(c, conf, coverage=cov), correct=c, confidence=conf, coverage=cov))
            out.append(case("risk_at_coverage", lambda cov=cov: es.risk_at_coverage(c, conf, coverage=cov), correct=c, confidence=conf, coverage=cov))
        for risk in (0.0, 0.1, 0.4):
            out.append(case("coverage_at_risk", lambda risk=risk: es.coverage_at_risk(c, conf, risk=risk), correct=c, confidence=conf, risk=risk))
        out.append(case("aurc", lambda: (lambda m: [m.value, m.params["e_aurc"]])(es.aurc(c, conf)), correct=c, confidence=conf))
        out.append(case("confidence_accuracy_correlation", lambda: es.confidence_accuracy_correlation(c, conf), correct=c, confidence=conf))
        from evalsuite.llmsys.uncertainty import risk_coverage_curve
        out.append(case("risk_coverage_curve", lambda: np.concatenate(risk_coverage_curve(c, conf)), correct=c, confidence=conf))
    return out


@group("retrieval")
def retrieval(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in range(40):
        nq = int(rng.integers(1, 40))
        retrieved, relevant = [], []
        for _ in range(nq):
            retrieved.append([int(x) for x in rng.permutation(60)[: int(rng.integers(0, 25))]])
            ids = [int(x) for x in rng.choice(60, int(rng.integers(1, 8)), replace=False)]
            relevant.append({i: float(rng.choice([1, 2, 3, 0.5])) for i in ids} if t % 2 else ids)
        rel_json = [{str(k): v for k, v in r.items()} if isinstance(r, dict) else {str(k): 1.0 for k in r} for r in relevant]
        for k in (1, 5, 10):
            for name in ("precision_at_k", "recall_at_k", "hit_rate_at_k"):
                fn = getattr(es, name)
                out.append(case(name, lambda fn=fn, k=k: fn(relevant, retrieved, k=k, average=None), relevant=rel_json, retrieved=retrieved, k=k))
        for k in (None, 3, 10):
            out.append(case("mrr", lambda k=k: es.mrr(relevant, retrieved, k=k, average=None), relevant=rel_json, retrieved=retrieved, k=k))
            out.append(case("mean_average_precision_at_k", lambda k=k: es.mean_average_precision_at_k(relevant, retrieved, k=k, average=None),
                            relevant=rel_json, retrieved=retrieved, k=k))
            for gains in ("linear", "exponential"):
                out.append(case("ndcg_at_k", lambda k=k, gains=gains: es.ndcg_at_k(relevant, retrieved, k=k, gains=gains, average=None),
                                relevant=rel_json, retrieved=retrieved, k=k, gains=gains))
    return out


@group("stats")
def stats_group(rng: np.random.Generator) -> list[dict[str, Any]]:
    st = es
    out = []
    alts = ("two-sided", "greater", "less")

    def tr(r: Any) -> list[float]:
        v = [r.statistic, r.p_value, r.estimate]
        if r.ci is not None:
            v += [r.ci.low, r.ci.high]
        return v

    for t in range(30):
        n1, n2 = int(rng.integers(2, 60)), int(rng.integers(2, 60))
        a = rng.normal(0, 1, n1).round(int(rng.integers(0, 3)) + 1)
        b = rng.normal(0.3, 1.5, n2).round(int(rng.integers(0, 3)) + 1)
        alt = alts[t % 3]
        for ev in (False, True):
            out.append(case("t_test", lambda ev=ev: tr(st.t_test(a, b, equal_var=ev, alternative=alt)), a=a, b=b, equal_var=ev, alternative=alt))
        out.append(case("mann_whitney_test", lambda: tr(st.mann_whitney_test(a, b, alternative=alt)), a=a, b=b, alternative=alt))
        m = min(n1, n2)
        out.append(case("paired_t_test", lambda: tr(st.paired_t_test(a[:m], b[:m], alternative=alt)), a=a[:m], b=b[:m], alternative=alt))
        out.append(case("wilcoxon_test", lambda: tr(st.wilcoxon_test(a[:m], b[:m], alternative=alt)), a=a[:m], b=b[:m], alternative=alt))
        c = rng.normal(0, 1, n1).round(1)
        out.append(case("kruskal_wallis_test", lambda: tr(st.kruskal_wallis_test(a, b, c)), groups=[a, b, c]))
        g = [rng.normal(i * 0.2, 1, m).round(1) for i in range(int(rng.integers(3, 6)))]
        out.append(case("friedman_test", lambda: tr(st.friedman_test(*g)), groups=g))
        out.append(case("shapiro_wilk_test", lambda: (lambda r: [r.statistic, r.p_value])(st.shapiro_wilk_test(a)), x=a))
        for paired in (False, True):
            aa, bb = (a[:m], b[:m]) if paired else (a, b)
            out.append(case("cohens_d", lambda aa=aa, bb=bb, paired=paired: st.cohens_d(aa, bb, paired=paired), a=aa, b=bb, paired=paired))
            out.append(case("hedges_g", lambda aa=aa, bb=bb, paired=paired: st.hedges_g(aa, bb, paired=paired), a=aa, b=bb, paired=paired))
        out.append(case("cliffs_delta", lambda: st.cliffs_delta(a, b), a=a, b=b))
        tab = rng.integers(0, 30, (int(rng.integers(2, 4)), int(rng.integers(2, 4)))) + (t % 2)
        for corr in (True, False):
            out.append(case("chi_square_test", lambda corr=corr: tr(st.chi_square_test(tab, correction=corr)), table=tab, correction=corr))
            out.append(case("cramers_v", lambda corr=corr: st.cramers_v(tab, bias_correction=corr), table=tab, bias_correction=corr))
        t2 = rng.integers(0, 15 if t % 2 else 200, (2, 2))
        out.append(case("fisher_exact_test", lambda: tr(st.fisher_exact_test(t2, alternative=alt)), table=t2, alternative=alt))
        pv = rng.random(int(rng.integers(1, 40))).round(3)
        for meth in ("bonferroni", "holm", "hochberg", "bh", "by"):
            out.append(case("adjust_pvalues", lambda meth=meth: st.adjust_pvalues(pv, method=meth), p=pv, method=meth))
        k, nn = int(rng.integers(0, 50)), 50
        for meth in ("wilson", "clopper-pearson", "normal"):
            out.append(case("proportion_ci", lambda meth=meth: (lambda r: [r.estimate, r.low, r.high])(st.proportion_ci(k, nn, level=0.9, method=meth)),
                            successes=k, n=nn, level=0.9, method=meth))
        yt = rng.integers(0, 2, 80)
        pa = np.where(rng.random(80) < 0.7, yt, 1 - yt)
        pb = np.where(rng.random(80) < 0.6, yt, 1 - yt)
        for ex in (None, True, False):
            out.append(case("mcnemar_test", lambda ex=ex: tr(st.mcnemar_test(yt, pa, pb, exact=ex)), y_true=yt, a=pa, b=pb, exact=ex))
        sa = np.round(np.clip(yt * 0.3 + rng.random(80) * 0.7, 0, 1), 2)
        sb = np.round(np.clip(yt * 0.2 + rng.random(80) * 0.8, 0, 1), 2)
        out.append(case("roc_auc_ci", lambda: (lambda r: [r.estimate, r.low, r.high])(st.roc_auc_ci(yt, sa)), y_true=yt, y_prob=sa))
        out.append(case("delong_test", lambda: tr(st.delong_test(yt, sa, sb)), y_true=yt, a=sa, b=sb))
        # bootstrap with the compiled core's stream: record the seed Python derives from random_state.
        seed = int(np.random.default_rng(t).integers(0, 2**63))
        for method in ("percentile", "basic", "bca"):
            for metric in ("accuracy", "roc_auc"):
                kw = {"y_pred": pa} if metric == "accuracy" else {"y_prob": sa}
                out.append(case("bootstrap_ci",
                                lambda method=method, metric=metric, kw=kw: (lambda r: [r.estimate, r.low, r.high])(
                                    st.bootstrap_ci(metric, yt, **kw, n_resamples=300, method=method, random_state=t, resampling="native", stratify=bool(t % 2))),
                                metric=metric, y_true=yt, pred=pa if metric == "accuracy" else sa, method=method, seed=seed, stratify=bool(t % 2)))
        out.append(case("paired_bootstrap_test",
                        lambda: tr(st.paired_bootstrap_test("accuracy", yt, pa, pb, n_resamples=300, random_state=t, resampling="native", stratify=False, alternative=alt)),
                        y_true=yt, a=pa, b=pb, seed=seed, alternative=alt))
    return out


def _blob(rng: np.random.Generator, shape: tuple[int, ...], k: int) -> np.ndarray:
    m = np.zeros(shape, dtype=np.int64)
    grid = np.ogrid[tuple(slice(0, s) for s in shape)]
    for c in range(1, k):
        for _ in range(int(rng.integers(0, 3))):
            ctr = [int(rng.integers(0, s)) for s in shape]
            rad = int(rng.integers(1, max(2, min(shape) // 3)))
            m[sum((g - ci) ** 2 for g, ci in zip(grid, ctr)) <= rad * rad] = c
    return m


@group("segmentation")
def segmentation(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in range(24):
        shape = (int(rng.integers(5, 40)), int(rng.integers(5, 40))) if t % 3 else (int(rng.integers(4, 14)),) * 3
        k = int(rng.integers(2, 5))
        n_img = int(rng.integers(1, 4))
        yt = [_blob(rng, shape, k) for _ in range(n_img)]
        yp = [np.where(rng.random(shape) < 0.85, m, rng.integers(0, k, shape)) for m in yt]
        ign = 9 if t % 4 == 0 else None
        if ign is not None:
            for m in yt:
                m[rng.random(shape) < 0.05] = ign
        imgs = {"y_true": [m.tolist() for m in yt], "y_pred": [m.tolist() for m in yp], "shape": list(shape), "ignore_index": ign}
        for name in ("dice", "iou"):
            fn = getattr(es, name)
            for avg in ("macro", "micro", "weighted", None):
                for agg in ("dataset", "image"):
                    for empty in ("ignore", 1.0):
                        out.append(case(name, lambda fn=fn, avg=avg, agg=agg, empty=empty: fn(yt, yp, num_classes=k, ignore_index=ign, average=avg, aggregate=agg, empty=empty),
                                        **imgs, num_classes=k, average=avg or "none", aggregate=agg, empty=empty))
        out.append(case("pixel_accuracy", lambda: es.pixel_accuracy(yt, yp, ignore_index=ign), **imgs))
        out.append(case("mean_pixel_accuracy", lambda: es.mean_pixel_accuracy(yt, yp, num_classes=k, ignore_index=ign), **imgs, num_classes=k))
        for avg in ("macro", None):
            out.append(case("boundary_iou", lambda avg=avg: es.boundary_iou(yt, yp, num_classes=k, ignore_index=ign, average=avg, dilation_ratio=0.05),
                            **imgs, num_classes=k, average=avg or "none"))
            sp = None if t % 2 else [0.7, 1.3, 1.1][: len(shape)]
            out.append(case("hausdorff_distance", lambda avg=avg, sp=sp: es.hausdorff_distance(yt, yp, num_classes=k, ignore_index=ign, average=avg, spacing=sp),
                            **imgs, num_classes=k, average=avg or "none", spacing=sp, percentile=100))
            out.append(case("hausdorff_distance", lambda avg=avg, sp=sp: es.hausdorff_distance(yt, yp, num_classes=k, ignore_index=ign, average=avg, spacing=sp, percentile=95),
                            **imgs, num_classes=k, average=avg or "none", spacing=sp, percentile=95))
            out.append(case("average_surface_distance", lambda avg=avg, sp=sp: es.average_surface_distance(yt, yp, num_classes=k, ignore_index=ign, average=avg, spacing=sp),
                            **imgs, num_classes=k, average=avg or "none", spacing=sp))
    return out


@group("detection")
def detection(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in range(25):
        imgs_t, imgs_p = [], []
        for _ in range(int(rng.integers(1, 8))):
            g = int(rng.integers(0, 8))
            xy = rng.uniform(0, 300, (g, 2))
            wh = rng.uniform(5, 150, (g, 2))
            gb = np.round(np.hstack([xy, xy + wh]), 1)
            gl = rng.integers(0, 3, g)
            crowd = rng.random(g) < 0.1
            keep = rng.random(g) < 0.8
            db = np.round(gb[keep] + rng.normal(0, 6, (int(keep.sum()), 4)), 1)
            db[:, 2:] = np.maximum(db[:, 2:], db[:, :2] + 1)
            extra = int(rng.integers(0, 4))
            exy = rng.uniform(0, 300, (extra, 2))
            db = np.vstack([db, np.round(np.hstack([exy, exy + rng.uniform(5, 80, (extra, 2))]), 1)])
            dl = np.concatenate([gl[keep], rng.integers(0, 3, extra)])
            ds = np.round(rng.random(db.shape[0]), 2)
            imgs_t.append({"boxes": gb.tolist(), "labels": gl.tolist(), "iscrowd": crowd.tolist()})
            imgs_p.append({"boxes": db.tolist(), "labels": dl.tolist(), "scores": ds.tolist()})
        args = {"y_true": imgs_t, "y_pred": imgs_p}
        out.append(case("mean_average_precision", lambda: es.mean_average_precision(imgs_t, imgs_p), **args, iou_threshold=None))
        out.append(case("mean_average_precision", lambda: es.mean_average_precision(imgs_t, imgs_p, iou_threshold=0.5), **args, iou_threshold=0.5))
        for interp in ("coco", "voc"):
            out.append(case("average_precision_detection",
                            lambda interp=interp: es.average_precision_detection(imgs_t, imgs_p, iou_threshold=0.5, interpolation=interp),
                            **args, interpolation=interp))
        from evalsuite.vision.detection import detection_report
        out.append(case("detection_report", lambda: [v for _, v in detection_report(imgs_t, imgs_p).values.items()], **args))
        b1 = np.round(rng.uniform(0, 50, (4, 4)), 1); b1[:, 2:] += b1[:, :2]
        b2 = np.round(rng.uniform(0, 50, (3, 4)), 1); b2[:, 2:] += b2[:, :2]
        out.append(case("box_iou", lambda: es.box_iou(b1, b2), a=b1, b=b2))
    return out


_WORDS = ("the a cat dog sat on mat quickly , . ! ? it's 3.5 e-mail don't New York run running runs "
          "jumped &amp; <skipped> x-1 1,000 (test) [ok] \"quote\" éclair ﬁne ßtraße").split(" ")


def _sentence(rng: np.random.Generator, lo: int = 0, hi: int = 18) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(int(rng.integers(lo, hi))))


@group("generation")
def generation(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in range(30):
        n = int(rng.integers(1, 25))
        preds = [_sentence(rng) for _ in range(n)]
        refs = [[_sentence(rng) for _ in range(int(rng.integers(1, 4)))] for _ in range(n)]
        if t % 3 == 0:
            refs = [[p] + rs[1:] for p, rs in zip(preds, refs)]
        if t % 5 == 0:
            refs = [[r.replace(" ", "\n", 2) for r in rs] for rs in refs]
        a = {"references": refs, "predictions": preds}
        for smooth in ("exp", "floor", "add-k", "none"):
            for lc in (False, True):
                out.append(case("bleu", lambda smooth=smooth, lc=lc: es.bleu(refs, preds, smooth=smooth, lowercase=lc), **a, smooth=smooth, lowercase=lc, tokenize="13a"))
        out.append(case("bleu", lambda: es.bleu(refs, preds, tokenize="none"), **a, smooth="exp", lowercase=False, tokenize="none"))
        out.append(case("sentence_bleu", lambda: es.sentence_bleu(refs, preds, average=None), **a))
        if n >= 2:
            out.append(case("self_bleu", lambda: es.self_bleu(preds), predictions=preds))
        for wo in (0, 2):
            for lc in (False, True):
                out.append(case("chrf", lambda wo=wo, lc=lc: es.chrf(refs, preds, word_order=wo, lowercase=lc), **a, word_order=wo, lowercase=lc, whitespace=False))
        out.append(case("chrf", lambda: es.chrf(refs, preds, whitespace=True, eps_smoothing=True), **a, word_order=0, lowercase=False, whitespace=True, eps_smoothing=True))
        for cs in (False, True):
            out.append(case("ter", lambda cs=cs: es.ter(refs, preds, case_sensitive=cs), **a, case_sensitive=cs))
        for name in ("rouge_1", "rouge_2", "rouge_l", "rouge_lsum"):
            fn = getattr(es, name)
            for measure in ("fmeasure", "precision", "recall"):
                out.append(case(name, lambda fn=fn, measure=measure: fn(refs, preds, measure=measure, average=None), **a, measure=measure))
        out.append(case("meteor", lambda: es.meteor(refs, preds, stemmer=None, average=None), **a))
        out.append(case("meteor", lambda: es.meteor(refs, preds, stemmer=None, alpha=0.7, beta=2.0, gamma=0.3, tokenize="13a", average=None),
                        **a, alpha=0.7, beta=2.0, gamma=0.3, tokenize="13a"))
        for lc in (False, True):
            out.append(case("cider", lambda lc=lc: es.cider(refs, preds, lowercase=lc, average=None), **a, lowercase=lc))
        for k in (1, 2, 3):
            out.append(case("distinct_n", lambda k=k: es.distinct_n(preds, n=k, lowercase=bool(t % 2)), predictions=preds, n=k, lowercase=bool(t % 2)))
        lp = [(-rng.exponential(1.0, int(rng.integers(1, 30)))).round(4) for _ in range(int(rng.integers(1, 6)))]
        out.append(case("cross_entropy", lambda: es.cross_entropy(lp, base=2), token_logprobs=lp, base=2))
        for avg in ("tokens", "sequences"):
            out.append(case("perplexity", lambda avg=avg: es.perplexity(lp, average=avg), token_logprobs=lp, average=avg))
    for line in ["Hello, world!", "3.5 e-mail x-1 1,000.", "&quot;a&quot; &amp; b<skipped>", "end-\nline", "a.b,c", "(ok) [no] {x}"]:
        from evalsuite.text.generation import _tok_13a
        out.append(case("tokenize_13a", lambda line=line: [len(_tok_13a(line))] + [ord(c) for c in "\x00".join(_tok_13a(line))], line=line))
    return out


# ---- agents --------------------------------------------------------------------------------------------
_TOOLS = {
    "search": {"type": "object", "required": ["q"], "properties": {"q": {"type": "string", "minLength": 1}, "k": {"type": "integer", "minimum": 1}}, "additionalProperties": False},
    "calc": {"type": "object", "properties": {"expr": {"type": "string", "pattern": "^[0-9+*/ -]+$"}}},
    "open": None,
}


def _call(rng: np.random.Generator) -> dict[str, Any]:
    name = ["search", "calc", "open", "delete"][int(rng.integers(0, 4))]
    r = int(rng.integers(0, 6))
    args: Any = [{"q": "x"}, {"q": ""}, {"q": "a", "k": 0}, {"expr": "1+2"}, '{"q": "y", "k": 2}', "{bad json"][r]
    c: dict[str, Any] = {"name": name, "arguments": args}
    o = int(rng.integers(0, 3))
    if o < 2:
        c["ok"] = bool(o)
    return c


@group("agents")
def agents(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for _ in range(60):
        n = int(rng.integers(1, 6))
        trajs = [[_call(rng) for _ in range(int(rng.integers(0, 6)))] for _ in range(n)]
        out.append(case("invalid_tool_call_rate", lambda t=trajs: es.invalid_tool_call_rate(t, _TOOLS), trajectories=trajs))
        for key in ("schema", "unknown_tool", "execution_failure_rate", "ci_low"):
            out.append(case("invalid_tool_call_rate." + key, lambda t=trajs, key=key: es.invalid_tool_call_rate(t, _TOOLS).params[key] if key in ("execution_failure_rate", "ci_low") else es.invalid_tool_call_rate(t, _TOOLS).params["by_reason"][key], trajectories=trajs))
        out.append(case("tool_failure_recovery_rate", lambda t=trajs: es.tool_failure_recovery_rate(t), trajectories=trajs))
        ll = int(rng.integers(2, 4))
        out.append(case("unnecessary_tool_call_rate", lambda t=trajs, ll=ll: es.unnecessary_tool_call_rate(t, loop_length=ll), trajectories=trajs, loop_length=ll))
        out.append(case("unnecessary_tool_call_rate.loop_rate", lambda t=trajs, ll=ll: es.unnecessary_tool_call_rate(t, loop_length=ll).params["loop_rate"], trajectories=trajs, loop_length=ll))
        m = int(rng.integers(1, 12))
        comp = [bool(x) for x in rng.integers(0, 2, m)]
        prog = rng.random(m).round(3).tolist()
        out.append(case("task_completion_rate", lambda c=comp: es.task_completion_rate(c), completed=comp))
        out.append(case("task_completion_rate.progress_rate", lambda c=comp, p=prog: es.task_completion_rate(c, progress=p).params["progress_rate"], completed=comp, progress=prog))
        a = rng.integers(0, 5, m).astype(float).tolist()
        o = rng.integers(0, 5, m).astype(float).tolist()
        out.append(case("tool_use_efficiency", lambda a=a, o=o: es.tool_use_efficiency(a, o), n_calls=a, optimal_calls=o))
        out.append(case("tool_use_efficiency.mean_excess_calls", lambda a=a, o=o: es.tool_use_efficiency(a, o).params["mean_excess_calls"], n_calls=a, optimal_calls=o))
        for key in ("value", "std", "p90", "p99", "mean_steps_completed"):
            out.append(case("steps_per_task." + key, lambda a=a, c=comp, key=key: (lambda r: r.value if key == "value" else r.params[key] if key == "mean_steps_completed" else r.params["steps"][key])(es.steps_per_task(a, completed=c)), steps=a, completed=comp))
        out.append(case("human_intervention_rate", lambda a=a: es.human_intervention_rate(a), interventions=a))
        cost = (rng.random(m) * 3).round(4).tolist()
        for key in ("value", "cost_per_success", "p50"):
            out.append(case("agent_cost_per_task." + key, lambda cst=cost, d=a, c=comp, key=key: (lambda r: r.value if key == "value" else r.params["time"]["p50"] if key == "p50" else r.params[key])(es.agent_cost_per_task(cst, durations=d, completed=c)), costs=cost, durations=a, completed=comp))
        steps = list("abcde")
        plans = [[steps[int(i)] for i in rng.integers(0, 5, int(rng.integers(0, 5)))] for _ in range(m)]
        exe = [[steps[int(i)] for i in rng.integers(0, 5, int(rng.integers(0, 5)))] for _ in range(m)]
        for key in ("value", "precision", "recall", "exact_match"):
            out.append(case("plan_adherence." + key, lambda p=plans, e=exe, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.plan_adherence(p, e)), plans=plans, executed=exe))
        sl = ["city", "date", "n"]
        st = lambda: {s: [1, 1.0, True, "x", "y"][int(rng.integers(0, 5))] for s in sl if rng.random() < 0.7}
        ex_, pr_ = [st() for _ in range(m)], [st() for _ in range(m)]
        for key in ("value", "slot_accuracy"):
            out.append(case("state_tracking_accuracy." + key, lambda e=ex_, p=pr_, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.state_tracking_accuracy(e, p)), expected=ex_, predicted=pr_))
    return out


# ---- structured outputs --------------------------------------------------------------------------------
_OUTS = ['{"a": 1, "b": [1, 2]}', ' [1, 2.5] ', 'Sure! ```json\n{"a": 2}\n``` done', 'text {"a": {"b": 3}} tail', "nope",
         '{"a": "x", "b": [1, 2, 2]}', '{"a": 1.0}', "```\n[true, null]\n```", "{'a': 1}", '{"a": -3, "c": "zz"}', "",
         '<root><x a="1"/></root>', "<root><x></root>", '<n:r xmlns:n="urn:a"/>', "<!DOCTYPE r><r/>", "  <a>t</a>\n"]
_SCHEMAS = [
    {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer", "minimum": 0}, "b": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}}},
    {"type": "array", "prefixItems": [{"type": "number"}], "items": {"type": ["number", "boolean", "null"]}, "minItems": 2},
    {"anyOf": [{"type": "object", "properties": {"a": {"const": 2}}}, {"type": "array", "contains": {"type": "boolean"}}]},
    {"type": "object", "properties": {"a": {"oneOf": [{"type": "integer"}, {"type": "number", "multipleOf": 0.5}]}}, "additionalProperties": {"type": "string", "pattern": "^z+$"}},
    {"$defs": {"pos": {"type": "integer", "exclusiveMinimum": 0}}, "type": "object", "properties": {"a": {"$ref": "#/$defs/pos"}}, "not": {"required": ["c"]}},
    {"type": "object", "patternProperties": {"^[ab]$": {"type": ["integer", "string", "array", "object"]}}, "propertyNames": {"maxLength": 1}, "maxProperties": 2},
    {"type": "object", "bogus": 1},
]


@group("structured")
def structured(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for _ in range(50):
        outs = [_OUTS[int(i)] for i in rng.integers(0, len(_OUTS), int(rng.integers(1, 7)))]
        for mode in ("strict", "fenced", "lenient"):
            out.append(case("json_validity", lambda o=outs, m=mode: es.json_validity(o, mode=m), outputs=outs, mode=mode))
            for si, sch in enumerate(_SCHEMAS):
                out.append(case("json_schema_compliance", lambda o=outs, m=mode, sch=sch: es.json_schema_compliance(o, sch, mode=m), outputs=outs, mode=mode, schema=sch))
        for rt in (None, "root", "{urn:a}r"):
            out.append(case("xml_validity", lambda o=outs, rt=rt: es.xml_validity(o, root_tag=rt), outputs=outs, root_tag=rt))
        out.append(case("extra_content_rate", lambda o=outs: es.extra_content_rate(o), outputs=outs))
        for pat in (r"\{.*\}", r"<[a-z]+>.*</[a-z]+>", r"\[?\d+.*"):
            out.append(case("format_compliance", lambda o=outs, p=pat: es.format_compliance(o, p), outputs=outs, pattern=pat))
            out.append(case("extra_content_rate.regex", lambda o=outs, p=pat: es.extra_content_rate(o, payload="regex", pattern=p), outputs=outs, pattern=pat))
        exp = [{"a": 1}, {"a": 2, "b.1": 2}, {"a.b": 3}, {"b.0": 1, "c": "zz"}]
        e2 = [exp[int(i)] for i in rng.integers(0, 4, len(outs))]
        for avg in ("micro", "macro"):
            for mode in ("strict", "lenient"):
                out.append(case("required_field_accuracy", lambda e=e2, o=outs, a=avg, m=mode: es.required_field_accuracy(e, o, mode=m, average=a), expected=e2, outputs=outs, average=avg, mode=mode))
        names = ["search", "calc", "open"]
        argsl = [{"q": "x"}, {"q": "x", "k": 1}, {"k": 1.0, "q": "x"}, '{"q": "x"}', "{oops", {}]
        mk = lambda: [[{"name": names[int(rng.integers(0, 3))], "arguments": argsl[int(rng.integers(0, len(argsl)))]} for _ in range(int(rng.integers(0, 4)))] for _ in range(len(outs))]
        ec, pc = mk(), mk()
        for ordered in (False, True):
            out.append(case("tool_selection_accuracy", lambda e=ec, p=pc, o=ordered: es.tool_selection_accuracy(e, p, ordered=o), expected=ec, predicted=pc, ordered=ordered))
        out.append(case("tool_argument_accuracy", lambda e=ec, p=pc: es.tool_argument_accuracy(e, p), expected=ec, predicted=pc))
        for m in ("arguments", "name"):
            for key in ("value", "precision", "recall", "invalid_call_rate"):
                out.append(case("tool_call_f1." + key, lambda e=ec, p=pc, m=m, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.tool_call_f1(e, p, match=m)), expected=ec, predicted=pc, match=m))
        res = [[True, False, 200, 404, 302, 99, None][int(i)] for i in rng.integers(0, 7, len(outs))]
        out.append(case("api_call_success_rate", lambda r=res: es.api_call_success_rate(r), results=res))
        chk = [[bool(x) for x in rng.integers(0, 2, int(rng.integers(0 if rng.random() < 0.05 else 1, 4)))] for _ in range(len(outs))]
        out.append(case("instruction_compliance_rate", lambda c=chk: es.instruction_compliance_rate(c), checks=chk))
        out.append(case("constraint_satisfaction_rate", lambda c=chk: es.constraint_satisfaction_rate(c), checks=chk))
        tc = [[[bool(x) for x in rng.integers(0, 2, int(rng.integers(0, 3)))] for _ in range(int(rng.integers(0, 3)))] for _ in range(len(outs))]
        out.append(case("instruction_retention", lambda t=tc: es.instruction_retention(t), turn_checks=tc))
    return out


# ---- QA and reasoning ----------------------------------------------------------------------------------
_ANS = ["Paris", "paris.", "The Eiffel Tower", "eiffel  tower!", "an apple", "Apple", "", "Ünïcode Straße", "ünïcode strasse",
        "42", "forty two", "the the", "A", "B) b", "Café-au-lait", "café au lait", "1,000", "1000"]
_REASON = ["So the answer is #### 1,234.00", "We get 3 apples and then $12.50 total", "\\boxed{\\frac{1}{2}} done", "\\fbox{ 7 }",
           "Answer: (C)", "I think B, not A.", "options A) and (D)", "nothing here", "line one\n\nfinal line  \n", "#### ",
           "The result is -7.0", "x\r\ny z", "ABC is (E)", "E.", "x \\boxed{a{b}c} y \\boxed{d", "answer:J", "-$3,000.000"]


@group("qa")
def qa(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for t in _ANS + _REASON:
        out.append(case("normalize_answer", lambda t=t: [ord(c) for c in es.normalize_answer(t)], text=t))
    for _ in range(80):
        n = int(rng.integers(1, 6))
        preds = [_ANS[int(i)] for i in rng.integers(0, len(_ANS), n)]
        refs = [[_ANS[int(i)] for i in rng.integers(0, len(_ANS), int(rng.integers(1, 3)))] for _ in range(n)]
        for norm in (True, False):
            out.append(case("exact_match", lambda r=refs, p=preds, nm=norm: es.exact_match(r, p, normalize=nm, average=None), references=refs, predictions=preds, normalize=norm))
            out.append(case("token_f1", lambda r=refs, p=preds, nm=norm: es.token_f1(r, p, normalize=nm, average=None), references=refs, predictions=preds, normalize=norm))
            samples = [[_ANS[int(i)] for i in rng.integers(0, len(_ANS), int(rng.integers(1, 6)))] for _ in range(n)]
            out.append(case("majority_vote_accuracy", lambda r=refs, s=samples, nm=norm: es.majority_vote_accuracy(r, s, normalize=nm, average=None), references=refs, samples=samples, normalize=norm))
        ns = rng.integers(1, 40, n).tolist()
        cs = [int(rng.integers(0, x + 1)) for x in ns]
        for k in (1, 5, 10):
            out.append(case("pass_at_k", lambda ns=ns, cs=cs, k=k: es.pass_at_k(ns, cs, k=k, average=None), n_samples=ns, n_correct=cs, k=k))
        for style in ("gsm8k", "boxed", "choice", "last_line"):
            golds = [_REASON[int(i)] if rng.random() < 0.5 else ["1234", "12.50", "-7", "C", "\\boxed{7}", "E", "final line"][int(rng.integers(0, 7))] for i in rng.integers(0, len(_REASON), n)]
            outs = [_REASON[int(i)] for i in rng.integers(0, len(_REASON), n)]
            out.append(case("benchmark_accuracy", lambda g=golds, o=outs, s=style: es.benchmark_accuracy(g, o, style=s, average=None), references=golds, outputs=outs, style=style))
    from evalsuite.text.reasoning import extract_answer
    for t in _REASON + _ANS:
        for style in ("gsm8k", "boxed", "choice", "last_line"):
            out.append(case("extract_answer", lambda t=t, s=style: (lambda a: [-1] if a is None else [ord(c) for c in a])(extract_answer(t, style=s)), text=t, style=style))
    return out


# ---- judge ---------------------------------------------------------------------------------------------
@group("judge")
def judge(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for _ in range(60):
        n = int(rng.integers(2, 30))
        oc = [[0.0, 0.5, 1.0][int(i)] for i in rng.integers(0, 3, n)]
        for ties in ("half", "exclude"):
            for key in ("value", "ci_low", "ci_high"):
                out.append(case("win_rate." + key, lambda o=oc, t=ties, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.win_rate(o, ties=t)), outcomes=oc, ties=ties))
        m = int(rng.integers(2, 6))
        comps = [[f"m{int(a)}", f"m{int(b)}", [0.0, 0.5, 1.0][int(rng.integers(0, 3))]] for a, b in rng.integers(0, m, (int(rng.integers(1, 60)), 2))]
        for sc in ("log", "elo"):
            out.append(case("bradley_terry", lambda c=comps, sc=sc: es.bradley_terry(c, scale=sc), comparisons=comps, scale=sc))
        kk = float(rng.choice([4.0, 16.0, 32.0]))
        out.append(case("elo_ratings", lambda c=comps, kk=kk: es.elo_ratings(c, k=kk), comparisons=comps, k=kk))
        raters, items = int(rng.integers(2, 5)), int(rng.integers(1, 15))
        rat = rng.integers(1, int(rng.integers(2, 6)), (raters, items)).astype(float)
        rat[rng.random(rat.shape) < 0.2] = np.nan
        for lv in ("nominal", "ordinal", "interval", "ratio"):
            out.append(case("krippendorff_alpha", lambda r=rat, lv=lv: es.krippendorff_alpha(r, level=lv), ratings=rat, level=lv))
        fl = rng.integers(0, int(rng.integers(1, 4)), (items, raters)).tolist()
        out.append(case("fleiss_kappa", lambda f=fl: es.fleiss_kappa(f), ratings=fl))
        j = rng.integers(1, 6, n).tolist()
        h = [x if rng.random() < 0.6 else int(rng.integers(1, 6)) for x in j]
        out.append(case("judge_agreement.categorical", lambda j=j, h=h: es.judge_agreement(j, h), judge=j, human=h))
        for key in ("value", "spearman"):
            out.append(case("judge_agreement.ordinal." + key, lambda j=j, h=h, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.judge_agreement(j, h, kind="ordinal")), judge=j, human=h))
        jc, hc = rng.normal(size=n).round(3).tolist(), rng.normal(size=n).round(3).tolist()
        for key in ("value", "spearman"):
            out.append(case("judge_agreement.continuous." + key, lambda j=jc, h=hc, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.judge_agreement(j, h, kind="continuous")), judge=jc, human=hc))
        vo = [["A", "B", "tie"][int(i)] for i in rng.integers(0, 3, n)]
        vs = [x if rng.random() < 0.5 else ["A", "B", "tie"][int(rng.integers(0, 3))] for x in vo]
        for key in ("value", "first_position_rate", "second_position_rate"):
            out.append(case("position_consistency." + key, lambda o=vo, s=vs, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.position_consistency(o, s)), original=vo, swapped=vs))
        la, lb = rng.integers(1, 6, n).tolist(), rng.integers(1, 6, n).tolist()
        for key in ("value", "p_value"):
            out.append(case("verbosity_bias." + key, lambda w=vo, a=la, b=lb, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.verbosity_bias(w, a, b)), winners=vo, length_a=la, length_b=lb))
        jp, hp = [bool(x) for x in rng.integers(0, 2, n)], [bool(x) for x in rng.integers(0, 2, n)]
        for key in ("value", "p_value"):
            out.append(case("self_preference_bias." + key, lambda j=jp, h=hp, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.self_preference_bias(j, h)), judge=jp, human=hp))
        sc_ = rng.integers(1, 6, (n, int(rng.integers(1, 4)))).astype(float).tolist()
        out.append(case("rubric_score", lambda s=sc_: es.rubric_score(s), scores=sc_))
        out.append(case("rubric_score.per_criterion", lambda s=sc_: list(es.rubric_score(s).params["per_criterion"].values()), scores=sc_))
    return out


# ---- RAG and factuality --------------------------------------------------------------------------------
@group("grounding")
def grounding(rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    V = ["supported", "contradicted", "unsupported"]
    for _ in range(80):
        n = int(rng.integers(1, 8))
        fl = lambda lo=0: [[bool(x) for x in rng.integers(0, 2, int(rng.integers(lo, 6)))] for _ in range(n)]
        cr = fl()
        out.append(case("context_precision", lambda c=cr: es.context_precision(c, average=None), flags=cr))
        out.append(case("context_recall", lambda c=cr: es.context_recall(c, average=None), flags=cr))
        out.append(case("context_recall.mean", lambda c=cr: es.context_recall(c), flags=cr))
        out.append(case("context_relevance.mean", lambda c=cr: es.context_relevance(c), flags=cr))
        lat = (rng.exponential(100, int(rng.integers(1, 50)))).round(2).tolist()
        for st in ("mean", "p50", "p90", "p95", "p99"):
            out.append(case("latency_summary." + st, lambda l=lat, st=st: es.latency_summary(l, statistic=st), latencies=lat))
        succ = [bool(x) for x in rng.integers(0, 2, n)]
        out.append(case("task_success_rate.ci_low", lambda s=succ: es.task_success_rate(s).params["ci_low"], success=succ))
        stages = [[None, "success", "Retrieval", " generation ", "evidence", "other", "none"][int(i)] for i in rng.integers(0, 7, n)]
        out.append(case("failure_attribution", lambda s=stages: es.failure_attribution(s), stages=stages))
        cv = [[V[int(i)] for i in rng.integers(0, 3, int(rng.integers(0, 5)))] for _ in range(n)]
        for avg in ("micro", "macro"):
            out.append(case("faithfulness", lambda c=cv, a=avg: es.faithfulness(c, average=a), claims=cv, average=avg))
            for cnt in ("unsupported", "contradicted"):
                out.append(case("hallucination_rate." + cnt, lambda c=cv, a=avg, cnt=cnt: es.hallucination_rate(c, count=cnt, average=a), claims=cv, average=avg))
            sup = [rng.random(int(rng.integers(0, 4))).round(3).tolist() for _ in range(n)]
            out.append(case("groundedness", lambda s=sup, a=avg: es.groundedness(s, average=a), scores=sup, average=avg))
            cit = [[{"supported": bool(rng.integers(0, 2)), "citations": [bool(x) for x in rng.integers(0, 2, int(rng.integers(0, 3)))]} for _ in range(int(rng.integers(0, 3)))] for _ in range(n)]
            out.append(case("citation_recall", lambda c=cit, a=avg: es.citation_recall(c, average=a), citations=cit, average=avg))
            out.append(case("citation_precision", lambda c=cit, a=avg: es.citation_precision(c, average=a), citations=cit, average=avg))
            facts = [[f"f{int(i)}" for i in rng.integers(0, 6, int(rng.integers(0, 4)))] for _ in range(n)]
            out.append(case("knowledge_consistency", lambda f=facts, a=avg: es.knowledge_consistency(f, {"f0", "f2", "f3"}, average=a), facts=facts, average=avg))
        gv = [V[int(i)] for i in rng.integers(0, 3, n)]
        pv = [x if rng.random() < 0.6 else V[int(rng.integers(0, 3))] for x in gv]
        for key in ("value", "macro_f1"):
            out.append(case("claim_verification_accuracy." + key, lambda g=gv, p=pv, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.claim_verification_accuracy(g, p)), gold=gv, predicted=pv))
        tp, fp, fn = (rng.integers(0, 4, n).astype(float).tolist() for _ in range(3))
        sim = rng.random(n).round(3).tolist()
        out.append(case("answer_correctness", lambda t=tp, f=fp, g=fn: es.answer_correctness(t, f, g, average=None), tp=tp, fp=fp, fneg=fn))
        out.append(case("answer_correctness.sim", lambda t=tp, f=fp, g=fn, s=sim: es.answer_correctness(t, f, g, similarity=s, average=None), tp=tp, fp=fp, fneg=fn, sim=sim))
        d = int(rng.integers(2, 6))
        q = rng.normal(size=(n, d)).round(3).tolist()
        gq = [rng.normal(size=(int(rng.integers(1, 4)), d)).round(3).tolist() for _ in range(n)]
        out.append(case("answer_relevance", lambda q=q, g=gq: es.answer_relevance(q, g, average=None), q=q, g=gq))
        sa, ab, co = ([bool(x) for x in rng.integers(0, 2, n)] for _ in range(3))
        for key in ("value", "abstention_precision", "abstention_recall", "coverage", "answered_accuracy"):
            out.append(case("abstention_accuracy." + key, lambda s=sa, a=ab, c=co, key=key: (lambda r: r.value if key == "value" else r.params[key])(es.abstention_accuracy(s, a, correct=c)), should=sa, abstained=ab, correct=co))
    return out


# ---- semantic ------------------------------------------------------------------------------------------
@group("semantic")
def semantic(rng: np.random.Generator) -> list[dict[str, Any]]:
    from evalsuite.text.semantic import mauve_from_histograms
    out = []
    for _ in range(60):
        n, d = int(rng.integers(1, 5)), int(rng.integers(2, 6))
        R = [rng.normal(size=(int(rng.integers(1, 7)), d)).round(3).tolist() for _ in range(n)]
        P = [rng.normal(size=(int(rng.integers(1, 7)), d)).round(3).tolist() for _ in range(n)]
        RW = [rng.integers(1, 4, len(r)).astype(float).tolist() for r in R]
        PW = [rng.integers(1, 4, len(p)).astype(float).tolist() for p in P]
        for m in ("f1", "precision", "recall"):
            out.append(case("bertscore." + m, lambda R=R, P=P, m=m: es.bertscore(R, P, measure=m, average=None), refs=R, preds=P))
        out.append(case("bertscore.weighted", lambda R=R, P=P, a=RW, b=PW: es.bertscore(R, P, reference_weights=a, prediction_weights=b, baseline=0.3, average=None), refs=R, preds=P, rw=RW, pw=PW))
        out.append(case("moverscore", lambda R=R, P=P: es.moverscore(R, P, average=None), refs=R, preds=P))
        out.append(case("moverscore.weighted", lambda R=R, P=P, a=RW, b=PW: es.moverscore(R, P, reference_weights=a, prediction_weights=b, average=None), refs=R, preds=P, rw=RW, pw=PW))
        A, B = rng.normal(size=(n, d)).round(3).tolist(), rng.normal(size=(n, d)).round(3).tolist()
        for m in ("cosine", "euclidean", "manhattan"):
            for nm in (False, True):
                out.append(case("embedding_similarity", lambda A=A, B=B, m=m, nm=nm: es.embedding_similarity(A, B, metric=m, normalize=nm, average=None), a=A, b=B, metric=m, normalize=nm))
        k = int(rng.integers(2, 8))
        ph, qh = rng.random(k), rng.random(k)
        ph[rng.random(k) < 0.2] = 0
        ph, qh = (ph / ph.sum()).tolist() if ph.sum() else [1 / k] * k, (qh / qh.sum()).tolist()
        for disc, sf in ((25, 5.0), (10, 2.0)):
            out.append(case("mauve_from_histograms", lambda p=ph, q=qh, disc=disc, sf=sf: mauve_from_histograms(np.array(p), np.array(q), discretization=disc, scaling_factor=sf), p=ph, q=qh, disc=disc, sf=sf))
    return out


# ---- SPICE ---------------------------------------------------------------------------------------------
_SYN = {"dog": ["canine"], "puppy": ["canine"], "grass": ["lawn"], "lawn": ["lawn"], "red": ["crimson"], "crimson": ["crimson"]}


@group("spice")
def spice_group(rng: np.random.Generator) -> list[dict[str, Any]]:
    objs = ["dog", "puppy", "grass", "lawn", "ball", "Dog "]
    attrs = ["red", "crimson", "small", "brown"]
    rels = ["on", "near", "with"]

    def tup() -> list[str]:
        k = int(rng.integers(1, 4))
        o = objs[int(rng.integers(0, len(objs)))]
        if k == 1:
            return [o]
        if k == 2:
            return [o, attrs[int(rng.integers(0, len(attrs)))]]
        return [o, rels[int(rng.integers(0, len(rels)))], objs[int(rng.integers(0, len(objs)))]]

    out = []
    for _ in range(150):
        n = int(rng.integers(1, 5))
        R = [[tup() for _ in range(int(rng.integers(0, 7)))] for _ in range(n)]
        C = [[tup() for _ in range(int(rng.integers(0, 6)))] for _ in range(n)]
        for syn in (False, True):
            kw = {"synonyms": (lambda w: _SYN.get(w, []))} if syn else {}
            out.append(case("spice", lambda R=R, C=C, kw=kw: es.spice(R, C, average=None, **kw), refs=R, cands=C, syn=syn))
            for key in ("object", "attribute", "relation"):
                out.append(case("spice." + key, lambda R=R, C=C, kw=kw, key=key: es.spice(R, C, **kw).params["by_type"][key], refs=R, cands=C, syn=syn))
    return out


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = argv or list(GROUPS)
    for name in names:
        cases = GROUPS[name](np.random.default_rng(abs(hash(name)) % 2**32 if False else len(name) * 7919))
        (OUT / f"{name}.json").write_text(json.dumps(cases, separators=(",", ":")))
        print(f"{name}: {len(cases)} cases")


if __name__ == "__main__":
    main(sys.argv[1:])
