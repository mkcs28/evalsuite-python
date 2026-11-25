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


def main(argv: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    names = argv or list(GROUPS)
    for name in names:
        cases = GROUPS[name](np.random.default_rng(abs(hash(name)) % 2**32 if False else len(name) * 7919))
        (OUT / f"{name}.json").write_text(json.dumps(cases, separators=(",", ":")))
        print(f"{name}: {len(cases)} cases")


if __name__ == "__main__":
    main(sys.argv[1:])
