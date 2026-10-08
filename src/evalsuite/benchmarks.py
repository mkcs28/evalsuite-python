"""Speed and memory benchmarks, against scikit-learn when it is installed.

Each case times the fastest of ``repeat`` runs (after one warm-up) and measures peak traced memory with
``tracemalloc`` (NumPy reports its allocations to it). Both libraries compute the same metrics on the same
data, and the largest absolute difference between their results is reported, so speed is never shown for
numbers that disagree.

    >>> from evalsuite.benchmarks import run_benchmarks
    >>> print(run_benchmarks(sizes=(10_000,), repeat=3))  # doctest: +SKIP
"""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

from .core.result import _json_safe, _latex_escape, _latex_table

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["BenchmarkResult", "run_benchmarks"]

_HEADER = (
    "case",
    "n",
    "evalsuite_ms",
    "sklearn_ms",
    "speedup",
    "evalsuite_peak_mb",
    "sklearn_peak_mb",
    "max_abs_diff",
)


def _measure(fn: Callable[[], Any], repeat: int) -> tuple[float, float, Any]:
    """(fastest seconds, peak MiB, result)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fn()  # warm-up
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        tracemalloc.start()
        try:
            fn()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    return best, peak / 2**20, result


def _cases(n: int, rng: np.random.Generator) -> list[tuple[str, Callable[[], Any], Optional[Callable[[], Any]]]]:
    import evalsuite as es

    y = rng.integers(0, 2, n)
    p = np.where(rng.random(n) < 0.8, y, 1 - y)
    prob = 1 / (1 + np.exp(-(2.0 * (y - 0.5) + rng.normal(0, 1, n))))
    yk = rng.integers(0, 10, n)
    pk = np.where(rng.random(n) < 0.7, yk, rng.integers(0, 10, n))
    yr = rng.normal(10, 3, n)
    pr = yr + rng.normal(0, 1, n)
    names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "mcc", "cohen_kappa"]

    def es_binary() -> list[float]:
        r = es.evaluate(y, p, metrics=names)
        return [float(r[m]) for m in names]

    def es_reg() -> list[float]:
        r = es.evaluate(yr, pr, metrics=["mae", "mse", "rmse", "r2"])
        return [float(r[m]) for m in ("mae", "mse", "rmse", "r2")]

    cases: list[tuple[str, Callable[[], Any], Optional[Callable[[], Any]]]] = [
        ("binary: 8 label metrics via evaluate()", es_binary, None),
        ("10 classes: macro F1", lambda: [float(es.f1(yk, pk, average="macro"))], None),
        ("binary: ROC AUC", lambda: [float(es.roc_auc(y, prob))], None),
        ("regression: MAE, MSE, RMSE, R² via evaluate()", es_reg, None),
    ]
    try:
        import sklearn.metrics as skm  # type: ignore[import-untyped]
    except ImportError:
        return cases

    def sk_binary() -> list[float]:
        return [
            skm.accuracy_score(y, p),
            skm.balanced_accuracy_score(y, p),
            skm.precision_score(y, p),
            skm.recall_score(y, p),
            skm.f1_score(y, p),
            skm.recall_score(y, p, pos_label=0),
            skm.matthews_corrcoef(y, p),
            skm.cohen_kappa_score(y, p),
        ]

    def sk_reg() -> list[float]:
        mse = skm.mean_squared_error(yr, pr)
        return [skm.mean_absolute_error(yr, pr), mse, float(np.sqrt(mse)), skm.r2_score(yr, pr)]

    sk = [sk_binary, lambda: [skm.f1_score(yk, pk, average="macro")], lambda: [skm.roc_auc_score(y, prob)], sk_reg]
    return [(name, es_fn, sk_fn) for (name, es_fn, _), sk_fn in zip(cases, sk)]


@dataclass(frozen=True, eq=False)
class BenchmarkResult:
    """Rows of timings (milliseconds), peak memory (MiB) and agreement, plus the environment they ran in."""

    rows: tuple[Any, ...]
    environment: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(r)) for r in self.rows))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))

    def _cells(self, digits: int) -> list[list[str]]:
        def f(v: Any, d: int = digits) -> str:
            return "–" if v is None else f"{v:.{d}f}"

        return [
            [
                r["case"],
                f"{r['n']:,}",
                f(r["evalsuite_ms"]),
                f(r["sklearn_ms"]),
                "–" if r["speedup"] is None else f"{r['speedup']:.2f}×",
                f(r["evalsuite_peak_mb"], 2),
                f(r["sklearn_peak_mb"], 2),
                "–" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.1e}",
            ]
            for r in self.rows
        ]

    _TITLES = (
        "Case",
        "n",
        "EvalSuite (ms)",
        "scikit-learn (ms)",
        "Speed-up",
        "EvalSuite peak (MiB)",
        "scikit-learn peak (MiB)",
        "Max |difference|",
    )

    def summary(self, *, digits: int = 3) -> str:
        cells = self._cells(digits)
        widths = [max(len(t), *(len(c[i]) for c in cells)) for i, t in enumerate(self._TITLES)]

        def line(c: Sequence[str]) -> str:
            return "  ".join(x.ljust(widths[i]) if i == 0 else x.rjust(widths[i]) for i, x in enumerate(c))

        env = self.environment
        head = (
            f"EvalSuite {env['evalsuite']} benchmarks | Python {env['python']} | NumPy {env['numpy']}"
            + (f" | scikit-learn {env['sklearn']}" if env.get("sklearn") else "")
            + f" | {env['machine']} | fastest of {env['repeat']} runs"
        )
        note = "Speed-up > 1 means EvalSuite is faster. Max |difference| compares the two libraries' results."
        return "\n".join([head, "", line(self._TITLES), *(line(c) for c in cells), "", note])

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            _json_safe({"environment": dict(self.environment), "rows": [dict(r) for r in self.rows]}),
        )

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_dataframe(self) -> pd.DataFrame:
        import pandas as pd

        frame: pd.DataFrame = pd.DataFrame([dict(r) for r in self.rows])
        return frame

    def to_csv(self, path: Optional[str] = None) -> str:
        from .core.export import csv_text

        rows = [[r[h] if r[h] is not None else float("nan") for h in _HEADER] for r in self.rows]
        text = csv_text(list(_HEADER), rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_markdown(self, *, digits: int = 3) -> str:
        lines = ["| " + " | ".join(self._TITLES) + " |", "| --- |" + " ---: |" * (len(self._TITLES) - 1)]
        return "\n".join(lines + ["| " + " | ".join(c) + " |" for c in self._cells(digits)])

    def to_latex(self, *, digits: int = 3, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        return _latex_table(
            [_latex_escape(t) for t in self._TITLES],
            [[_latex_escape(x) for x in c] for c in self._cells(digits)],
            caption=caption or "EvalSuite benchmarks.",
            label=label,
        )

    def to_html(self, *, digits: int = 3, full: bool = False) -> str:
        from .core.export import html_document, html_table

        table = html_table(list(self._TITLES), self._cells(digits), caption="Benchmarks")
        env = self.environment
        meta = f"Python {env['python']}, NumPy {env['numpy']}, {env['machine']}"
        return html_document("EvalSuite benchmarks", table, meta) if full else table


def run_benchmarks(
    sizes: Sequence[int] = (1_000, 100_000, 1_000_000),
    *,
    repeat: int = 5,
    compare_sklearn: bool = True,
    random_state: Optional[int] = 0,
) -> BenchmarkResult:
    """Time and memory for common evaluation workloads at each size, against scikit-learn if installed."""
    import evalsuite as es

    if repeat < 1:
        raise ValueError("repeat must be at least 1.")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    sk_version: Optional[str] = None
    if compare_sklearn:
        try:
            import sklearn

            sk_version = sklearn.__version__
        except ImportError:
            compare_sklearn = False
    for n in sizes:
        for name, es_fn, sk_fn in _cases(int(n), rng):
            es_t, es_mem, es_val = _measure(es_fn, repeat)
            row: dict[str, Any] = {
                "case": name,
                "n": int(n),
                "evalsuite_ms": es_t * 1000,
                "evalsuite_peak_mb": es_mem,
                "sklearn_ms": None,
                "sklearn_peak_mb": None,
                "speedup": None,
                "max_abs_diff": None,
            }
            if compare_sklearn and sk_fn is not None:
                sk_t, sk_mem, sk_val = _measure(sk_fn, repeat)
                row.update(
                    sklearn_ms=sk_t * 1000,
                    sklearn_peak_mb=sk_mem,
                    speedup=sk_t / es_t if es_t else None,
                    max_abs_diff=float(np.max(np.abs(np.asarray(es_val, float) - np.asarray(sk_val, float)))),
                )
            rows.append(row)
    env = {
        "evalsuite": es.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sk_version,
        "machine": f"{platform.system()} {platform.machine()}",
        "repeat": repeat,
    }
    return BenchmarkResult(tuple(rows), env)
