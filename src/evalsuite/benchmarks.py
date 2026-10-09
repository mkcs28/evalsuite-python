"""Speed and memory benchmarks against reference implementations.

Each case times the fastest of ``repeat`` runs (after one warm-up) and measures peak traced memory with
``tracemalloc`` (NumPy reports its allocations to it). EvalSuite and the reference compute the same
quantities on the same data, and the largest absolute difference between their results is reported, so
speed is never shown for numbers that disagree.

References: scikit-learn for classification and regression (``suite="core"``); scikit-learn, statsmodels
and SciPy for the v0.2.0 clinical, calibration and statistics functions (``suite="clinical"``);
scikit-learn, SciPy and pycocotools for segmentation and detection (``suite="vision"``). A case
whose reference library is not installed is timed for EvalSuite only.

    >>> from evalsuite.benchmarks import run_benchmarks
    >>> print(run_benchmarks(sizes=(10_000,), repeat=3))  # doctest: +SKIP
"""

from __future__ import annotations

import contextlib
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
    "reference",
    "evalsuite_ms",
    "reference_ms",
    "speedup",
    "evalsuite_peak_mb",
    "reference_peak_mb",
    "max_abs_diff",
)

Case = tuple[str, str, Callable[[], Any], Optional[Callable[[], Any]]]


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


def _core_cases(n: int, rng: np.random.Generator) -> list[Case]:
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

    cases: list[Case] = [
        ("binary: 8 label metrics via evaluate()", "scikit-learn", es_binary, None),
        ("10 classes: macro F1", "scikit-learn", lambda: [float(es.f1(yk, pk, average="macro"))], None),
        ("binary: ROC AUC", "scikit-learn", lambda: [float(es.roc_auc(y, prob))], None),
        ("regression: MAE, MSE, RMSE, R² via evaluate()", "scikit-learn", es_reg, None),
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
    return [(name, ref, es_fn, sk_fn) for (name, ref, es_fn, _), sk_fn in zip(cases, sk)]


def _vision_cases(n: int, rng: np.random.Generator) -> list[Case]:
    """v0.3.0: segmentation overlap and surface distance (n = pixels) and COCO detection (n / 1000 images)."""
    import evalsuite as es

    side = 64
    n_img = max(1, n // (side * side))
    k = 5
    yy, xx = np.ogrid[:side, :side]
    true = np.zeros((n_img, side, side), dtype=np.int64)
    for i in range(n_img):
        for c in range(1, k):
            cy, cx, r = rng.integers(8, side - 8), rng.integers(8, side - 8), rng.integers(4, 14)
            true[i][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = c
    pred = np.roll(true, 1, axis=2)
    noise = rng.random(pred.shape) < 0.02
    pred[noise] = rng.integers(0, k, int(noise.sum()))
    labels = list(range(k))

    def es_overlap() -> list[float]:
        d = np.asarray(es.dice(true, pred, average=None).value)
        j = np.asarray(es.iou(true, pred, average=None).value)
        return [*d, *j]

    n_hd = min(n_img, 50)

    def es_hd() -> list[float]:
        return [float(es.hausdorff_distance(true[i], pred[i], labels=[k - 1])) for i in range(n_hd)]

    n_det = max(10, n // 1000)
    y_true, y_pred = [], []
    for _ in range(n_det):
        m = int(rng.integers(1, 8))
        xy = rng.uniform(0, 500, (m, 2))
        wh = rng.uniform(8, 160, (m, 2))
        boxes = np.column_stack([xy, xy + wh])
        lab = rng.integers(1, 6, m)
        y_true.append({"boxes": boxes, "labels": lab, "iscrowd": np.zeros(m, int)})
        jitter = boxes + rng.normal(0, 4, boxes.shape)
        jitter[:, 2:] = np.maximum(jitter[:, 2:], jitter[:, :2] + 1)
        extra = rng.uniform(0, 500, (3, 2))
        fp = np.column_stack([extra, extra + rng.uniform(10, 80, (3, 2))])
        y_pred.append(
            {
                "boxes": np.vstack([jitter, fp]),
                "labels": np.r_[lab, rng.integers(1, 6, 3)],
                "scores": rng.random(m + 3),
            }
        )

    def es_map() -> list[float]:
        r = es.detection_report(y_true, y_pred)
        return [r["map"], r["map_50"], r["map_75"], r["mar_100"]]

    cases: list[Case] = [
        ("segmentation: Dice and IoU per class (n = pixels)", "scikit-learn", es_overlap, None),
        (f"segmentation: Hausdorff distance ({n_hd} image{'s' if n_hd != 1 else ''})", "SciPy", es_hd, None),
        (f"detection: COCO evaluation ({n_det} images)", "pycocotools", es_map, None),
    ]
    refs: dict[str, Callable[[], Any]] = {}
    try:
        import sklearn.metrics as skm

        def sk_overlap() -> list[float]:
            ft, fp_ = true.ravel(), pred.ravel()
            return [
                *skm.f1_score(ft, fp_, labels=labels, average=None),
                *skm.jaccard_score(ft, fp_, labels=labels, average=None),
            ]

        refs[cases[0][0]] = sk_overlap
    except ImportError:
        pass
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff

    def scipy_hd() -> list[float]:
        out = []
        for i in range(n_hd):
            pts = []
            for m_ in (true[i] == k - 1, pred[i] == k - 1):
                er = ndimage.binary_erosion(m_, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
                pts.append(np.argwhere(m_ & ~er).astype(float))
            out.append(max(directed_hausdorff(pts[0], pts[1])[0], directed_hausdorff(pts[1], pts[0])[0]))
        return out

    refs[cases[1][0]] = scipy_hd
    try:
        import contextlib as _ctx
        import io

        from pycocotools.coco import COCO  # type: ignore[import-untyped]
        from pycocotools.cocoeval import COCOeval  # type: ignore[import-untyped]

        def coco_map() -> list[float]:
            images, anns, dets, aid = [], [], [], 1
            for i, (t, p) in enumerate(zip(y_true, y_pred)):
                images.append({"id": i + 1})
                for b, c in zip(t["boxes"], t["labels"]):
                    w, h = b[2] - b[0], b[3] - b[1]
                    anns.append(
                        {
                            "id": aid,
                            "image_id": i + 1,
                            "category_id": int(c),
                            "bbox": [b[0], b[1], w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                    aid += 1
                for b, c, sc in zip(p["boxes"], p["labels"], p["scores"]):
                    dets.append(
                        {
                            "image_id": i + 1,
                            "category_id": int(c),
                            "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                            "score": float(sc),
                        }
                    )
            with _ctx.redirect_stdout(io.StringIO()):
                gt = COCO()
                gt.dataset = {
                    "images": images,
                    "annotations": anns,
                    "categories": [{"id": c} for c in range(1, 6)],
                }
                gt.createIndex()
                ev = COCOeval(gt, gt.loadRes(dets), "bbox")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            return [ev.stats[0], ev.stats[1], ev.stats[2], ev.stats[8]]

        refs[cases[2][0]] = coco_map
    except ImportError:
        pass
    return [(name, ref, es_fn, refs.get(name)) for name, ref, es_fn, _ in cases]


def _clinical_cases(n: int, rng: np.random.Generator) -> list[Case]:
    """v0.2.0: diagnostic accuracy, calibration, decision curves and statistical tests."""
    import evalsuite as es

    y = rng.integers(0, 2, n)
    p = np.where(rng.random(n) < 0.8, y, 1 - y)
    x = rng.normal(size=n)
    yc = (rng.random(n) < 1 / (1 + np.exp(-(0.4 + 1.3 * x)))).astype(int)
    risk = 1 / (1 + np.exp(-(0.1 + 2.0 * x)))
    a, b = rng.normal(0, 1, n), rng.normal(0.05, 1.2, n)
    pvals = rng.random(n) ** 2
    ga, gb = rng.integers(0, 5, n), rng.integers(0, 5, n)
    table = np.zeros((5, 5), dtype=np.int64)
    np.add.at(table, (ga, gb), 1)
    thresholds = np.arange(1, 100) / 100

    def es_diag() -> list[float]:
        return [
            float(es.sensitivity(y, p)),
            float(es.specificity(y, p)),
            float(es.lr_positive(y, p)),
            float(es.lr_negative(y, p)),
        ]

    report_keys = ("sensitivity", "specificity", "ppv", "npv", "accuracy", "prevalence", "diagnostic_odds_ratio")

    def es_report() -> list[float]:
        r = es.diagnostic_report(y, p)
        return [v for k in report_keys for v in (r[k].low, r[k].high)]

    def es_cal() -> list[float]:
        return [float(es.calibration_slope(yc, risk)), float(es.calibration_intercept(yc, risk))]

    def es_dca() -> list[float]:
        curve: list[float] = es.decision_curve(yc, risk, thresholds=thresholds).net_benefit["model"].tolist()
        return curve

    def numpy_dca() -> list[float]:  # the textbook loop, one threshold at a time
        out = []
        for t in thresholds:
            treat = risk >= t
            tp = np.sum(treat & (yc == 1))
            fp = np.sum(treat & (yc == 0))
            out.append(tp / n - fp / n * t / (1 - t))
        return out

    cases: list[Case] = [
        ("clinical: sensitivity, specificity, LR+, LR−", "scikit-learn", es_diag, None),
        ("clinical: diagnostic report (7 CIs)", "statsmodels", es_report, None),
        ("calibration: slope and intercept", "statsmodels", es_cal, None),
        ("decision curve: 99 thresholds", "NumPy loop", es_dca, numpy_dca),
        ("statistics: Welch t-test", "SciPy", lambda: [es.t_test(a, b).p_value], None),
        ("statistics: Mann–Whitney U", "SciPy", lambda: [es.mann_whitney_test(a, b).p_value], None),
        ("statistics: Cramér's V (5×5 table)", "SciPy", lambda: [es.cramers_v(table)], None),
        (
            "multiple testing: Hochberg (n p-values)",
            "statsmodels",
            lambda: es.adjust_pvalues(pvals, method="hochberg").tolist(),
            None,
        ),
    ]
    refs: dict[str, Callable[[], Any]] = {}
    try:
        import sklearn.metrics as skm

        def sk_diag() -> list[float]:
            lr_pos, lr_neg = skm.class_likelihood_ratios(y, p)
            return [skm.recall_score(y, p), skm.recall_score(y, p, pos_label=0), lr_pos, lr_neg]

        refs["clinical: sensitivity, specificity, LR+, LR−"] = sk_diag
    except (ImportError, AttributeError):
        pass
    from scipy import stats

    refs["statistics: Welch t-test"] = lambda: [stats.ttest_ind(a, b, equal_var=False).pvalue]
    refs["statistics: Mann–Whitney U"] = lambda: [stats.mannwhitneyu(a, b).pvalue]
    refs["statistics: Cramér's V (5×5 table)"] = lambda: [stats.contingency.association(table, method="cramer")]
    try:
        import statsmodels.api as sm  # type: ignore[import-untyped]
        from statsmodels.stats.contingency_tables import Table2x2  # type: ignore[import-untyped]
        from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]
        from statsmodels.stats.proportion import proportion_confint  # type: ignore[import-untyped]

        def sm_report() -> list[float]:
            tp = int(np.sum((y == 1) & (p == 1)))
            fn = int(np.sum((y == 1) & (p == 0)))
            fp = int(np.sum((y == 0) & (p == 1)))
            tn = int(np.sum((y == 0) & (p == 0)))
            out: list[float] = []
            for k, m in ((tp, tp + fn), (tn, tn + fp), (tp, tp + fp), (tn, tn + fn), (tp + tn, n), (tp + fn, n)):
                out.extend(proportion_confint(k, m, method="wilson"))
            out.extend(Table2x2(np.array([[tp, fn], [fp, tn]])).oddsratio_confint())
            return out

        def sm_cal() -> list[float]:
            lp = np.log(risk / (1 - risk))
            fam = sm.families.Binomial()
            slope = sm.GLM(yc, sm.add_constant(lp), family=fam).fit().params[1]
            intercept = sm.GLM(yc, np.ones((n, 1)), family=fam, offset=lp).fit().params[0]
            return [slope, intercept]

        refs["clinical: diagnostic report (7 CIs)"] = sm_report
        refs["calibration: slope and intercept"] = sm_cal
        refs["multiple testing: Hochberg (n p-values)"] = lambda: multipletests(pvals, method="simes-hochberg")[
            1
        ].tolist()
    except ImportError:
        pass
    return [(name, ref, es_fn, own if own is not None else refs.get(name)) for name, ref, es_fn, own in cases]


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
                r.get("reference") or "–",
                f(r["evalsuite_ms"]),
                f(r.get("reference_ms")),
                "–" if r["speedup"] is None else f"{r['speedup']:.2f}×",
                f(r["evalsuite_peak_mb"], 2),
                f(r.get("reference_peak_mb"), 2),
                "–" if r["max_abs_diff"] is None else f"{r['max_abs_diff']:.1e}",
            ]
            for r in self.rows
        ]

    _TITLES = (
        "Case",
        "n",
        "Reference",
        "EvalSuite (ms)",
        "Reference (ms)",
        "Speed-up",
        "EvalSuite peak (MiB)",
        "Reference peak (MiB)",
        "Max |difference|",
    )

    def summary(self, *, digits: int = 3) -> str:
        cells = self._cells(digits)
        widths = [max(len(t), *(len(c[i]) for c in cells)) for i, t in enumerate(self._TITLES)]

        def line(c: Sequence[str]) -> str:
            return "  ".join(x.ljust(widths[i]) if i in (0, 2) else x.rjust(widths[i]) for i, x in enumerate(c))

        env = self.environment
        head = (
            f"EvalSuite {env['evalsuite']} benchmarks | Python {env['python']} | NumPy {env['numpy']}"
            + (f" | scikit-learn {env['sklearn']}" if env.get("sklearn") else "")
            + (f" | statsmodels {env['statsmodels']}" if env.get("statsmodels") else "")
            + (f" | SciPy {env['scipy']}" if env.get("scipy") else "")
            + (f" | pycocotools {env['pycocotools']}" if env.get("pycocotools") else "")
            + f" | {env['machine']} | fastest of {env['repeat']} runs"
        )
        note = "Speed-up > 1 means EvalSuite is faster. Max |difference| compares EvalSuite with the reference."
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

        rows = [[r.get(h) if r.get(h) is not None else float("nan") for h in _HEADER] for r in self.rows]
        text = csv_text(list(_HEADER), rows)
        if path is not None:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
        return text

    def to_markdown(self, *, digits: int = 3) -> str:
        lines = [
            "| " + " | ".join(self._TITLES) + " |",
            "| --- | ---: | --- |" + " ---: |" * (len(self._TITLES) - 3),
        ]
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
    suite: str = "all",
) -> BenchmarkResult:
    """Time and memory for evaluation workloads at each size, against a reference implementation.

    ``suite``: ``"core"`` (classification and regression vs scikit-learn), ``"clinical"`` (v0.2.0 clinical,
    calibration and statistics vs scikit-learn, statsmodels, SciPy), ``"vision"`` (segmentation and COCO
    detection vs scikit-learn, SciPy, pycocotools) or ``"all"`` (default).
    ``compare_sklearn=False`` times EvalSuite alone. Rows keep ``sklearn_ms``/``sklearn_peak_mb`` for rows
    whose reference is scikit-learn, for compatibility with 0.1.x.
    """
    import scipy

    import evalsuite as es

    if repeat < 1:
        raise ValueError("repeat must be at least 1.")
    if suite not in ("all", "core", "clinical", "vision"):
        raise ValueError("suite must be 'all', 'core', 'clinical' or 'vision'.")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    versions: dict[str, Optional[str]] = {"sklearn": None, "statsmodels": None, "pycocotools": None}
    if compare_sklearn:
        for mod in versions:
            with contextlib.suppress(ImportError):
                __import__(mod)
                from importlib.metadata import version as _dist_version

                versions[mod] = _dist_version("scikit-learn" if mod == "sklearn" else mod)
    builders = {
        "core": [_core_cases],
        "clinical": [_clinical_cases],
        "vision": [_vision_cases],
        "all": [_core_cases, _clinical_cases, _vision_cases],
    }[suite]
    for n in sizes:
        for build in builders:
            for name, ref_name, es_fn, ref_fn in build(int(n), rng):
                es_t, es_mem, es_val = _measure(es_fn, repeat)
                row: dict[str, Any] = {
                    "case": name,
                    "n": int(n),
                    "reference": None,
                    "evalsuite_ms": es_t * 1000,
                    "evalsuite_peak_mb": es_mem,
                    "reference_ms": None,
                    "reference_peak_mb": None,
                    "sklearn_ms": None,
                    "sklearn_peak_mb": None,
                    "speedup": None,
                    "max_abs_diff": None,
                }
                if compare_sklearn and ref_fn is not None:
                    ref_t, ref_mem, ref_val = _measure(ref_fn, repeat)
                    row.update(
                        reference=ref_name,
                        reference_ms=ref_t * 1000,
                        reference_peak_mb=ref_mem,
                        speedup=ref_t / es_t if es_t else None,
                        max_abs_diff=float(np.max(np.abs(np.asarray(es_val, float) - np.asarray(ref_val, float)))),
                    )
                    if ref_name == "scikit-learn":
                        row.update(sklearn_ms=row["reference_ms"], sklearn_peak_mb=ref_mem)
                rows.append(row)
    env = {
        "evalsuite": es.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__ if suite != "core" else None,
        "sklearn": versions["sklearn"],
        "statsmodels": versions["statsmodels"] if suite != "core" else None,
        "pycocotools": versions["pycocotools"] if suite in ("all", "vision") else None,
        "machine": f"{platform.system()} {platform.machine()}",
        "repeat": repeat,
        "suite": suite,
    }
    return BenchmarkResult(tuple(rows), env)
