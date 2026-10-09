"""Per-metric benchmark cases: one row for every registered metric and every statistics function.

Used by :func:`evalsuite.benchmarks.run_benchmarks` with ``suite="metrics"``. Each case is named after the
metric's registry id (``classification.f1``, ``text.bleu``, ``statistics.t_test``) and times one call on
data of the requested size. Where an independent implementation exists it is timed on the same data and the
largest absolute difference is reported:

* a reference library (scikit-learn, SciPy, statsmodels, pycocotools, sacreBLEU, rouge-score, NLTK,
  pycocoevalcap, ranx, krippendorff, choix, POT, jsonschema) when it is installed;
* otherwise, for metrics with a closed-form definition, a direct NumPy / standard-library implementation of
  the textbook formula (labelled ``NumPy formula`` or ``Python stdlib``).

Metrics with neither (learned or judge-dependent scores, randomised procedures) are timed for EvalSuite only.

Sizes: ``n`` observations for classification, regression, clinical and statistics; ``n`` pixels for
segmentation; ``n / 1000`` images for detection; ``n / 100`` examples for text, retrieval, RAG, judge and
structured-output metrics; ``n / 1000`` examples for the token-embedding metrics (BERTScore, MoverScore)
and MAUVE.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

Case = tuple[str, str, Callable[[], Any], Optional[Callable[[], Any]]]

_VOCAB = (  # noqa: SIM905
    "the a cat dog sat on mat quickly model models data run running ran results show improves baseline "
    "evaluation paper method score test train large small new old good bad"
).split()


def _opt(module: str) -> Any:
    """An optional reference module, or None when it is not installed (fixed names only)."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def _f(x: Any) -> float:
    return float(x)


def _numeric(x: Any) -> list[float]:
    """Flatten a result (MetricResult, array, scalar, test result) to a list of floats for comparison."""
    if hasattr(x, "p_value"):
        return [float(x.statistic), float(x.p_value)]
    if hasattr(x, "low") and hasattr(x, "high"):
        return [float(x.low), float(x.high)]
    v = getattr(x, "value", x)
    return [float(t) for t in np.ravel(np.asarray(v, dtype=float))]


class _Cases:
    """Collects (name, reference label, EvalSuite call, reference call)."""

    def __init__(self) -> None:
        self.rows: list[Case] = []

    def add(
        self,
        name: str,
        es_fn: Callable[[], Any],
        ref: str = "",
        ref_fn: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.rows.append((name, ref if ref_fn is not None else "", es_fn, ref_fn))


# --------------------------------------------------------------------------------------------- core
def _classification(c: _Cases, n: int, rng: np.random.Generator) -> None:
    import evalsuite as es

    y = rng.integers(0, 2, n)
    p = np.where(rng.random(n) < 0.8, y, 1 - y)
    prob = 1 / (1 + np.exp(-(2.0 * (y - 0.5) + rng.normal(0, 1, n))))
    yk = rng.integers(0, 10, n)
    logits = rng.normal(size=(n, 10))
    logits[np.arange(n), yk] += 1.5
    pk = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    Y = (rng.random((n, 5)) < 0.3).astype(int)
    P = np.where(rng.random((n, 5)) < 0.85, Y, 1 - Y)

    def ece_np() -> float:
        bins = np.minimum((prob * 10).astype(int), 9)
        total = 0.0
        for b in range(10):
            m = bins == b
            if m.any():
                total += m.mean() * abs(prob[m].mean() - y[m].mean())
        return total

    skm = _opt("sklearn.metrics")

    def sk(fn: Callable[[], Any]) -> Optional[Callable[[], Any]]:
        return fn if skm is not None else None

    c.add(
        "classification.accuracy",
        lambda: _f(es.accuracy(y, p)),
        "scikit-learn",
        sk(lambda: skm.accuracy_score(y, p)),
    )
    c.add(
        "classification.average_precision",
        lambda: _f(es.average_precision(y, prob)),
        "scikit-learn",
        sk(lambda: skm.average_precision_score(y, prob)),
    )
    c.add(
        "classification.balanced_accuracy",
        lambda: _f(es.balanced_accuracy(yk, pk.argmax(1))),
        "scikit-learn",
        sk(lambda: skm.balanced_accuracy_score(yk, pk.argmax(1))),
    )
    c.add(
        "classification.brier_score",
        lambda: _f(es.brier_score(y, prob)),
        "scikit-learn",
        sk(lambda: skm.brier_score_loss(y, prob)),
    )
    c.add(
        "classification.cohen_kappa",
        lambda: _f(es.cohen_kappa(y, p)),
        "scikit-learn",
        sk(lambda: skm.cohen_kappa_score(y, p)),
    )
    c.add(
        "classification.expected_calibration_error",
        lambda: _f(es.expected_calibration_error(y, prob, n_bins=10)),
        "NumPy formula",
        ece_np,
    )
    c.add(
        "classification.f1",
        lambda: _f(es.f1(yk, pk.argmax(1), average="macro")),
        "scikit-learn",
        sk(lambda: skm.f1_score(yk, pk.argmax(1), average="macro")),
    )
    c.add(
        "classification.fbeta",
        lambda: _f(es.fbeta(y, p, beta=2)),
        "scikit-learn",
        sk(lambda: skm.fbeta_score(y, p, beta=2)),
    )
    c.add(
        "classification.hamming_loss",
        lambda: _f(es.hamming_loss(Y, P)),
        "scikit-learn",
        sk(lambda: skm.hamming_loss(Y, P)),
    )
    c.add(
        "classification.jaccard",
        lambda: _f(es.jaccard(y, p)),
        "scikit-learn",
        sk(lambda: skm.jaccard_score(y, p)),
    )
    c.add(
        "classification.log_loss",
        lambda: _f(es.log_loss(yk, pk)),
        "scikit-learn",
        sk(lambda: skm.log_loss(yk, pk)),
    )
    c.add(
        "classification.mcc",
        lambda: _f(es.mcc(yk, pk.argmax(1))),
        "scikit-learn",
        sk(lambda: skm.matthews_corrcoef(yk, pk.argmax(1))),
    )
    c.add(
        "classification.npv",
        lambda: _f(es.npv(y, p)),
        "scikit-learn",
        sk(lambda: skm.precision_score(y, p, pos_label=0)),
    )
    c.add(
        "classification.precision",
        lambda: _f(es.precision(yk, pk.argmax(1), average="weighted")),
        "scikit-learn",
        sk(lambda: skm.precision_score(yk, pk.argmax(1), average="weighted")),
    )
    c.add(
        "classification.recall",
        lambda: _f(es.recall(yk, pk.argmax(1), average="micro")),
        "scikit-learn",
        sk(lambda: skm.recall_score(yk, pk.argmax(1), average="micro")),
    )
    c.add(
        "classification.roc_auc",
        lambda: _f(es.roc_auc(yk, pk, multi_class="ovr")),
        "scikit-learn",
        sk(lambda: skm.roc_auc_score(yk, pk, multi_class="ovr")),
    )
    c.add(
        "classification.specificity",
        lambda: _f(es.specificity(y, p)),
        "scikit-learn",
        sk(lambda: skm.recall_score(y, p, pos_label=0)),
    )
    c.add(
        "classification.top_k_accuracy",
        lambda: _f(es.top_k_accuracy(yk, pk, k=3)),
        "scikit-learn",
        sk(lambda: skm.top_k_accuracy_score(yk, pk, k=3)),
    )


def _regression(c: _Cases, n: int, rng: np.random.Generator) -> None:
    import evalsuite as es

    yt = rng.gamma(3.0, 3.0, n) + 0.5
    yp = np.abs(yt + rng.normal(0, 1, n)) + 0.1
    e = yt - yp
    skm = _opt("sklearn.metrics")

    def sk(fn: Callable[[], Any]) -> Optional[Callable[[], Any]]:
        return fn if skm is not None else None

    def huber_np() -> float:
        a = np.abs(e)
        return float(np.mean(np.where(a <= 1.0, 0.5 * e**2, a - 0.5)))

    def r2_np() -> float:
        return float(1 - np.sum(e**2) / np.sum((yt - yt.mean()) ** 2))

    c.add(
        "regression.adjusted_r2",
        lambda: _f(es.adjusted_r2(yt, yp, n_features=5)),
        "NumPy formula",
        lambda: 1 - (1 - r2_np()) * (n - 1) / (n - 5 - 1),
    )
    c.add(
        "regression.explained_variance",
        lambda: _f(es.explained_variance(yt, yp)),
        "scikit-learn",
        sk(lambda: skm.explained_variance_score(yt, yp)),
    )
    c.add("regression.huber_loss", lambda: _f(es.huber_loss(yt, yp)), "NumPy formula", huber_np)
    c.add(
        "regression.mae", lambda: _f(es.mae(yt, yp)), "scikit-learn", sk(lambda: skm.mean_absolute_error(yt, yp))
    )
    c.add(
        "regression.mape",
        lambda: _f(es.mape(yt, yp)),
        "scikit-learn",
        sk(lambda: skm.mean_absolute_percentage_error(yt, yp)),
    )
    c.add(
        "regression.max_error", lambda: _f(es.max_error(yt, yp)), "scikit-learn", sk(lambda: skm.max_error(yt, yp))
    )
    c.add(
        "regression.mean_bias_error",
        lambda: _f(es.mean_bias_error(yt, yp)),
        "NumPy formula",
        lambda: float(np.mean(yp - yt)),
    )
    c.add(
        "regression.median_absolute_error",
        lambda: _f(es.median_absolute_error(yt, yp)),
        "scikit-learn",
        sk(lambda: skm.median_absolute_error(yt, yp)),
    )
    c.add("regression.mse", lambda: _f(es.mse(yt, yp)), "scikit-learn", sk(lambda: skm.mean_squared_error(yt, yp)))
    c.add(
        "regression.msle",
        lambda: _f(es.msle(yt, yp)),
        "scikit-learn",
        sk(lambda: skm.mean_squared_log_error(yt, yp)),
    )
    c.add(
        "regression.quantile_loss",
        lambda: _f(es.quantile_loss(yt, yp, alpha=0.9)),
        "scikit-learn",
        sk(lambda: skm.mean_pinball_loss(yt, yp, alpha=0.9)),
    )
    c.add("regression.r2", lambda: _f(es.r2(yt, yp)), "scikit-learn", sk(lambda: skm.r2_score(yt, yp)))
    c.add(
        "regression.rae",
        lambda: _f(es.rae(yt, yp)),
        "NumPy formula",
        lambda: float(np.sum(np.abs(e)) / np.sum(np.abs(yt - yt.mean()))),
    )
    c.add(
        "regression.rmse",
        lambda: _f(es.rmse(yt, yp)),
        "NumPy formula",
        lambda: float(np.sqrt(np.mean(e**2))),
    )
    c.add(
        "regression.rmsle",
        lambda: _f(es.rmsle(yt, yp)),
        "NumPy formula",
        lambda: float(np.sqrt(np.mean((np.log1p(yt) - np.log1p(yp)) ** 2))),
    )
    c.add(
        "regression.rse",
        lambda: _f(es.rse(yt, yp)),
        "NumPy formula",
        lambda: float(np.sum(e**2) / np.sum((yt - yt.mean()) ** 2)),
    )
    c.add(
        "regression.smape",
        lambda: _f(es.smape(yt, yp)),
        "NumPy formula",
        lambda: float(np.mean(2 * np.abs(e) / (np.abs(yt) + np.abs(yp)))),
    )


# --------------------------------------------------------------------------------------------- clinical
def _clinical(c: _Cases, n: int, rng: np.random.Generator) -> None:
    import evalsuite as es

    y = rng.integers(0, 2, n)
    p = np.where(rng.random(n) < 0.8, y, 1 - y)
    x = rng.normal(size=n)
    yc = (rng.random(n) < 1 / (1 + np.exp(-(0.4 + 1.3 * x)))).astype(int)
    risk = 1 / (1 + np.exp(-(0.1 + 2.0 * x)))
    tp = int(np.sum((y == 1) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    sens, spec = tp / (tp + fn), tn / (tn + fp)

    def counts() -> tuple[int, int, int, int]:
        return (
            int(np.sum((y == 1) & (p == 1))),
            int(np.sum((y == 1) & (p == 0))),
            int(np.sum((y == 0) & (p == 1))),
            int(np.sum((y == 0) & (p == 0))),
        )

    def mce_np() -> float:
        bins = np.minimum((risk * 10).astype(int), 9)
        return float(
            max(abs(risk[bins == b].mean() - yc[bins == b].mean()) for b in range(10) if (bins == b).any())
        )

    def dor_np() -> float:
        t, f_n, f_p, t_n = counts()
        return (t * t_n) / (f_p * f_n)

    def youden_np() -> float:
        t, f_n, f_p, t_n = counts()
        return t / (t + f_n) + t_n / (t_n + f_p) - 1

    skm = _opt("sklearn.metrics")

    def lrs() -> tuple[float, float]:
        if skm is not None:
            a, b = skm.class_likelihood_ratios(y, p)
            return float(a), float(b)
        t, f_n, f_p, t_n = counts()
        se, sp = t / (t + f_n), t_n / (t_n + f_p)
        return se / (1 - sp), (1 - se) / sp

    sm = _opt("statsmodels.api")

    def cal_slope() -> float:
        lp = np.log(risk / (1 - risk))
        return float(sm.GLM(yc, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])

    def cal_int() -> float:
        lp = np.log(risk / (1 - risk))
        return float(sm.GLM(yc, np.ones((n, 1)), family=sm.families.Binomial(), offset=lp).fit().params[0])

    c.add(
        "calibration.calibration_intercept",
        lambda: _f(es.calibration_intercept(yc, risk)),
        "statsmodels",
        cal_int if sm is not None else None,
    )
    c.add(
        "calibration.calibration_slope",
        lambda: _f(es.calibration_slope(yc, risk)),
        "statsmodels",
        cal_slope if sm is not None else None,
    )
    c.add(
        "calibration.maximum_calibration_error",
        lambda: _f(es.maximum_calibration_error(yc, risk, n_bins=10)),
        "NumPy formula",
        mce_np,
    )
    c.add(
        "clinical.diagnostic_odds_ratio",
        lambda: _f(es.diagnostic_odds_ratio(y, p)),
        "NumPy formula",
        dor_np,
    )
    c.add(
        "clinical.lr_negative",
        lambda: _f(es.lr_negative(y, p)),
        "scikit-learn" if skm else "NumPy formula",
        lambda: lrs()[1],
    )
    c.add(
        "clinical.lr_positive",
        lambda: _f(es.lr_positive(y, p)),
        "scikit-learn" if skm else "NumPy formula",
        lambda: lrs()[0],
    )
    c.add(
        "clinical.net_benefit",
        lambda: _f(es.net_benefit(yc, risk, threshold=0.2)),
        "NumPy formula",
        lambda: float(np.sum((risk >= 0.2) & (yc == 1)) / n - np.sum((risk >= 0.2) & (yc == 0)) / n * 0.25),
    )
    c.add(
        "clinical.ppv",
        lambda: _f(es.ppv(y, p)),
        "scikit-learn",
        (lambda: skm.precision_score(y, p)) if skm is not None else None,
    )
    c.add(
        "clinical.sensitivity",
        lambda: _f(es.sensitivity(y, p)),
        "scikit-learn",
        (lambda: skm.recall_score(y, p)) if skm is not None else None,
    )
    c.add(
        "clinical.youden_j",
        lambda: _f(es.youden_j(y, p)),
        "NumPy formula",
        youden_np,
    )
    del sens, spec, tp, fn, fp, tn


# --------------------------------------------------------------------------------------------- statistics
def _statistics(c: _Cases, n: int, rng: np.random.Generator) -> None:
    from scipy import stats

    import evalsuite as es

    a, b = rng.normal(0, 1, n), rng.normal(0.05, 1.2, n)
    d = a + rng.normal(0.02, 0.5, n)
    g3 = [rng.normal(mu, 1, n) for mu in (0.0, 0.05, 0.1)]
    y = rng.integers(0, 2, n)
    pa = np.where(rng.random(n) < 0.8, y, 1 - y)
    pb = np.where(rng.random(n) < 0.78, y, 1 - y)
    prob_a = 1 / (1 + np.exp(-(2.0 * (y - 0.5) + rng.normal(0, 1, n))))
    prob_b = 1 / (1 + np.exp(-(1.6 * (y - 0.5) + rng.normal(0, 1, n))))
    ga, gb = rng.integers(0, 5, n), rng.integers(0, 5, n)
    table = np.zeros((5, 5), dtype=np.int64)
    np.add.at(table, (ga, gb), 1)
    t2 = np.array([[int(np.sum((y == 1) & (pa == 1))), int(np.sum((y == 1) & (pa == 0)))],
                   [int(np.sum((y == 0) & (pa == 1))), int(np.sum((y == 0) & (pa == 0)))]])  # fmt: skip
    pvals = rng.random(n) ** 2
    xs = rng.normal(size=min(n, 5000))
    k = int(np.sum(y == pa))
    n_boot = 200

    sm_prop = _opt("statsmodels.stats.proportion")
    sm_mt = _opt("statsmodels.stats.multitest")
    sm_ct = _opt("statsmodels.stats.contingency_tables")

    def cohens_d_np() -> float:
        sp = math.sqrt(((n - 1) * a.var(ddof=1) + (n - 1) * b.var(ddof=1)) / (2 * n - 2))
        return float((a.mean() - b.mean()) / sp)

    c.add(
        "statistics.accuracy_ci",
        lambda: _numeric(es.accuracy_ci(y, pa)),
        "statsmodels",
        (lambda: list(sm_prop.proportion_confint(k, n, method="wilson"))) if sm_prop is not None else None,
    )
    c.add(
        "statistics.adjust_pvalues",
        lambda: es.adjust_pvalues(pvals, method="bh").tolist(),
        "statsmodels",
        (lambda: sm_mt.multipletests(pvals, method="fdr_bh")[1].tolist()) if sm_mt is not None else None,
    )
    c.add(
        "statistics.bootstrap_ci",
        lambda: _numeric(es.bootstrap_ci(es.accuracy, y, pa, n_resamples=n_boot, random_state=0)),
    )
    c.add(
        "statistics.chi_square_test",
        lambda: _numeric(es.chi_square_test(table)),
        "SciPy",
        lambda: list(stats.chi2_contingency(table)[:2]),
    )
    c.add(
        "statistics.cliffs_delta",
        lambda: _f(es.cliffs_delta(a, b)),
        "SciPy",
        lambda: 2 * stats.mannwhitneyu(a, b).statistic / (n * n) - 1,
    )
    c.add("statistics.cohens_d", lambda: _f(es.cohens_d(a, b)), "NumPy formula", cohens_d_np)
    c.add(
        "statistics.compare",
        lambda: es.compare(y, {"a": pa, "b": pb}, metrics=["accuracy"], n_resamples=n_boot, random_state=0),
    )
    c.add(
        "statistics.cramers_v",
        lambda: _f(es.cramers_v(table)),
        "SciPy",
        lambda: stats.contingency.association(table, method="cramer"),
    )
    c.add(
        "statistics.delong_test",
        lambda: [float(es.delong_test(y, prob_a, prob_b).p_value)],
    )
    c.add(
        "statistics.fisher_exact_test",
        lambda: [float(es.fisher_exact_test(t2).p_value)],
        "SciPy",
        lambda: [stats.fisher_exact(t2).pvalue],
    )
    c.add(
        "statistics.friedman_test",
        lambda: _numeric(es.friedman_test(*g3)),
        "SciPy",
        lambda: list(stats.friedmanchisquare(*g3)),
    )
    c.add(
        "statistics.hedges_g",
        lambda: _f(es.hedges_g(a, b)),
        "NumPy formula",
        lambda: cohens_d_np() * (1 - 3 / (4 * (2 * n) - 9)),
    )
    c.add(
        "statistics.kruskal_wallis_test",
        lambda: _numeric(es.kruskal_wallis_test(*g3)),
        "SciPy",
        lambda: list(stats.kruskal(*g3)),
    )
    c.add(
        "statistics.mann_whitney_test",
        lambda: _numeric(es.mann_whitney_test(a, b)),
        "SciPy",
        lambda: list(stats.mannwhitneyu(a, b)),
    )

    def mcnemar_ref() -> list[float]:
        b01 = int(np.sum((pa == y) & (pb != y)))
        b10 = int(np.sum((pa != y) & (pb == y)))
        r = sm_ct.mcnemar(np.array([[0, b01], [b10, 0]]), exact=False, correction=True)
        return [float(r.statistic), float(r.pvalue)]

    c.add(
        "statistics.mcnemar_test",
        lambda: _numeric(es.mcnemar_test(y, pa, pb, exact=False)),
        "statsmodels",
        mcnemar_ref if sm_ct is not None else None,
    )
    c.add(
        "statistics.paired_bootstrap_test",
        lambda: [
            float(es.paired_bootstrap_test(es.accuracy, y, pa, pb, n_resamples=n_boot, random_state=0).p_value)
        ],
    )
    c.add(
        "statistics.paired_t_test",
        lambda: _numeric(es.paired_t_test(a, d)),
        "SciPy",
        lambda: list(stats.ttest_rel(a, d)),
    )
    c.add(
        "statistics.proportion_ci",
        lambda: _numeric(es.proportion_ci(k, n, method="clopper-pearson")),
        "statsmodels",
        (lambda: list(sm_prop.proportion_confint(k, n, method="beta"))) if sm_prop is not None else None,
    )
    c.add("statistics.roc_auc_ci", lambda: _numeric(es.roc_auc_ci(y, prob_a)))
    c.add(
        "statistics.shapiro_wilk_test",
        lambda: _numeric(es.shapiro_wilk_test(xs)),
        "SciPy",
        lambda: list(stats.shapiro(xs)),
    )
    c.add(
        "statistics.t_test",
        lambda: _numeric(es.t_test(a, b)),
        "SciPy",
        lambda: list(stats.ttest_ind(a, b, equal_var=False)),
    )
    c.add(
        "statistics.wilcoxon_test",
        lambda: _numeric(es.wilcoxon_test(a, d)),
        "SciPy",
        lambda: list(stats.wilcoxon(a, d)),
    )


# --------------------------------------------------------------------------------------------- vision
def _vision(c: _Cases, n: int, rng: np.random.Generator) -> None:
    import evalsuite as es

    side = 64
    n_img = max(1, n // (side * side))
    k = 5
    yy, xx = np.ogrid[:side, :side]
    true = np.zeros((n_img, side, side), dtype=np.int64)
    for i in range(n_img):
        for cl in range(1, k):
            cy, cx, r = rng.integers(8, side - 8), rng.integers(8, side - 8), rng.integers(4, 14)
            true[i][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = cl
    pred = np.roll(true, 1, axis=2)
    noise = rng.random(pred.shape) < 0.02
    pred[noise] = rng.integers(0, k, int(noise.sum()))
    labels = list(range(k))
    n_sd = min(n_img, 50)
    bt, bp = true[:n_sd] == k - 1, pred[:n_sd] == k - 1
    ft, fp_ = true.ravel(), pred.ravel()

    skm = _opt("sklearn.metrics")
    from scipy import ndimage
    from scipy.spatial.distance import directed_hausdorff

    def surface(m: Any) -> Any:
        er = ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(2, 1), border_value=0)
        return np.argwhere(m & ~er).astype(float)

    def hd_ref() -> list[float]:
        out = []
        for i in range(n_sd):
            s1, s2 = surface(bt[i]), surface(bp[i])
            out.append(max(directed_hausdorff(s1, s2)[0], directed_hausdorff(s2, s1)[0]))
        return out

    def asd_ref() -> list[float]:
        from scipy.spatial import cKDTree

        out = []
        for i in range(n_sd):
            s1, s2 = surface(bt[i]), surface(bp[i])
            d1, _ = cKDTree(s2).query(s1)
            d2, _ = cKDTree(s1).query(s2)
            out.append(float((d1.sum() + d2.sum()) / (len(d1) + len(d2))))
        return out

    def miou_ref() -> float:
        return float(np.mean(skm.jaccard_score(ft, fp_, labels=labels, average=None)))

    def mpa_ref() -> float:
        return float(np.mean(skm.recall_score(ft, fp_, labels=labels, average=None)))

    c.add(
        "segmentation.average_surface_distance",
        lambda: [
            float(es.average_surface_distance(bt[i].astype(int), bp[i].astype(int), labels=[1]))
            for i in range(n_sd)
        ],
        "SciPy",
        asd_ref,
    )
    c.add("segmentation.boundary_iou", lambda: _f(es.boundary_iou(true, pred, labels=[k - 1])))
    c.add(
        "segmentation.dice",
        lambda: _numeric(es.dice(true, pred, average=None)),
        "scikit-learn",
        (lambda: list(skm.f1_score(ft, fp_, labels=labels, average=None))) if skm is not None else None,
    )
    c.add(
        "segmentation.hausdorff_distance",
        lambda: [
            float(es.hausdorff_distance(bt[i].astype(int), bp[i].astype(int), labels=[1])) for i in range(n_sd)
        ],
        "SciPy",
        hd_ref,
    )
    c.add(
        "segmentation.iou",
        lambda: _numeric(es.iou(true, pred, average=None)),
        "scikit-learn",
        (lambda: list(skm.jaccard_score(ft, fp_, labels=labels, average=None))) if skm is not None else None,
    )
    c.add(
        "segmentation.mean_pixel_accuracy",
        lambda: _f(es.mean_pixel_accuracy(true, pred, num_classes=k)),
        "scikit-learn",
        mpa_ref if skm is not None else None,
    )
    c.add(
        "segmentation.miou",
        lambda: _f(es.miou(true, pred, num_classes=k)),
        "scikit-learn",
        miou_ref if skm is not None else None,
    )
    c.add(
        "segmentation.pixel_accuracy",
        lambda: _f(es.pixel_accuracy(true, pred)),
        "scikit-learn",
        (lambda: skm.accuracy_score(ft, fp_)) if skm is not None else None,
    )

    # detection
    n_det = max(10, n // 1000)
    y_true, y_pred = [], []
    for _ in range(n_det):
        m = int(rng.integers(1, 8))
        xy = rng.uniform(0, 500, (m, 2))
        wh = rng.uniform(8, 160, (m, 2))
        boxes = np.column_stack([xy, xy + wh])
        lab = rng.integers(1, 6, m)
        y_true.append({"boxes": boxes, "labels": lab})
        jitter = boxes + rng.normal(0, 4, boxes.shape)
        jitter[:, 2:] = np.maximum(jitter[:, 2:], jitter[:, :2] + 1)
        extra = rng.uniform(0, 500, (3, 2))
        fpb = np.column_stack([extra, extra + rng.uniform(10, 80, (3, 2))])
        y_pred.append(
            {
                "boxes": np.vstack([jitter, fpb]),
                "labels": np.r_[lab, rng.integers(1, 6, 3)],
                "scores": rng.random(m + 3),
            }
        )
    ba = np.vstack([t["boxes"] for t in y_true])
    bb = np.vstack([p["boxes"] for p in y_pred])[: max(1, min(len(ba), 2000))]
    ba = ba[:2000]

    def iou_np() -> Any:
        x1 = np.maximum(ba[:, None, 0], bb[None, :, 0])
        y1 = np.maximum(ba[:, None, 1], bb[None, :, 1])
        x2 = np.minimum(ba[:, None, 2], bb[None, :, 2])
        y2 = np.minimum(ba[:, None, 3], bb[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        aa = (ba[:, 2] - ba[:, 0]) * (ba[:, 3] - ba[:, 1])
        ab = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
        return (inter / (aa[:, None] + ab[None, :] - inter)).ravel().tolist()

    coco: Optional[Callable[[], Any]] = None
    with contextlib.suppress(ImportError):
        import io

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        def coco_eval() -> Any:
            images, anns, dets, aid = [], [], [], 1
            for i, (t, p) in enumerate(zip(y_true, y_pred)):
                images.append({"id": i + 1})
                for bx, cl in zip(t["boxes"], t["labels"]):
                    w, h = bx[2] - bx[0], bx[3] - bx[1]
                    anns.append(
                        {
                            "id": aid,
                            "image_id": i + 1,
                            "category_id": int(cl),
                            "bbox": [bx[0], bx[1], w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                    aid += 1
                for bx, cl, sc in zip(p["boxes"], p["labels"], p["scores"]):
                    dets.append(
                        {
                            "image_id": i + 1,
                            "category_id": int(cl),
                            "bbox": [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]],
                            "score": float(sc),
                        }
                    )
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO()
                gt.dataset = {
                    "images": images,
                    "annotations": anns,
                    "categories": [{"id": cl} for cl in range(1, 6)],
                }
                gt.createIndex()
                ev = COCOeval(gt, gt.loadRes(dets), "bbox")
                ev.evaluate()
                ev.accumulate()
                ev.summarize()
            return ev

        coco = coco_eval

    c.add(
        "detection.average_precision_detection",
        lambda: _f(es.average_precision_detection(y_true, y_pred, iou_threshold=0.5, average="macro")),
        "pycocotools",
        (lambda: float(coco().stats[1])) if coco is not None else None,
    )
    c.add("detection.box_iou", lambda: es.box_iou(ba, bb).ravel().tolist(), "NumPy formula", iou_np)
    c.add(
        "detection.mean_average_precision",
        lambda: _f(es.mean_average_precision(y_true, y_pred)),
        "pycocotools",
        (lambda: float(coco().stats[0])) if coco is not None else None,
    )


# --------------------------------------------------------------------------------------------- LLM
def _sentence(rng: np.random.Generator, lo: int = 5, hi: int = 25) -> list[str]:
    return list(rng.choice(_VOCAB, int(rng.integers(lo, hi))))


def _llm(c: _Cases, n: int, rng: np.random.Generator) -> None:
    import evalsuite as es

    m = max(20, n // 100)
    me = max(10, n // 1000)
    refs, preds, multi = [], [], []
    for _ in range(m):
        base = _sentence(rng)
        keep = [w for w in base if rng.random() > 0.2] + list(rng.choice(_VOCAB, int(rng.integers(0, 4))))
        preds.append(" ".join(keep))
        refs.append(" ".join(base))
        multi.append([" ".join(base), " ".join(_sentence(rng, 3, 12)), " ".join(_sentence(rng, 3, 12))])
    lsum_refs = [r.replace(" mat ", " mat\n") for r in refs]
    lsum_preds = [p.replace(" mat ", " mat\n") for p in preds]

    # ---- text generation (sacreBLEU, rouge-score, NLTK, pycocoevalcap)
    sacre = _opt("sacrebleu")

    c.add(
        "text.bleu",
        lambda: _f(es.bleu(refs, preds)),
        "sacreBLEU",
        (lambda: sacre.corpus_bleu(preds, [refs], force=True).score) if sacre else None,
    )
    c.add(
        "text.chrf",
        lambda: _f(es.chrf(refs, preds)),
        "sacreBLEU",
        (lambda: sacre.corpus_chrf(preds, [refs]).score) if sacre else None,
    )
    c.add(
        "text.sentence_bleu",
        lambda: _numeric(es.sentence_bleu(refs, preds, average=None)),
        "sacreBLEU",
        (lambda: [sacre.sentence_bleu(p, [r]).score for r, p in zip(refs, preds)]) if sacre else None,
    )
    c.add(
        "text.ter",
        lambda: _f(es.ter(refs, preds)),
        "sacreBLEU",
        (lambda: sacre.corpus_ter(preds, [refs]).score) if sacre else None,
    )
    ms = refs[: max(20, m // 10)]
    msp = preds[: len(ms)]

    def self_bleu_ref() -> float:
        out = []
        for i, p in enumerate(msp):
            others = [msp[j] for j in range(len(msp)) if j != i]
            out.append(sacre.sentence_bleu(p, others).score)
        return float(np.mean(out))

    c.add("text.self_bleu", lambda: _f(es.self_bleu(msp)), "sacreBLEU", self_bleu_ref if sacre else None)
    del ms

    rs = _opt("rouge_score.rouge_scorer")

    def rouge_ref(key: str, r_: list[str], p_: list[str]) -> Callable[[], float]:
        def run() -> float:
            sc = rs.RougeScorer([key])
            return float(np.mean([sc.score(a, b)[key].fmeasure for a, b in zip(r_, p_)]))

        return run

    for name, key in (("rouge_1", "rouge1"), ("rouge_2", "rouge2"), ("rouge_l", "rougeL")):
        fn = getattr(es, name)
        c.add(
            f"text.{name}",
            lambda fn=fn: _f(fn(refs, preds)),  # type: ignore[misc]
            "rouge-score",
            rouge_ref(key, refs, preds) if rs else None,
        )
    c.add(
        "text.rouge_lsum",
        lambda: _f(es.rouge_lsum(lsum_refs, lsum_preds)),
        "rouge-score",
        rouge_ref("rougeLsum", lsum_refs, lsum_preds) if rs else None,
    )

    meteor_ref: Optional[Callable[[], float]] = None
    with contextlib.suppress(ImportError):
        from nltk.stem.porter import PorterStemmer
        from nltk.translate.meteor_score import meteor_score

        class _NoWordNet:
            def synsets(self, _w: str) -> list[Any]:
                return []

        def _meteor() -> float:
            st, wn = PorterStemmer(), _NoWordNet()
            return float(
                np.mean(
                    [meteor_score([r.split()], p.split(), stemmer=st, wordnet=wn) for r, p in zip(refs, preds)]
                )
            )

        meteor_ref = _meteor
    c.add("text.meteor", lambda: _f(es.meteor(refs, preds)), "NLTK", meteor_ref)

    cider_ref: Optional[Callable[[], float]] = None
    with contextlib.suppress(ImportError):
        from pycocoevalcap.cider.cider import Cider  # type: ignore[import-untyped]

        def _cider() -> float:
            res = {i: [p] for i, p in enumerate(preds)}
            return float(Cider().compute_score(dict(enumerate(multi)), res)[0])

        cider_ref = _cider
    c.add("text.cider", lambda: _f(es.cider(multi, preds)), "pycocoevalcap", cider_ref)

    def distinct_ref() -> float:
        grams = [tuple(t[i : i + 2]) for p in preds for t in [p.split()] for i in range(len(t) - 1)]
        return len(set(grams)) / len(grams)

    c.add("text.distinct_n", lambda: _f(es.distinct_n(preds, n=2)), "Python stdlib", distinct_ref)

    lps = [np.log(rng.uniform(0.05, 1.0, int(rng.integers(5, 40)))) for _ in range(m)]
    c.add(
        "text.perplexity",
        lambda: _f(es.perplexity(lps)),
        "NumPy formula",
        lambda: float(np.exp(-np.concatenate(lps).mean())),
    )
    c.add(
        "text.cross_entropy",
        lambda: _f(es.cross_entropy(lps, base=2)),
        "NumPy formula",
        lambda: float(-np.concatenate(lps).mean() / np.log(2)),
    )

    # ---- semantic
    dim = 32
    re_ = rng.normal(size=(m, dim))
    pe = re_ + rng.normal(0, 0.5, (m, dim))
    tok_r = [rng.normal(size=(int(rng.integers(5, 20)), dim)) for _ in range(me)]
    tok_p = [rng.normal(size=(int(rng.integers(5, 20)), dim)) for _ in range(me)]

    def unit(x: Any) -> Any:
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def bert_np() -> float:
        out = []
        for r_, p_ in zip(tok_r, tok_p):
            s = unit(r_) @ unit(p_).T
            rec, prec = s.max(1).mean(), s.max(0).mean()
            out.append(2 * prec * rec / (prec + rec))
        return float(np.mean(out))

    c.add("text.bertscore", lambda: _f(es.bertscore(tok_r, tok_p)), "NumPy formula", bert_np)
    c.add(
        "text.embedding_similarity",
        lambda: _f(es.embedding_similarity(re_, pe)),
        "NumPy formula",
        lambda: float(np.mean(np.sum(unit(re_) * unit(pe), 1))),
    )
    ot = _opt("ot")

    def mover_ref() -> float:
        out = []
        for x, y_ in zip(tok_r, tok_p):
            a_, b_ = np.full(len(x), 1 / len(x)), np.full(len(y_), 1 / len(y_))
            out.append(1 - ot.emd2(a_, b_, ot.dist(unit(x), unit(y_), metric="euclidean")))
        return float(np.mean(out))

    c.add("text.moverscore", lambda: _f(es.moverscore(tok_r, tok_p)), "POT", mover_ref if ot else None)
    hf = rng.normal(size=(me * 5, 16))
    gf = rng.normal(0.3, 1, size=(me * 5, 16))
    c.add("text.mauve", lambda: _f(es.mauve(hf, gf, random_state=0)))
    scores = rng.random(m)
    c.add(
        "text.model_score",
        lambda: _f(es.model_score(refs, preds, scorer=lambda r, p: scores[: len(p)].tolist(), batch_size=10**9)),
    )

    # ---- QA and factuality
    def squad_norm(s: str) -> list[str]:
        s = s.lower()
        s = "".join(ch for ch in s if ch not in set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return s.split()

    def em_ref() -> float:
        return float(np.mean([squad_norm(r) == squad_norm(p) for r, p in zip(refs, preds)]))

    def f1_ref() -> float:
        out = []
        for r, p in zip(refs, preds):
            rt, pt = squad_norm(r), squad_norm(p)
            common = sum((Counter(rt) & Counter(pt)).values())
            if not rt or not pt:
                out.append(float(rt == pt))
            elif common == 0:
                out.append(0.0)
            else:
                pr, rc = common / len(pt), common / len(rt)
                out.append(2 * pr * rc / (pr + rc))
        return float(np.mean(out))

    c.add("qa.exact_match", lambda: _f(es.exact_match(refs, preds)), "SQuAD formula", em_ref)
    c.add("qa.token_f1", lambda: _f(es.token_f1(refs, preds)), "SQuAD formula", f1_ref)

    verdict_vals = np.array(["supported", "contradicted", "unsupported"])
    claims = [list(rng.choice(verdict_vals, int(rng.integers(1, 8)), p=[0.7, 0.1, 0.2])) for _ in range(m)]
    c.add(
        "text.faithfulness",
        lambda: _f(es.faithfulness(claims)),
        "NumPy formula",
        lambda: float(np.mean([np.mean(np.asarray(v) == "supported") for v in claims])),
    )
    c.add(
        "text.hallucination_rate",
        lambda: _f(es.hallucination_rate(claims)),
        "NumPy formula",
        lambda: float(np.mean(np.concatenate([np.asarray(v) for v in claims]) != "supported")),
    )
    support = [rng.random(int(rng.integers(1, 6))) for _ in range(m)]
    c.add(
        "text.groundedness",
        lambda: _f(es.groundedness(support)),
        "NumPy formula",
        lambda: float(np.mean([s.mean() for s in support])),
    )
    cites = [
        [
            {
                "supported": bool(rng.random() < 0.7),
                "citations": [bool(t) for t in rng.random(int(rng.integers(0, 4))) < 0.6],
            }
            for _ in range(int(rng.integers(1, 5)))
        ]
        for _ in range(m)
    ]
    c.add("text.citation_precision", lambda: _f(es.citation_precision(cites)))
    c.add("text.citation_recall", lambda: _f(es.citation_recall(cites)))
    gold = rng.choice(verdict_vals, m)
    predv = np.where(rng.random(m) < 0.8, gold, rng.choice(verdict_vals, m))
    c.add(
        "text.claim_verification_accuracy",
        lambda: _f(es.claim_verification_accuracy(gold, predv)),
        "NumPy formula",
        lambda: float(np.mean(gold == predv)),
    )
    kb = {f"f{i}" for i in range(0, 500, 2)}
    facts = [{f"f{int(t)}" for t in rng.integers(0, 500, int(rng.integers(1, 6)))} for _ in range(m)]
    c.add(
        "text.knowledge_consistency",
        lambda: _f(es.knowledge_consistency(facts, kb)),
        "Python stdlib",
        lambda: sum(len(f & kb) for f in facts) / sum(len(f) for f in facts),
    )
    tp, fp, fn = rng.integers(0, 6, m), rng.integers(0, 3, m), rng.integers(0, 3, m)
    tp[tp + fp + fn == 0] = 1

    def ac_ref() -> float:
        return float(np.mean(tp / (tp + 0.5 * (fp + fn))))

    c.add("text.answer_correctness", lambda: _f(es.answer_correctness(tp, fp, fn)), "NumPy formula", ac_ref)
    qe = rng.normal(size=(m, dim))
    ge = [qe[i] + rng.normal(0, 0.7, (3, dim)) for i in range(m)]
    c.add(
        "text.answer_relevance",
        lambda: _f(es.answer_relevance(qe, ge)),
        "NumPy formula",
        lambda: float(np.mean([np.mean(unit(g) @ unit(q)) for q, g in zip(qe, ge)])),
    )
    sa = rng.random(m) < 0.2
    ab = np.where(rng.random(m) < 0.85, sa, ~sa)
    c.add(
        "text.abstention_accuracy",
        lambda: _f(es.abstention_accuracy(sa, ab)),
        "NumPy formula",
        lambda: float(np.mean(sa == ab)),
    )

    # ---- judges and preferences
    out_vals = np.array(["win", "tie", "loss"])
    outcomes = rng.choice(out_vals, m, p=[0.45, 0.15, 0.4])
    c.add(
        "text.win_rate",
        lambda: _f(es.win_rate(outcomes)),
        "NumPy formula",
        lambda: float(np.mean((outcomes == "win") + 0.5 * (outcomes == "tie"))),
    )
    strength = rng.normal(size=8)
    comps, pairs = [], []
    for _ in range(max(m, 400)):  # enough games that every model wins and loses (the MLE exists)
        i, j = rng.choice(8, 2, replace=False)
        win = rng.random() < 1 / (1 + np.exp(strength[j] - strength[i]))
        w, lo = (i, j) if win else (j, i)
        comps.append((f"m{w}", f"m{lo}", "win"))
        pairs.append((int(w), int(lo)))

    choix = _opt("choix")

    def bt_es() -> list[float]:
        r = es.bradley_terry(comps)
        order = np.argsort([int(s[1:]) for s in (r.labels or ())])
        return [float(v) for v in np.asarray(r.value, float)[order]]

    def bt_ref() -> list[float]:
        v = choix.mm_pairwise(8, pairs, max_iter=100_000, tol=1e-12)
        return [float(t) for t in v - v.mean()]

    c.add("text.bradley_terry", bt_es, "choix", bt_ref if choix else None)

    def elo_ref() -> list[float]:
        rating = {f"m{i}": 1000.0 for i in range(8)}
        for a_, b_, _ in comps:
            ea = 1 / (1 + 10 ** ((rating[b_] - rating[a_]) / 400))
            rating[a_] += 4 * (1 - ea)
            rating[b_] -= 4 * (1 - ea)
        return [rating[f"m{i}"] for i in range(8)]

    def elo_es() -> list[float]:
        r = es.elo_ratings(comps)
        d = dict(zip(r.labels or (), np.asarray(r.value, float)))
        return [d[f"m{i}"] for i in range(8)]

    c.add("text.elo_ratings", elo_es, "Python stdlib", elo_ref)

    ratings = rng.integers(1, 6, (4, m)).astype(float)
    ratings[rng.random(ratings.shape) < 0.1] = np.nan
    kd = _opt("krippendorff")
    c.add(
        "text.krippendorff_alpha",
        lambda: _f(es.krippendorff_alpha(ratings, level="ordinal")),
        "krippendorff",
        (lambda: float(kd.alpha(reliability_data=ratings, level_of_measurement="ordinal"))) if kd else None,
    )
    item_r = rng.integers(0, 3, (m, 4))
    ir = _opt("statsmodels.stats.inter_rater")
    c.add(
        "text.fleiss_kappa",
        lambda: _f(es.fleiss_kappa(item_r)),
        "statsmodels",
        (lambda: float(ir.fleiss_kappa(ir.aggregate_raters(item_r)[0]))) if ir else None,
    )
    js = rng.integers(1, 6, m)
    hs = np.clip(js + rng.integers(-1, 2, m), 1, 5)
    skm = _opt("sklearn.metrics")
    c.add(
        "text.judge_agreement",
        lambda: _f(es.judge_agreement(js, hs)),
        "scikit-learn",
        (lambda: skm.cohen_kappa_score(js, hs)) if skm is not None else None,
    )
    vo = rng.choice(np.array(["A", "B"]), m)
    vs = np.where(rng.random(m) < 0.8, vo, rng.choice(np.array(["A", "B"]), m))
    c.add(
        "text.position_consistency",
        lambda: _f(es.position_consistency(vo, vs)),
        "NumPy formula",
        lambda: float(np.mean(vo == vs)),
    )
    la, lb = rng.integers(20, 400, m), rng.integers(20, 400, m)
    c.add("text.verbosity_bias", lambda: _f(es.verbosity_bias(vo, la, lb)))
    jp, hp = rng.random(m) < 0.6, rng.random(m) < 0.5
    c.add("text.self_preference_bias", lambda: _f(es.self_preference_bias(jp, hp)))
    rub = rng.integers(1, 6, (m, 3))
    c.add(
        "text.rubric_score",
        lambda: _f(es.rubric_score(rub)),
        "NumPy formula",
        lambda: float(np.mean((rub - 1) / 4)),
    )

    # ---- reasoning
    ns = np.full(m, 20)
    nc = rng.integers(0, 21, m)

    def pass_ref() -> float:
        return float(
            np.mean([1 - math.comb(int(a_) - int(b_), 5) / math.comb(int(a_), 5) for a_, b_ in zip(ns, nc)])
        )

    c.add("reasoning.pass_at_k", lambda: _f(es.pass_at_k(ns, nc, k=5)), "Python stdlib", pass_ref)
    answers = [str(int(t)) for t in rng.integers(0, 50, m)]
    samples = [[a_ if rng.random() < 0.6 else str(int(rng.integers(0, 50))) for _ in range(5)] for a_ in answers]

    def mv_ref() -> float:
        return float(np.mean([Counter(s).most_common(1)[0][0] == a_ for a_, s in zip(answers, samples)]))

    c.add(
        "reasoning.majority_vote_accuracy",
        lambda: _f(es.majority_vote_accuracy(answers, samples)),
        "Python stdlib",
        mv_ref,
    )
    sols = [f"step one ... #### {a_}" for a_ in answers]
    outs = [f"so the answer is {s[0]}" for s in samples]
    c.add("reasoning.benchmark_accuracy", lambda: _f(es.benchmark_accuracy(sols, outs, style="gsm8k")))

    # ---- retrieval (ranx)
    rel = [set(rng.choice(200, int(rng.integers(1, 6)), replace=False).tolist()) for _ in range(m)]
    ret = [rng.permutation(200)[:20].tolist() for _ in range(m)]
    ranx = _opt("ranx")

    def ranx_ref(metric: str) -> Callable[[], float]:
        def run() -> float:
            qrels = ranx.Qrels({f"q{i}": {f"d{d}": 1 for d in r} for i, r in enumerate(rel)})
            run_ = ranx.Run(
                {f"q{i}": {f"d{d}": float(len(r) - j) for j, d in enumerate(r)} for i, r in enumerate(ret)}
            )
            return float(ranx.evaluate(qrels, run_, metric))

        return run

    for name, call, metric in (
        ("hit_rate_at_k", lambda: es.hit_rate_at_k(rel, ret, k=10), "hit_rate@10"),
        ("mean_average_precision_at_k", lambda: es.mean_average_precision_at_k(rel, ret, k=20), "map@20"),
        ("mrr", lambda: es.mrr(rel, ret), "mrr"),
        ("ndcg_at_k", lambda: es.ndcg_at_k(rel, ret, k=10), "ndcg@10"),
        ("precision_at_k", lambda: es.precision_at_k(rel, ret, k=10), "precision@10"),
        ("recall_at_k", lambda: es.recall_at_k(rel, ret, k=10), "recall@10"),
    ):
        c.add(
            f"retrieval.{name}",
            lambda call=call: _f(call()),  # type: ignore[misc]
            "ranx",
            ranx_ref(metric) if ranx else None,
        )

    # ---- RAG and operations
    chunk_rel = [list(rng.random(int(rng.integers(1, 10))) < 0.5) for _ in range(m)]

    def cp_ref() -> float:
        out = []
        for ch in chunk_rel:
            arr = np.asarray(ch, float)
            if arr.sum() == 0:
                out.append(0.0)
                continue
            prec = np.cumsum(arr) / np.arange(1, len(arr) + 1)
            out.append(float(np.sum(prec * arr) / arr.sum()))
        return float(np.mean(out))

    c.add("rag.context_precision", lambda: _f(es.context_precision(chunk_rel)), "NumPy formula", cp_ref)
    attributed = [list(rng.random(int(rng.integers(1, 8))) < 0.7) for _ in range(m)]
    c.add(
        "rag.context_recall",
        lambda: _f(es.context_recall(attributed)),
        "NumPy formula",
        lambda: float(np.mean([np.mean(a_) for a_ in attributed])),
    )
    c.add(
        "rag.context_relevance",
        lambda: _f(es.context_relevance(chunk_rel)),
        "NumPy formula",
        lambda: float(np.mean([np.mean(a_) for a_ in chunk_rel])),
    )
    stage_vals = np.array(["retrieval", "generation", "evidence", "orchestration", "other", ""])
    stages = [s or None for s in rng.choice(stage_vals, m)]
    c.add("rag.failure_attribution", lambda: _numeric(es.failure_attribution(stages)))
    lat = rng.lognormal(5, 0.5, m)
    c.add(
        "rag.latency_summary",
        lambda: _f(es.latency_summary(lat, statistic="p95")),
        "NumPy formula",
        lambda: float(np.percentile(lat, 95)),
    )
    succ = rng.random(m) < 0.8
    c.add(
        "rag.task_success_rate",
        lambda: _f(es.task_success_rate(succ)),
        "NumPy formula",
        lambda: float(np.mean(succ)),
    )

    # ---- structured output
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    docs = []
    for i in range(m):
        doc: dict[str, Any] = {"name": f"n{i}", "age": int(rng.integers(-2, 90)), "tags": ["a", "b"]}
        if rng.random() < 0.1:
            doc["extra"] = 1
        if rng.random() < 0.1:
            del doc["name"]
        docs.append(doc)
    docs_text = [json.dumps(d_) if rng.random() > 0.05 else json.dumps(d_)[:-1] for d_ in docs]

    def valid_json(t: str) -> bool:
        try:
            json.loads(t)
        except ValueError:
            return False
        return True

    c.add(
        "text.json_validity",
        lambda: _f(es.json_validity(docs_text)),
        "Python stdlib",
        lambda: float(np.mean([valid_json(t) for t in docs_text])),
    )
    jsch = _opt("jsonschema")

    def jsch_ref() -> float:
        v = jsch.Draft202012Validator(schema)
        return float(np.mean([valid_json(t) and v.is_valid(json.loads(t)) for t in docs_text]))

    c.add(
        "text.json_schema_compliance",
        lambda: _f(es.json_schema_compliance(docs_text, schema)),
        "jsonschema",
        jsch_ref if jsch else None,
    )
    xmls = [f"<a><b>{i}</b></a>" if rng.random() < 0.8 else f"<a><b>{i}</a>" for i in range(m)]

    def valid_xml(t: str) -> bool:
        try:
            ET.fromstring(t)  # noqa: S314 - benchmark data generated above, not user input
        except ET.ParseError:
            return False
        return True

    c.add(
        "text.xml_validity",
        lambda: _f(es.xml_validity(xmls)),
        "Python stdlib",
        lambda: float(np.mean([valid_xml(t) for t in xmls])),
    )
    expected = [{"name": d_.get("name", "x"), "age": d_["age"]} for d_ in docs]
    c.add("text.required_field_accuracy", lambda: _f(es.required_field_accuracy(expected, docs_text)))
    tools = ["search", "lookup", "calc", "email"]
    exp_calls = [
        [{"name": str(rng.choice(tools)), "arguments": {"q": f"x{int(rng.integers(0, 5))}"}}] for _ in range(m)
    ]
    pred_calls = [
        [c_ if rng.random() < 0.8 else {"name": str(rng.choice(tools)), "arguments": {"q": "x0"}} for c_ in e_]
        for e_ in exp_calls
    ]
    c.add(
        "text.tool_selection_accuracy",
        lambda: _f(es.tool_selection_accuracy(exp_calls, pred_calls)),
        "Python stdlib",
        lambda: float(
            np.mean([[x["name"] for x in e_] == [x["name"] for x in p_] for e_, p_ in zip(exp_calls, pred_calls)])
        ),
    )
    c.add("text.tool_argument_accuracy", lambda: _f(es.tool_argument_accuracy(exp_calls, pred_calls)))
    c.add("text.tool_call_f1", lambda: _f(es.tool_call_f1(exp_calls, pred_calls)))
    api = [bool(t) for t in rng.random(m) < 0.9]
    c.add(
        "text.api_call_success_rate",
        lambda: _f(es.api_call_success_rate(api)),
        "NumPy formula",
        lambda: float(np.mean(api)),
    )
    checks = [list(rng.random(int(rng.integers(1, 6))) < 0.8) for _ in range(m)]
    c.add(
        "text.instruction_compliance_rate",
        lambda: _f(es.instruction_compliance_rate(checks)),
        "NumPy formula",
        lambda: float(np.mean([all(ch) for ch in checks])),
    )
    c.add(
        "text.constraint_satisfaction_rate",
        lambda: _f(es.constraint_satisfaction_rate(checks)),
        "NumPy formula",
        lambda: float(np.mean(np.concatenate([np.asarray(ch, float) for ch in checks]))),
    )
    fmt = [f"Answer: {rng.choice(list('ABCDE'))}" for _ in range(m)]
    pat = re.compile(r"Answer: [A-D]", re.DOTALL)
    c.add(
        "text.format_compliance",
        lambda: _f(es.format_compliance(fmt, r"Answer: [A-D]")),
        "Python stdlib",
        lambda: float(np.mean([pat.fullmatch(t) is not None for t in fmt])),
    )
    turns = [[list(rng.random(3) < 0.8) for _ in range(int(rng.integers(1, 4)))] for _ in range(m)]
    c.add("text.instruction_retention", lambda: _f(es.instruction_retention(turns)))
    wrapped = [t if rng.random() < 0.8 else f"Sure! {t}" for t in docs_text]
    c.add("text.extra_content_rate", lambda: _f(es.extra_content_rate(wrapped)))


def metric_cases(n: int, rng: np.random.Generator) -> list[Case]:
    """One benchmark case for every registered metric and every statistics function."""
    c = _Cases()
    for build in (_classification, _regression, _clinical, _statistics, _vision, _llm):
        build(c, n, rng)
    return sorted(c.rows, key=lambda r: r[0])
