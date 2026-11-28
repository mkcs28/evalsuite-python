"""Multi-label classification and ranked retrieval / recommendation reports, with confidence intervals.

>>> import evalsuite as es
>>> es.multilabel_report(Y_true, Y_pred, Y_score)        # n × labels 0/1 matrices (+ scores for ranking metrics)
>>> es.ranking_report(qrels, {"bm25": run_a, "dense": run_b}, k=10)

``multilabel_report``: subset accuracy, Hamming loss, micro / macro / sample-averaged precision, recall and F1,
Jaccard, and with scores label-ranking average precision, coverage error, ranking loss and micro / macro ROC AUC
(definitions as scikit-learn), each with a percentile bootstrap interval over examples, plus a per-label table.

``ranking_report``: per-query nDCG@k, AP@k, reciprocal rank, precision@k and recall@k (trec_eval conventions:
linear gains, log2 discount, AP and recall over all relevant documents in the judgements), averaged over queries
with bootstrap intervals, and paired randomization tests (Smucker, Allan & Carterette 2007) of every system
against a baseline with Holm adjustment.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError

__all__ = ["MultilabelReport", "multilabel_report", "RankingReport", "ranking_report"]


def _ci(draws: np.ndarray, level: float) -> tuple[float, float]:
    d = draws[np.isfinite(draws)]
    if d.size < 10:
        return (math.nan, math.nan)
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


# --------------------------------------------------------------------------------------------- multi-label


def _prf(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1), 0.0)
        r = np.where(tp + fn > 0, tp / np.where(tp + fn > 0, tp + fn, 1), 0.0)
        f = np.where(2 * tp + fp + fn > 0, 2 * tp / np.where(2 * tp + fp + fn > 0, 2 * tp + fp + fn, 1), 0.0)
    return p, r, f


def _per_sample(Y: np.ndarray, P: np.ndarray, S: Optional[np.ndarray]) -> dict[str, np.ndarray]:
    """Per-example quantities whose (weighted) mean is a sample-averaged metric."""
    n, L = Y.shape
    stp = (Y & P).sum(axis=1).astype(float)
    sfp = (~Y & P).sum(axis=1).astype(float)
    sfn = (Y & ~P).sum(axis=1).astype(float)
    _, _, sf = _prf(stp, sfp, sfn)
    union = (Y | P).sum(axis=1)
    out = {
        "subset_accuracy": np.all(Y == P, axis=1).astype(float),
        "hamming_loss": np.mean(Y != P, axis=1),
        "samples_f1": sf,
        "jaccard_samples": np.where(union > 0, stp / np.where(union > 0, union, 1), 1.0),
    }
    if S is None:
        return out
    nrel = Y.sum(axis=1)
    lrap = np.ones(n)
    cov = np.zeros(n)
    rl = np.zeros(n)
    step = max(1, int(4_000_000 // max(1, L * L)))
    for a in range(0, n, step):
        y, sc = Y[a : a + step], S[a : a + step]
        ge = sc[:, None, :] >= sc[:, :, None]  # [i, l, m]: score of m ≥ score of l
        rank_all = ge.sum(axis=2)
        rank_rel = (ge & y[:, None, :]).sum(axis=2)
        nr = nrel[a : a + step]
        ok = (nr > 0) & (nr < L)
        ratio = np.where(y, rank_rel / rank_all, 0.0).sum(axis=1)
        lrap[a : a + step] = np.where(ok, ratio / np.where(nr > 0, nr, 1), 1.0)
        low = np.where(y, sc, np.inf).min(axis=1)
        cov[a : a + step] = np.where(nr > 0, (sc >= low[:, None]).sum(axis=1), 0.0)
        wrong = (ge & y[:, :, None] & ~y[:, None, :]).sum(axis=(1, 2))  # relevant l scored ≤ irrelevant m
        rl[a : a + step] = np.where(ok, wrong / np.where(ok, nr * (L - nr), 1), 0.0)
    out.update(label_ranking_average_precision=lrap, coverage_error=cov, ranking_loss=rl)
    return out


def _all_metrics(Y: np.ndarray, P: np.ndarray, S: Optional[np.ndarray], M: np.ndarray) -> dict[str, np.ndarray]:
    """Every metric on each row of multiplicities ``M`` (resamples × examples): a resample counts example i
    M[b, i] times, which equals evaluating the resampled rows."""
    from .stats.intervals import _weighted_curves

    n, L = Y.shape
    Mf = M.astype(np.float64)
    tot = Mf.sum(axis=1)
    out: dict[str, np.ndarray] = {}
    for k, v in _per_sample(Y, P, S).items():
        out[k] = (Mf @ v) / tot
    tp = Mf @ (Y & P).astype(float)
    fp = Mf @ (~Y & P).astype(float)
    fn = Mf @ (Y & ~P).astype(float)
    mp, mr, mf = _prf(tp.sum(axis=1), fp.sum(axis=1), fn.sum(axis=1))
    lp, lr, lf = _prf(tp, fp, fn)
    out.update(
        micro_precision=mp,
        micro_recall=mr,
        micro_f1=mf,
        macro_precision=lp.mean(axis=1),
        macro_recall=lr.mean(axis=1),
        macro_f1=lf.mean(axis=1),
    )
    if S is not None:
        rows = np.arange(n)
        per_label = [
            _weighted_curves(True, Y[:, j].astype(float), S[:, j], None, rows, M)
            for j in range(L)
            if 0 < Y[:, j].sum() < n
        ]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["macro_roc_auc"] = (
                np.nanmean(np.vstack(per_label), axis=0) if per_label else np.full(M.shape[0], math.nan)
            )
        out["micro_roc_auc"] = _weighted_curves(
            True, Y.ravel().astype(float), S.ravel(), None, np.repeat(rows, L), M
        )
    order = [
        "subset_accuracy",
        "hamming_loss",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "samples_f1",
        "jaccard_samples",
        "label_ranking_average_precision",
        "coverage_error",
        "ranking_loss",
        "macro_roc_auc",
        "micro_roc_auc",
    ]
    return {k: out[k] for k in order if k in out}


@dataclass(frozen=True)
class MultilabelReport:
    """Multi-label metrics with bootstrap intervals (``metrics[name] = {value, low, high}``) and per-label rows."""

    metrics: dict[str, dict[str, float]]
    labels: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def __getitem__(self, name: str) -> float:
        return float(self.metrics[name]["value"])

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        lines = [
            f"Multi-label report: {s['n']} examples, {s['n_labels']} labels, {s['level'] * 100:g}% bootstrap CIs"
        ]
        for k, m in self.metrics.items():
            lines.append(f"  {k:<34} {m['value']:.{d}f} ({m['low']:.{d}f}–{m['high']:.{d}f})")
        if s.get("undefined"):
            lines.append(f"  Undefined on these data (left out): {', '.join(s['undefined'])}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {"metrics": self.metrics, "labels": list(self.labels), "settings": dict(self.settings)}
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Metric | Value | CI |", "| --- | ---: | --- |"]
        rows += [
            f"| {k} | {m['value']:.{d}f} | {m['low']:.{d}f}–{m['high']:.{d}f} |" for k, m in self.metrics.items()
        ]
        rows += ["", "| Label | Support | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
        rows += [
            f"| {r['label']} | {r['support']} | {r['precision']:.{d}f} | {r['recall']:.{d}f} | {r['f1']:.{d}f} |"
            for r in self.labels
        ]
        return "\n".join(rows) + "\n"


def _matrix(x: Any, what: str, *, binary: bool) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 2:
        raise InputValidationError(f"{what} must be an n × labels matrix.")
    if binary:
        if not np.all(np.isin(a, (0, 1, True, False))):
            raise InputValidationError(f"{what} must contain only 0 / 1.")
        return a.astype(bool)
    a = a.astype(np.float64)
    if not np.all(np.isfinite(a)):
        raise InputValidationError(f"{what} must be finite.")
    return a


def multilabel_report(
    y_true: Any,
    y_pred: Any = None,
    y_score: Any = None,
    *,
    threshold: float = 0.5,
    labels: Optional[Sequence[str]] = None,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: Optional[int] = None,
) -> MultilabelReport:
    """Multi-label evaluation (see the module docstring). Give ``y_pred`` (0/1), ``y_score`` (probabilities or
    scores; predictions are then ``y_score >= threshold`` unless ``y_pred`` is given), or both."""
    Y = _matrix(y_true, "y_true", binary=True)
    if y_pred is None and y_score is None:
        raise InputValidationError("Give y_pred, y_score or both.")
    S = _matrix(y_score, "y_score", binary=False) if y_score is not None else None
    if y_pred is not None:
        P = _matrix(y_pred, "y_pred", binary=True)
    elif S is not None:
        P = np.asarray(threshold <= S)
    if P.shape != Y.shape or (S is not None and S.shape != Y.shape):
        raise InputValidationError("y_true, y_pred and y_score must have the same shape.")
    n, L = Y.shape
    if n < 2:
        raise InputValidationError("At least 2 examples are needed.")
    names = list(labels) if labels is not None else [str(j) for j in range(L)]
    if len(names) != L:
        raise InputValidationError(f"{len(names)} label names for {L} columns.")

    point = {k: float(v[0]) for k, v in _all_metrics(Y, P, S, np.ones((1, n), dtype=np.int64)).items()}
    rng = np.random.default_rng(random_state)
    draws: dict[str, list[np.ndarray]] = {k: [] for k in point}
    step = max(1, min(n_resamples, int(2_000_000 // max(1, n))))
    done = 0
    while done < n_resamples:
        chunk = min(step, n_resamples - done)
        idx = rng.integers(0, n, size=(chunk, n))
        mult = np.zeros((chunk, n), dtype=np.int64)
        np.add.at(mult, (np.repeat(np.arange(chunk), n), idx.ravel()), 1)
        for k, v in _all_metrics(Y, P, S, mult).items():
            draws[k].append(v)
        done += chunk
    boot = {k: np.concatenate(v) for k, v in draws.items()}
    undefined = [k for k, v in point.items() if not math.isfinite(v)]
    metrics = {
        k: {"value": v, **dict(zip(("low", "high"), _ci(boot[k], level)))}
        for k, v in point.items()
        if k not in undefined
    }
    tp = (Y & P).sum(axis=0).astype(float)
    fp = (~Y & P).sum(axis=0).astype(float)
    fn = (Y & ~P).sum(axis=0).astype(float)
    lp, lr, lf = _prf(tp, fp, fn)
    rows = tuple(
        {
            "label": names[j],
            "support": int(Y[:, j].sum()),
            "precision": float(lp[j]),
            "recall": float(lr[j]),
            "f1": float(lf[j]),
        }
        for j in range(L)
    )
    settings = {"n": n, "n_labels": L, "level": level, "n_resamples": n_resamples, "undefined": undefined}
    return MultilabelReport(metrics, rows, settings)


# ------------------------------------------------------------------------------------------------- ranking


def _run_lists(run: Mapping[Any, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for q, r in run.items():
        if isinstance(r, Mapping):  # doc -> score: by score, ties by document id descending (as trec_eval)
            scored = [(float(s), str(d)) for d, s in r.items()]
            if any(not math.isfinite(s) for s, _ in scored):
                raise InputValidationError(f"Query {q!r}: document scores must be finite.")
            ranked = [d for _, d in sorted(scored, reverse=True)]
        else:
            ranked = [str(d) for d in r]
        out[str(q)] = list(dict.fromkeys(ranked))  # a document counts once, at its first (best) rank
    return out


def _per_query(rel: Mapping[str, float], ranked: list[str], k: int) -> dict[str, float]:
    top = ranked[:k]
    gains = [float(rel.get(d, 0.0)) for d in top]
    relevant = {d for d, g in rel.items() if g > 0}
    R = len(relevant)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains) if g > 0)
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    hits = 0
    ap = 0.0
    rr = 0.0
    for i, d in enumerate(top):
        if d in relevant:
            hits += 1
            ap += hits / (i + 1)
            if rr == 0.0:
                rr = 1.0 / (i + 1)
    return {
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
        "ap": ap / R if R else 0.0,
        "rr": rr,
        "precision": hits / k,
        "recall": hits / R if R else 0.0,
    }


@dataclass(frozen=True)
class RankingReport:
    """Retrieval / recommendation metrics per system with intervals and tests against a baseline."""

    systems: tuple[dict[str, Any], ...]
    per_query: Mapping[str, Mapping[str, Mapping[str, float]]]
    settings: Mapping[str, Any]

    def best(self, metric: str = "ndcg") -> str:
        return str(max(self.systems, key=lambda s: s[metric]["value"])["system"])

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        lines = [
            f"Ranking report: {s['n_queries']} queries, k={s['k']}, {s['level'] * 100:g}% bootstrap CIs, "
            f"baseline {s['baseline']!r}"
        ]
        for row in self.systems:
            parts = [
                f"{m} {row[m]['value']:.{d}f} ({row[m]['low']:.{d}f}–{row[m]['high']:.{d}f})" for m in s["metrics"]
            ]
            line = f"  {row['system']:<14} " + "  ".join(parts)
            if row.get("test"):
                t = row["test"]
                line += f"  | Δ{s['primary']} {t['difference']:+.{d}f} p_holm={t['p_adjusted']:.{d}g}"
            lines.append(line)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe({"systems": list(self.systems), "settings": dict(self.settings)})
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        ms = self.settings["metrics"]
        rows = [
            "| System | " + " | ".join(ms) + " | Δ vs baseline (p, Holm) |",
            "| --- |" + " ---: |" * len(ms) + " --- |",
        ]
        for r in self.systems:
            cells = [f"{r[m]['value']:.{d}f} ({r[m]['low']:.{d}f}–{r[m]['high']:.{d}f})" for m in ms]
            t = r.get("test")
            tail = f"{t['difference']:+.{d}f} (p={t['p_adjusted']:.{d}g})" if t else "—"
            rows.append(f"| {r['system']} | " + " | ".join(cells) + f" | {tail} |")
        return "\n".join(rows) + "\n"


def _randomization_p(diff: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    obs = abs(diff.mean())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (n_perm + 1))


def ranking_report(
    qrels: Mapping[Any, Mapping[Any, float]],
    runs: Mapping[str, Mapping[Any, Any]],
    *,
    k: int = 10,
    baseline: Optional[str] = None,
    primary: str = "ndcg",
    level: float = 0.95,
    n_resamples: int = 2000,
    n_permutations: int = 10_000,
    random_state: Optional[int] = None,
) -> RankingReport:
    """Evaluate ranked runs against relevance judgements (see the module docstring).

    ``qrels`` maps query → {doc: relevance (0 = not relevant)}; each run maps query → ranked list of docs or
    {doc: score}. Queries in ``qrels`` with no relevant document are skipped; a query a run does not answer scores
    0. ``primary`` (ndcg, ap, rr, precision, recall) is the measure tested against ``baseline`` (first run)."""
    if k < 1:
        raise InputValidationError("k must be at least 1.")
    if not isinstance(qrels, Mapping) or not isinstance(runs, Mapping):
        raise InputValidationError(
            "qrels must map query → {doc: relevance} and runs must map system name → run (query → ranked docs)."
        )
    if not runs:
        raise InputValidationError("runs must map system names to runs.")
    metrics = ["ndcg", "ap", "rr", "precision", "recall"]
    if primary not in metrics:
        raise InputValidationError(f"primary must be one of {', '.join(metrics)}.")
    rel = {str(q): {str(d): float(g) for d, g in docs.items()} for q, docs in qrels.items()}
    queries = sorted(q for q, docs in rel.items() if any(g > 0 for g in docs.values()))
    if not queries:
        raise InputValidationError("No query in qrels has a relevant document.")
    names = list(runs)
    base = baseline if baseline is not None else names[0]
    if base not in runs:
        raise InputValidationError(f"baseline {base!r} is not one of the runs.")
    per: dict[str, dict[str, dict[str, float]]] = {}
    for name in names:
        lists = _run_lists(runs[name])
        per[name] = {q: _per_query(rel[q], lists.get(q, []), k) for q in queries}
    rng = np.random.default_rng(random_state)
    nq = len(queries)
    boot_idx = rng.integers(0, nq, size=(n_resamples, nq))
    rows: list[dict[str, Any]] = []
    pvals: list[float] = []
    for name in names:
        row: dict[str, Any] = {"system": name}
        for m in metrics:
            v = np.array([per[name][q][m] for q in queries])
            lo, hi = _ci(v[boot_idx].mean(axis=1), level)
            row[m] = {"value": float(v.mean()), "low": lo, "high": hi}
        if name != base:
            a = np.array([per[name][q][primary] for q in queries])
            b = np.array([per[base][q][primary] for q in queries])
            p = _randomization_p(a - b, n_permutations, rng)
            row["test"] = {"difference": float((a - b).mean()), "p_value": p, "test": "paired randomization"}
            pvals.append(p)
        rows.append(row)
    if pvals:
        from .stats.effect import adjust_pvalues

        adj = iter(adjust_pvalues(pvals, method="holm"))
        for row in rows:
            if "test" in row:
                row["test"]["p_adjusted"] = float(next(adj))
    settings = {
        "k": k,
        "n_queries": nq,
        "baseline": base,
        "primary": primary,
        "metrics": metrics,
        "level": level,
        "n_resamples": n_resamples,
        "n_permutations": n_permutations,
    }
    return RankingReport(tuple(rows), per, settings)
