"""Fairness: performance by group, gaps between groups with confidence intervals, and corrected tests.

``fairness_report(y_true, y_pred, y_prob, groups=...)`` reports, for every group (and every intersection of
several attributes), the selection rate, true and false positive rates, positive predictive value and accuracy
(Wilson intervals), plus ROC AUC (DeLong) and calibration-in-the-large with probabilities. It then gives the
standard gap measures with bootstrap intervals that resample within each group:

* demographic parity difference (max − min selection rate) and ratio (min / max, the "four-fifths rule"),
* equalized odds difference (the larger of the TPR and FPR gaps; Hardt et al. 2016),
* equal opportunity difference (TPR gap) and predictive parity difference (PPV gap),

and tests every group against a reference group (Fisher's exact test per metric, Holm-adjusted across all
tests). Groups smaller than ``min_group_size`` are listed but not tested. The gap definitions match Fairlearn's.
"""

from __future__ import annotations

import html as _html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .core.exceptions import InputValidationError

__all__ = ["FairnessReport", "fairness_report"]

_RATES = {
    "selection_rate": ("Selection rate", "pred_pos", "n"),
    "tpr": ("True positive rate", "tp", "pos"),
    "fpr": ("False positive rate", "fp", "neg"),
    "ppv": ("Positive predictive value", "tp", "pred_pos"),
    "accuracy": ("Accuracy", "correct", "n"),
}


def _counts(y: np.ndarray, p: np.ndarray) -> dict[str, int]:
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    return {
        "n": tp + fp + fn + tn,
        "pos": tp + fn,
        "neg": fp + tn,
        "pred_pos": tp + fp,
        "tp": tp,
        "fp": fp,
        "correct": tp + tn,
    }


def _rate(c: Mapping[str, int], key: str) -> float:
    _, num, den = _RATES[key]
    return c[num] / c[den] if c[den] else math.nan


@dataclass(frozen=True, eq=False)
class FairnessReport:
    """Per-group metrics, gap measures and tests (see :func:`fairness_report`)."""

    groups: tuple[dict[str, Any], ...]
    gaps: tuple[dict[str, Any], ...]
    tests: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        pct = f"{self.settings['level'] * 100:g}%"
        lines = [
            f"EvalSuite fairness report ({len(self.groups)} groups, reference: "
            f"{self.settings['reference']}, {pct} CIs)"
        ]
        keys = [k for k in _RATES if any(k in g for g in self.groups)]
        if any("roc_auc" in g for g in self.groups):
            keys.append("roc_auc")
        width = max(len(str(g["group"])) for g in self.groups)
        lines.append("  " + "group".ljust(width) + "      n  " + "  ".join(k.ljust(20) for k in keys))
        for g in self.groups:
            cells = []
            for k in keys:
                v = g.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("–".ljust(20))
                else:
                    lo, hi = g.get(f"{k}_low"), g.get(f"{k}_high")
                    cell = f"{v:.{d}f}" + (
                        f" ({lo:.{d}f}–{hi:.{d}f})" if lo is not None and not math.isnan(lo) else ""
                    )
                    cells.append(cell.ljust(20))
            flag = "" if g["tested"] else "  (too small to test)"
            lines.append(f"  {str(g['group']).ljust(width)} {g['n']:6d}  " + "  ".join(cells) + flag)
        lines.append("  Gaps:")
        for gp in self.gaps:
            lines.append(f"    {gp['name']:<36} {gp['value']:.{d}f} ({gp['low']:.{d}f}–{gp['high']:.{d}f})")
        sig = [t for t in self.tests if t["significant"]]
        lines.append(
            f"  Tests vs {self.settings['reference']}: {len(sig)} of {len(self.tests)} significant (Holm)."
        )
        for t in sig:
            lines.append(
                f"   * {t['group']} {t['metric']}: {t['difference']:+.{d}f} (p_adj={t['p_adjusted']:.3g})"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "groups": list(self.groups),
                "gaps": list(self.gaps),
                "tests": list(self.tests),
                "settings": dict(self.settings),
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        keys = [k for k in (*_RATES, "roc_auc", "calibration_in_the_large") if any(k in g for g in self.groups)]
        head = "| Group | n | " + " | ".join(keys) + " |"
        rows = [head, "| --- | ---: |" + " --- |" * len(keys)]
        for g in self.groups:
            cells = []
            for k in keys:
                v = g.get(k)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cells.append("–")
                    continue
                lo, hi = g.get(f"{k}_low"), g.get(f"{k}_high")
                cells.append(
                    f"{v:.{d}f}" + (f" ({lo:.{d}f}–{hi:.{d}f})" if lo is not None and not math.isnan(lo) else "")
                )
            rows.append(f"| {g['group']} | {g['n']} | " + " | ".join(cells) + " |")
        rows += ["", "| Gap | Value | CI |", "| --- | ---: | --- |"]
        rows += [f"| {g['name']} | {g['value']:.{d}f} | {g['low']:.{d}f}–{g['high']:.{d}f} |" for g in self.gaps]
        if self.tests:
            rows += [
                "",
                f"| Group vs {self.settings['reference']} | Metric | Difference | p (Holm) |",
                "| --- | --- | ---: | ---: |",
            ]
            rows += [
                f"| {t['group']} | {t['metric']} | {t['difference']:+.{d}f} | "
                f"{t['p_adjusted']:.3g}{' *' if t['significant'] else ''} |"
                for t in self.tests
            ]
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        md = self.to_markdown(digits=digits)
        body = []
        for block in md.strip().split("\n\n"):
            lines = [line for line in block.splitlines() if not set(line.replace("|", "").strip()) <= set("-: ")]
            cells = [[_html.escape(c.strip()) for c in line.strip("|").split("|")] for line in lines]
            if not cells:
                continue
            head = "".join(f"<th>{c}</th>" for c in cells[0])
            rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells[1:])
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        return "\n".join(body)


def _group_labels(groups: Any, n: int, intersectional: bool) -> tuple[np.ndarray, list[str]]:
    if isinstance(groups, Mapping):
        if not groups:
            raise InputValidationError("groups must name at least one attribute.")
        cols = {str(k): np.asarray(v).ravel() for k, v in groups.items()}
    else:
        arr = np.asarray(groups)
        cols = {f"attr{j}": arr[:, j] for j in range(arr.shape[1])} if arr.ndim == 2 else {"group": arr.ravel()}
    for k, v in cols.items():
        if v.shape[0] != n:
            raise InputValidationError(f"groups[{k!r}] must have one value per observation ({n}).")
        missing = [i for i, x in enumerate(v) if x is None or (isinstance(x, float) and math.isnan(x))]
        if missing:
            raise InputValidationError(
                f"groups[{k!r}] has {len(missing)} missing value(s) (first at position {missing[0]}); give every "
                "observation a group, for example an explicit 'unknown' category."
            )
    names = list(cols)
    if len(names) == 1:
        return np.array([str(x) for x in cols[names[0]]], dtype=object), names
    labels = np.array([" & ".join(f"{k}={cols[k][i]}" for k in names) for i in range(n)], dtype=object)
    return labels, names


def fairness_report(
    y_true: Any,
    y_pred: Any = None,
    y_prob: Any = None,
    *,
    groups: Union[Mapping[str, Any], Sequence[Any], np.ndarray],
    pos_label: Any = 1,
    threshold: float = 0.5,
    reference: Optional[str] = None,
    intersectional: bool = True,
    min_group_size: int = 10,
    level: float = 0.95,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_state: Optional[int] = 0,
) -> FairnessReport:
    """Group fairness of a binary classifier.

    ``groups``: one protected-attribute value per observation, or a mapping of attribute name → values (several
    attributes are crossed into intersectional groups such as ``sex=F & age=65+``; with
    ``intersectional=False`` each attribute is analysed on its own and the reports' groups are listed per
    attribute). ``y_pred`` gives labels; with only ``y_prob`` labels are ``y_prob >= threshold``. ``reference``:
    the group the others are tested against (default: the largest)."""
    from . import adjust_pvalues, fisher_exact_test, proportion_ci, roc_auc_ci

    if not 0 < level < 1 or not 0 < alpha < 1:
        raise InputValidationError("level and alpha must be between 0 and 1.")
    yt = np.asarray(y_true).ravel()
    n = yt.shape[0]
    if n == 0:
        raise InputValidationError("y_true is empty.")
    y = (yt == pos_label).astype(int)
    if y_pred is None and y_prob is None:
        raise InputValidationError("fairness_report needs y_pred, y_prob, or both.")
    prob = None if y_prob is None else np.asarray(y_prob, dtype=float).ravel()
    if prob is not None and (prob.shape[0] != n or not np.all((prob >= 0) & (prob <= 1))):
        raise InputValidationError("y_prob must hold one probability in [0, 1] per observation.")
    if y_pred is not None:
        yp_raw = np.asarray(y_pred).ravel()
        if yp_raw.shape[0] != n:
            raise InputValidationError("y_pred must have one label per observation.")
        p = (yp_raw == pos_label).astype(int)
    else:
        p = (np.asarray(prob) >= threshold).astype(int)

    if isinstance(groups, Mapping) and len(groups) > 1 and not intersectional:
        # one report per attribute, merged: group names carry the attribute
        parts = [
            fairness_report(
                yt,
                y_pred,
                y_prob,
                groups={k: v},
                pos_label=pos_label,
                threshold=threshold,
                intersectional=True,
                min_group_size=min_group_size,
                level=level,
                alpha=alpha,
                n_resamples=n_resamples,
                random_state=random_state,
            )
            for k, v in groups.items()
        ]
        return FairnessReport(
            groups=tuple(g for r in parts for g in r.groups),
            gaps=tuple(
                {**g, "name": f"{g['name']} [{r.settings['attributes'][0]}]"} for r in parts for g in r.gaps
            ),
            tests=tuple(t for r in parts for t in r.tests),
            settings={**parts[0].settings, "attributes": list(groups), "reference": "per attribute"},
        )

    labels, attrs = _group_labels(groups, n, intersectional)
    if isinstance(groups, Mapping) and len(attrs) == 1:
        labels = np.array([f"{attrs[0]}={v}" for v in np.asarray(groups[attrs[0]]).ravel()], dtype=object)
    names, codes = np.unique(labels.astype(str), return_inverse=True)
    if names.size < 2:
        raise InputValidationError("fairness_report needs at least two groups.")
    sizes = np.bincount(codes, minlength=names.size)
    if reference is None:
        ref_code = int(np.argmax(sizes))
    elif str(reference) in names:
        ref_code = int(np.flatnonzero(names == str(reference))[0])
    else:
        raise InputValidationError(f"reference {reference!r} is not one of the groups: {list(names)}.")
    rates = [k for k in _RATES]

    rows: list[dict[str, Any]] = []
    counts: list[dict[str, int]] = []
    for g, name in enumerate(names):
        m = codes == g
        c = _counts(y[m], p[m])
        counts.append(c)
        row: dict[str, Any] = {
            "group": str(name),
            "n": c["n"],
            "positives": c["pos"],
            "tested": bool(c["n"] >= min_group_size),
        }
        for k in rates:
            _, num, den = _RATES[k]
            row[k] = _rate(c, k)
            if c[den]:
                ci = proportion_ci(c[num], c[den], level=level)
                row[f"{k}_low"], row[f"{k}_high"] = ci.low, ci.high
        if prob is not None:
            row["calibration_in_the_large"] = float(y[m].mean() - prob[m].mean())
            if 0 < c["pos"] < c["n"]:
                ci = roc_auc_ci(y[m], prob[m], level=level)
                row["roc_auc"], row["roc_auc_low"], row["roc_auc_high"] = ci.estimate, ci.low, ci.high
        rows.append(row)

    # gap measures, with a bootstrap that resamples within each group (group sizes fixed)
    def gaps_of(cs: list[dict[str, int]]) -> dict[str, float]:
        tested = [c for c in cs if c["n"] >= min_group_size] or cs
        sel = [_rate(c, "selection_rate") for c in tested]
        tpr = [_rate(c, "tpr") for c in tested if c["pos"]]
        fpr = [_rate(c, "fpr") for c in tested if c["neg"]]
        ppv = [_rate(c, "ppv") for c in tested if c["pred_pos"]]

        def span(v: list[float]) -> float:
            return (max(v) - min(v)) if len(v) >= 2 else math.nan

        return {
            "Demographic parity difference": span(sel),
            "Demographic parity ratio": (min(sel) / max(sel)) if sel and max(sel) > 0 else math.nan,
            "Equalized odds difference": max(span(tpr), span(fpr)),
            "Equal opportunity difference": span(tpr),
            "Predictive parity difference": span(ppv),
        }

    point = gaps_of(counts)
    rng = np.random.default_rng(random_state)
    members = [np.flatnonzero(codes == g) for g in range(names.size)]
    boot: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_resamples):
        cs = []
        for idx in members:
            pick = idx[rng.integers(0, idx.size, idx.size)]
            cs.append(_counts(y[pick], p[pick]))
        for k, v in gaps_of(cs).items():
            boot[k].append(v)
    a = (1 - level) / 2
    gaps = []
    for k, v in point.items():
        b = np.asarray(boot[k], dtype=float)
        b = b[~np.isnan(b)]
        lo, hi = np.quantile(b, [a, 1 - a]) if b.size else (math.nan, math.nan)
        gaps.append({"name": k, "value": float(v), "low": float(lo), "high": float(hi)})

    tests: list[dict[str, Any]] = []
    rc = counts[ref_code]
    for g, name in enumerate(names):
        if g == ref_code or not rows[g]["tested"]:
            continue
        c = counts[g]
        for k in rates:
            _, num, den = _RATES[k]
            if not c[den] or not rc[den]:
                continue
            table = [[c[num], c[den] - c[num]], [rc[num], rc[den] - rc[num]]]
            res = fisher_exact_test(table)
            tests.append(
                {
                    "group": str(name),
                    "metric": k,
                    "difference": _rate(c, k) - _rate(rc, k),
                    "p_value": float(res.p_value),
                }
            )
    if tests:
        adj = np.atleast_1d(adjust_pvalues([t["p_value"] for t in tests], method="holm"))
        for t, pa in zip(tests, adj):
            t["p_adjusted"] = float(pa)
            t["significant"] = bool(pa < alpha)
    return FairnessReport(
        groups=tuple(rows),
        gaps=tuple(gaps),
        tests=tuple(tests),
        settings={
            "reference": str(names[ref_code]),
            "attributes": attrs,
            "level": level,
            "alpha": alpha,
            "min_group_size": min_group_size,
            "n_resamples": n_resamples,
            "random_state": random_state,
            "threshold": threshold if y_pred is None else None,
        },
    )
