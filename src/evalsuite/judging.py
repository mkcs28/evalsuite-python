"""LLM-as-judge evaluation in one report: who wins, and whether the judge can be trusted.

``judge_report(records)`` takes pairwise judgements (one record per comparison) and returns:

* **Win rates** of every model with Wilson intervals, the pairwise win matrix and Bradley–Terry ratings with
  bootstrap intervals (comparisons resampled).
* **Position bias**: how often the verdict survives swapping the order (``verdict_swapped``), and whether the
  inconsistent verdicts favour the response shown first (two-sided binomial test).
* **Length bias**: how often the longer response wins decisive comparisons (binomial test against 0.5).
* **Self-preference**: when the judge is itself one of the models (``judge``), how much more often it picks its
  own outputs than other judges or humans do on the same comparisons.
* **Agreement with humans** (``human`` verdicts): raw agreement (Wilson) and Cohen's kappa (bootstrap interval).

``rubric_report(records)`` summarises rubric scores (one record per item × criterion × model, optionally per
rater): the mean per model and criterion with t intervals, inter-rater reliability (Krippendorff's alpha) and
correlation with human scores.

Every bias is reported with its test and is flagged only when significant, so a report does not cry wolf on
small samples.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError


class _LazyModule:
    """Imports a SciPy submodule on first use, so ``import evalsuite`` stays free of SciPy."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        import importlib

        return getattr(importlib.import_module(self._name), attr)


stats = _LazyModule("scipy.stats")

__all__ = ["JudgeReport", "RubricReport", "judge_report", "rubric_report"]


def _verdict(v: Any) -> str:
    s = str(v).strip().lower()
    if s in ("a", "1", "first", "model_a"):
        return "a"
    if s in ("b", "2", "second", "model_b"):
        return "b"
    if s in ("tie", "draw", "t", "0.5", "equal"):
        return "tie"
    raise InputValidationError(f"Verdicts must be 'A', 'B' or 'tie'; got {v!r}.")


def _records(records: Any) -> list[dict[str, Any]]:
    if hasattr(records, "to_dict") and hasattr(records, "columns"):
        rows = records.to_dict("records")
    elif isinstance(records, str):
        with open(records, encoding="utf-8") as fh:
            text = fh.read()
        rows = (
            json.loads(text)
            if text.lstrip().startswith("[")
            else [json.loads(x) for x in text.splitlines() if x.strip()]
        )
    else:
        rows = list(records)
    if not rows:
        raise InputValidationError("No records.")
    for i, r in enumerate(rows):
        if not isinstance(r, Mapping):
            raise InputValidationError(f"Record {i} is not a mapping.")
    return [dict(r) for r in rows]


def _finite(value: Any, what: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise InputValidationError(f"{what} must be a number; got {value!r}.") from None
    if not math.isfinite(x):
        raise InputValidationError(f"{what} must be finite; got {value!r}.")
    return x


def _wilson(k: float, n: float, level: float) -> tuple[float, float]:
    from . import proportion_ci

    if n <= 0:
        return math.nan, math.nan
    ci = proportion_ci(round(k), round(n), level=level)
    return ci.low, ci.high


def _binom(k: int, n: int) -> float:
    return float(stats.binomtest(k, n, 0.5).pvalue) if n else math.nan


@dataclass(frozen=True, eq=False)
class JudgeReport:
    """Result of :func:`judge_report`."""

    models: tuple[dict[str, Any], ...]
    win_matrix: Mapping[str, Mapping[str, float]]
    checks: Mapping[str, Mapping[str, Any]]
    settings: Mapping[str, Any]

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        pct = f"{self.settings['level'] * 100:g}%"
        n = self.settings["n"]
        lines = [f"EvalSuite judge report ({n} comparison{'s' * (n != 1)}, {pct} CIs)"]
        lines.append("  Model                 Win rate             Bradley–Terry (log strength)")
        for m in self.models:
            bt = m.get("bt")
            bt_txt = "" if bt is None else f"{bt:+.{d}f} ({m['bt_low']:+.{d}f}–{m['bt_high']:+.{d}f})"
            lines.append(
                f"  {m['model']:<20}  {m['win_rate']:.{d}f} ({m['win_low']:.{d}f}–{m['win_high']:.{d}f})  {bt_txt}"
            )
        for c in self.checks.values():
            flag = " ⚠" if c.get("flag") else ""
            lines.append(f"  {c['label']}: {c['text']}{flag}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "models": list(self.models),
                "win_matrix": {k: dict(v) for k, v in self.win_matrix.items()},
                "checks": {k: dict(v) for k, v in self.checks.items()},
                "settings": dict(self.settings),
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        pct = f"{self.settings['level'] * 100:g}%"
        rows = [f"| Model | Win rate ({pct} CI) | Bradley–Terry ({pct} CI) |", "| --- | --- | --- |"]
        for m in self.models:
            bt = "" if m.get("bt") is None else f"{m['bt']:+.{d}f} ({m['bt_low']:+.{d}f}–{m['bt_high']:+.{d}f})"
            rows.append(
                f"| {m['model']} | {m['win_rate']:.{d}f} ({m['win_low']:.{d}f}–{m['win_high']:.{d}f}) | {bt} |"
            )
        rows += ["", "| Check | Result |", "| --- | --- |"]
        rows += [f"| {c['label']} | {c['text']}{' ⚠' if c.get('flag') else ''} |" for c in self.checks.values()]
        return "\n".join(rows) + "\n"


def judge_report(
    records: Any,
    *,
    level: float = 0.95,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_state: Optional[int] = 0,
) -> JudgeReport:
    """Pairwise LLM-judge evaluation. Each record: ``model_a``, ``model_b``, ``verdict`` (``"A"``, ``"B"`` or
    ``"tie"``, naming the response by identity) and optionally ``verdict_swapped`` (the verdict when the two
    responses were shown in the other order, again by identity), ``length_a`` / ``length_b`` (or
    ``response_a`` / ``response_b`` to measure them), ``judge`` (the judge model's name) and ``human`` (a human
    verdict on the same comparison)."""
    rows = _records(records)
    if not 0 < level < 1 or not 0 < alpha < 1:
        raise InputValidationError("level and alpha must be between 0 and 1.")
    comps: list[tuple[str, str, float]] = []
    for i, r in enumerate(rows):
        if "model_a" not in r or "model_b" not in r or "verdict" not in r:
            raise InputValidationError(f"Record {i} needs model_a, model_b and verdict.")
        v = _verdict(r["verdict"])
        comps.append((str(r["model_a"]), str(r["model_b"]), {"a": 1.0, "b": 0.0, "tie": 0.5}[v]))
    n = len(comps)
    names = sorted({x for a, b, _ in comps for x in (a, b)})
    if len(names) < 2:
        raise InputValidationError("judge_report needs comparisons between at least two models.")

    # win rates (ties half) and the win matrix
    won: dict[str, float] = defaultdict(float)
    games: dict[str, int] = defaultdict(int)
    mat: dict[str, dict[str, float]] = {a: {b: 0.0 for b in names if b != a} for a in names}
    played: dict[str, dict[str, int]] = {a: {b: 0 for b in names if b != a} for a in names}
    for a, b, o in comps:
        won[a] += o
        won[b] += 1 - o
        games[a] += 1
        games[b] += 1
        mat[a][b] += o
        mat[b][a] += 1 - o
        played[a][b] += 1
        played[b][a] += 1
    win_matrix = {a: {b: (mat[a][b] / played[a][b] if played[a][b] else math.nan) for b in mat[a]} for a in names}

    # Bradley–Terry with bootstrap over comparisons
    from . import bradley_terry

    def bt(cs: list[tuple[str, str, float]]) -> Optional[dict[str, float]]:
        rows_bt = [(a, b, "win" if o == 1 else "loss" if o == 0 else "tie") for a, b, o in cs]
        try:
            r = bradley_terry(rows_bt)
        except Exception:
            return None
        return dict(zip(r.labels or (), np.atleast_1d(r.value).tolist()))

    point = bt(comps)
    rng = np.random.default_rng(random_state)
    boots: dict[str, list[float]] = defaultdict(list)
    if point is not None:
        from .text.judge import _bt_fit

        # Resamples as win matrices: one weighted count per resample instead of re-parsing the comparisons.
        pos = {m: k for k, m in enumerate(names)}
        mm = len(names)
        ia = np.array([pos[a] for a, _, _ in comps])
        ib = np.array([pos[b] for _, b, _ in comps])
        oc = np.array([o for _, _, o in comps])
        arr = np.arange(n)
        for _ in range(n_resamples):
            idx = rng.choice(arr, n)
            a_i, b_i, o_i = ia[idx], ib[idx], oc[idx]
            wins = np.bincount(a_i * mm + b_i, weights=o_i, minlength=mm * mm) + np.bincount(
                b_i * mm + a_i, weights=1.0 - o_i, minlength=mm * mm
            )
            present = np.flatnonzero(np.bincount(np.r_[a_i, b_i], minlength=mm))
            sub = wins.reshape(mm, mm)[np.ix_(present, present)]
            try:
                logp = _bt_fit(sub)
            except Exception:  # noqa: S112 - an undefined fit on this resample is skipped, as before
                continue
            for k, v in zip(present, logp.tolist()):
                boots[names[k]].append(v)
    a_ = (1 - level) / 2
    models = []
    for m in names:
        lo, hi = _wilson(won[m], games[m], level)
        row: dict[str, Any] = {
            "model": m,
            "win_rate": won[m] / games[m],
            "win_low": lo,
            "win_high": hi,
            "comparisons": games[m],
        }
        if point is not None and m in point and len(boots[m]) >= 10:
            bl, bh = np.quantile(boots[m], [a_, 1 - a_])
            row.update(bt=float(point[m]), bt_low=float(bl), bt_high=float(bh))
        models.append(row)
    models.sort(key=lambda r: -r["win_rate"])

    checks: dict[str, dict[str, Any]] = {}
    # position bias
    sw = [
        (i, _verdict(r["verdict"]), _verdict(r["verdict_swapped"]))
        for i, r in enumerate(rows)
        if r.get("verdict_swapped") is not None
    ]
    if sw:
        same = sum(o == s for _, o, s in sw)
        lo, hi = _wilson(same, len(sw), level)
        inc = [(o, s) for _, o, s in sw if o != s]
        # original order shows A first; swapped shows B first. Favouring the first-shown both times = (a, b).
        first = sum(o == "a" and s == "b" for o, s in inc)
        second = sum(o == "b" and s == "a" for o, s in inc)
        p = _binom(first, first + second)
        checks["position"] = {
            "label": "Position consistency",
            "value": same / len(sw),
            "low": lo,
            "high": hi,
            "n": len(sw),
            "first_position_wins": first,
            "second_position_wins": second,
            "p_value": p,
            "flag": bool(p < alpha),
            "text": f"{same / len(sw):.3f} ({lo:.3f}–{hi:.3f}) of {len(sw)} verdicts survive swapping; "
            f"inconsistent ones favour the first-shown response {first}:{second} (p={p:.3g})",
        }
    # length bias
    la, lb = [], []
    for r in rows:
        x = r.get("length_a", len(str(r["response_a"]).split()) if r.get("response_a") is not None else None)
        y = r.get("length_b", len(str(r["response_b"]).split()) if r.get("response_b") is not None else None)
        la.append(x)
        lb.append(y)
    dec = [
        ((_verdict(r["verdict"]) == "a") == (float(x) > float(y)))
        for r, x, y in zip(rows, la, lb)
        if x is not None and y is not None and float(x) != float(y) and _verdict(r["verdict"]) != "tie"
    ]
    if dec:
        kk, m_ = int(sum(dec)), len(dec)
        lo, hi = _wilson(kk, m_, level)
        p = _binom(kk, m_)
        checks["length"] = {
            "label": "Length bias",
            "value": kk / m_,
            "low": lo,
            "high": hi,
            "n": m_,
            "p_value": p,
            "flag": bool(p < alpha),
            "text": f"the longer response wins {kk / m_:.3f} ({lo:.3f}–{hi:.3f}) of {m_} decisive comparisons "
            f"(p={p:.3g} against 0.5)",
        }
    # self-preference: judge picks its own model vs other judges / humans on comparisons involving that model
    own, other = [], []
    for r in rows:
        j = r.get("judge")
        if j is None:
            continue
        a, b, v = str(r["model_a"]), str(r["model_b"]), _verdict(r["verdict"])
        if v == "tie":
            continue
        for mdl, side in ((a, "a"), (b, "b")):
            if str(j) == mdl:
                own.append(v == side)
            elif mdl in {str(x.get("judge")) for x in rows if x.get("judge") is not None}:
                other.append(v == side)
    if own and other:
        from . import fisher_exact_test

        ko, kt = int(sum(own)), int(sum(other))
        res = fisher_exact_test([[ko, len(own) - ko], [kt, len(other) - kt]])
        diff = ko / len(own) - kt / len(other)
        checks["self_preference"] = {
            "label": "Self-preference",
            "value": diff,
            "own_rate": ko / len(own),
            "other_rate": kt / len(other),
            "n_own": len(own),
            "n_other": len(other),
            "p_value": float(res.p_value),
            "flag": bool(res.p_value < alpha and diff > 0),
            "text": f"judges pick their own model's response {ko / len(own):.3f} vs {kt / len(other):.3f} when "
            f"other judges see it ({diff:+.3f}, Fisher p={res.p_value:.3g})",
        }
    # agreement with humans
    hv = [(_verdict(r["verdict"]), _verdict(r["human"])) for r in rows if r.get("human") is not None]
    if hv:
        from . import cohen_kappa

        jl, hl = [x for x, _ in hv], [y for _, y in hv]
        agree = sum(x == y for x, y in hv)
        lo, hi = _wilson(agree, len(hv), level)
        try:
            kappa = float(cohen_kappa(hl, jl))
        except Exception:
            kappa = math.nan
        kb = []
        idx = np.arange(len(hv))
        for _ in range(n_resamples):
            s = rng.choice(idx, len(hv))
            with contextlib.suppress(Exception):  # a resample with one class only: no kappa
                kb.append(float(cohen_kappa([hl[i] for i in s], [jl[i] for i in s])))
        kb_arr = np.asarray(kb)
        kb_arr = kb_arr[np.isfinite(kb_arr)]
        klo, khi = np.quantile(kb_arr, [a_, 1 - a_]) if kb_arr.size else (math.nan, math.nan)
        checks["human"] = {
            "label": "Agreement with humans",
            "agreement": agree / len(hv),
            "low": lo,
            "high": hi,
            "kappa": kappa,
            "kappa_low": float(klo),
            "kappa_high": float(khi),
            "n": len(hv),
            "flag": bool(kappa < 0.4) if math.isfinite(kappa) else False,
            "text": f"{agree / len(hv):.3f} ({lo:.3f}–{hi:.3f}) raw agreement, Cohen's kappa {kappa:.3f} "
            f"({klo:.3f}–{khi:.3f}) on {len(hv)} comparison{'s' * (len(hv) != 1)}",
        }
    return JudgeReport(
        models=tuple(models),
        win_matrix=win_matrix,
        checks=checks,
        settings={
            "n": n,
            "level": level,
            "alpha": alpha,
            "n_resamples": n_resamples,
            "random_state": random_state,
        },
    )


@dataclass(frozen=True, eq=False)
class RubricReport:
    """Result of :func:`rubric_report`."""

    scores: tuple[dict[str, Any], ...]
    reliability: Mapping[str, Any]
    human: Mapping[str, Any]
    settings: Mapping[str, Any]

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        lines = [f"EvalSuite rubric report ({self.settings['level'] * 100:g}% CIs)"]
        for s in self.scores:
            lines.append(
                f"  {s['model']:<18} {s['criterion']:<18} {s['mean']:.{d}f} "
                f"({s['low']:.{d}f}–{s['high']:.{d}f})  n={s['n']}"
            )
        if self.reliability:
            lines.append(
                f"  Inter-rater reliability (Krippendorff's alpha, {self.reliability['level']}): "
                f"{self.reliability['alpha']:.{d}f}"
            )
        if self.human:
            lines.append(
                f"  Correlation with human scores: Spearman {self.human['spearman']:.{d}f} "
                f"({self.human['low']:.{d}f}–{self.human['high']:.{d}f}), n={self.human['n']}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_html(self, *, digits: int = 3) -> str:
        """The Markdown tables as an HTML fragment (escaped), e.g. for :func:`evalsuite.html_report`."""
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Model | Criterion | Mean (CI) | n |", "| --- | --- | --- | ---: |"]
        rows += [
            f"| {s['model']} | {s['criterion']} | {s['mean']:.{d}f} "
            f"({s['low']:.{d}f}–{s['high']:.{d}f}) | {s['n']} |"
            for s in self.scores
        ]
        return "\n".join(rows) + "\n"

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "scores": list(self.scores),
                "reliability": dict(self.reliability),
                "human": dict(self.human),
                "settings": dict(self.settings),
            }
        )
        return out


def rubric_report(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    level: float = 0.95,
    scale: str = "interval",
) -> RubricReport:
    """Rubric scores: records with ``id`` (item), ``model``, ``criterion``, ``score`` and optionally ``rater``
    (several raters per item and criterion give inter-rater reliability; ``scale`` is Krippendorff's level) and
    ``human`` (a human score for the same item, model and criterion)."""
    rows = _records(records)
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_item: dict[tuple[Any, str, str], list[float]] = defaultdict(list)
    raters: dict[tuple[Any, str, str], dict[str, float]] = defaultdict(dict)
    pairs: list[tuple[float, float]] = []
    for i, r in enumerate(rows):
        for f in ("id", "model", "criterion", "score"):
            if f not in r:
                raise InputValidationError(f"Record {i} needs id, model, criterion and score.")
        s = _finite(r["score"], f"Record {i} score")
        key = (r["id"], str(r["model"]), str(r["criterion"]))
        per_item[key].append(s)
        if r.get("rater") is not None:
            raters[key][str(r["rater"])] = s
        if r.get("human") is not None:
            pairs.append((s, _finite(r["human"], f"Record {i} human score")))
    for (_, model, crit), vals in per_item.items():
        groups[(model, crit)].append(float(np.mean(vals)))  # average raters within an item
    out = []
    for (model, crit), vals in sorted(groups.items()):
        v = np.asarray(vals)
        m = float(v.mean())
        if v.size >= 2:
            half = float(stats.t.ppf(1 - (1 - level) / 2, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size))
        else:
            half = math.nan
        out.append(
            {"model": model, "criterion": crit, "mean": m, "low": m - half, "high": m + half, "n": int(v.size)}
        )
    reliability: dict[str, Any] = {}
    all_raters = sorted({k for d in raters.values() for k in d})
    if len(all_raters) >= 2:
        from . import krippendorff_alpha

        units = list(raters)
        mat = np.full((len(all_raters), len(units)), np.nan)
        for j, u in enumerate(units):
            for k, rname in enumerate(all_raters):
                if rname in raters[u]:
                    mat[k, j] = raters[u][rname]
        try:
            reliability = {
                "alpha": float(krippendorff_alpha(mat, level=scale)),
                "level": scale,
                "raters": len(all_raters),
                "units": len(units),
            }
        except Exception as exc:  # e.g. no variation
            reliability = {"alpha": math.nan, "level": scale, "note": str(exc)}
    human: dict[str, Any] = {}
    if len(pairs) >= 4:
        a, b = np.asarray(pairs).T
        rho = float(stats.spearmanr(a, b).statistic)
        z, se = np.arctanh(np.clip(rho, -0.999999, 0.999999)), 1 / math.sqrt(len(pairs) - 3)
        q = stats.norm.ppf(1 - (1 - level) / 2)
        human = {
            "spearman": rho,
            "low": float(np.tanh(z - q * se)),
            "high": float(np.tanh(z + q * se)),
            "n": len(pairs),
        }
    return RubricReport(scores=tuple(out), reliability=reliability, human=human, settings={"level": level})
