"""A statistically careful LLM evaluation harness.

Point it at each model's outputs (JSONL files, lists of records or DataFrames) and it scores them, puts an
interval on every number and only ranks models apart when the difference is significant:

* every model's score with a Wilson (or bootstrap) confidence interval;
* paired tests between models on the items they share (exact McNemar for correct / incorrect, a paired
  bootstrap for graded scores), with Holm-adjusted p-values;
* rank groups: a model's rank is 1 + the number of models significantly better than it, so statistical ties
  share a rank;
* per-category breakdowns with intervals;
* checks: unmatched or missing items, extraction failures, contamination (n-gram overlap of each item with a
  training corpus you provide, and scores on clean versus overlapping items), and judge position bias when the
  records carry verdicts in both presentation orders.

Each record is a mapping with an ``id`` and either a ``correct`` / ``score`` value from your own grader, or a
``prediction`` (model output) scored against the ``reference`` (gold answer) with ``scorer``. Optional fields:
``category``, ``prompt`` (for the contamination check), ``verdict`` and ``verdict_swapped`` (judge verdicts
"A" / "B" / "tie" with the two answers in the original and the swapped order).

>>> import evalsuite as es
>>> runs = {
...     "model-a": [{"id": i, "prediction": "4", "reference": "4"} for i in range(30)],
...     "model-b": [{"id": i, "prediction": "4" if i % 3 else "5", "reference": "4"} for i in range(30)],
... }
>>> lb = es.leaderboard(runs, scorer="exact_match")
>>> best = lb.rows[0]
>>> best.get("model"), best.get("rank")
('model-a', 1)
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from .core.exceptions import InputValidationError

Records = Union[str, "os.PathLike[str]", Iterable[Mapping[str, Any]], Any]

_SCORERS = ("exact_match", "contains", "gsm8k", "boxed", "choice", "last_line", "numeric")


def _load(src: Records, name: str) -> list[dict[str, Any]]:
    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        stripped = text.lstrip()
        if stripped.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif hasattr(src, "to_dict") and hasattr(src, "columns"):
        rows = src.to_dict("records")
    else:
        rows = list(src)
    out = []
    for i, r in enumerate(rows):
        if not isinstance(r, Mapping):
            raise InputValidationError(f"{name}: record {i} is not a mapping (one JSON object per line).")
        if "id" not in r:
            raise InputValidationError(f"{name}: record {i} has no 'id'.")
        out.append(dict(r))
    if not out:
        raise InputValidationError(f"{name} has no records.")
    return out


def _number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _scorer(name: Union[str, Callable[[str, str], Any]]) -> Callable[[str, str], tuple[float, bool]]:
    """(score in [0, 1], whether an answer could be extracted) for one prediction / reference pair."""
    from .text.qa import normalize_answer
    from .text.reasoning import extract_answer

    if callable(name):
        user = name

        def custom(pred: str, ref: str) -> tuple[float, bool]:
            v = float(user(pred, ref))
            if not (0.0 <= v <= 1.0):
                raise InputValidationError("A custom scorer must return a value in [0, 1].")
            return v, True

        return custom
    if name not in _SCORERS:
        raise InputValidationError(
            f"scorer must be one of {', '.join(_SCORERS)} or a callable (prediction, reference)."
        )
    if name == "exact_match":
        return lambda p, r: (float(normalize_answer(p) == normalize_answer(r)), True)
    if name == "contains":
        return lambda p, r: (float(normalize_answer(r) in normalize_answer(p)), True)
    if name == "numeric":

        def numeric(p: str, r: str) -> tuple[float, bool]:
            a = extract_answer(p, style="gsm8k")
            x, y = (_number(a) if a is not None else None), _number(r)
            if x is None or y is None:
                return 0.0, x is not None
            return float(math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-9)), True

        return numeric

    def extracted(p: str, r: str) -> tuple[float, bool]:
        a = extract_answer(p, style=name)
        gold = extract_answer(r, style=name) if ("####" in r or "\\boxed" in r) else r.strip()
        if a is None:
            return 0.0, False
        if name == "gsm8k" and gold is not None:
            ga, gb = _number(a), _number(gold)
            if ga is not None and gb is not None:
                return float(ga == gb), True
        return float(a.strip() == (gold or "").strip()), True

    return extracted


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = text.lower().split()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass(frozen=True, eq=False)
class Leaderboard:
    """Scores with intervals, paired tests, rank groups, per-category results and data checks."""

    rows: tuple[dict[str, Any], ...]
    tests: tuple[dict[str, Any], ...]
    categories: Mapping[str, tuple[dict[str, Any], ...]]
    checks: Mapping[str, Any]
    settings: Mapping[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def summary(self, *, digits: int = 3) -> str:
        pct = f"{self.settings['level'] * 100:g}%"
        width = max(len(r["model"]) for r in self.rows)
        lines = [
            f"EvalSuite leaderboard ({self.settings['n_items']} items, scorer={self.settings['scorer']}, "
            f"{pct} CIs; "
            f"ranks differ only when Holm-adjusted p < {self.settings['alpha']})",
            f"  {'Rank':<5}{'Model':<{width}}  {'Score':>7}  {pct + ' CI':<17}  n",
        ]
        for r in self.rows:
            lines.append(
                f"  {r['rank']:<5}{r['model']:<{width}}  {r['score']:>7.{digits}f}  "
                f"{r['ci_low']:.{digits}f}–{r['ci_high']:.{digits}f}{'':<4}  {r['n']}"
            )
        sig = [t for t in self.tests if t["significant"]]
        lines.append(f"  Paired tests: {len(sig)} of {len(self.tests)} differences significant.")
        for t in self.tests:
            p = "<0.001" if t["p_adjusted"] < 0.001 else f"={t['p_adjusted']:.3f}"
            mark = "*" if t["significant"] else " "
            lines.append(
                f"   {mark} {t['model_a']} − {t['model_b']}: {t['difference']:+.{digits}f} "
                f"(p{p}, {t['test']}, {t['n_shared']} shared items)"
            )
        for name, msg in self._warnings():
            lines.append(f"  ! {name}: {msg}")
        return "\n".join(lines)

    def _warnings(self) -> list[tuple[str, str]]:
        out = []
        c = self.checks
        if c.get("unmatched_items"):
            out.append(
                (
                    "items",
                    f"{c['unmatched_items']} item ids are not answered by every model; tests use shared items",
                )
            )
        for m, k in c.get("extraction_failures", {}).items():
            if k:
                out.append(
                    ("extraction", f"{m}: no answer could be extracted from {k} outputs (scored incorrect)")
                )
        cont = c.get("contamination")
        if cont and cont["contaminated_items"]:
            out.append(
                (
                    "contamination",
                    f"{cont['contaminated_items']} of {cont['n_items']} items overlap the training corpus "
                    f"({cont['ngram']}-grams); see checks['contamination'] for clean-subset scores",
                )
            )
        for m, v in c.get("position_consistency", {}).items():
            if v["value"] < 0.9:
                out.append(("judge", f"{m}: verdicts agree under swapped order only {v['value']:.0%} of the time"))
        return out

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "rows": list(self.rows),
                "tests": list(self.tests),
                "categories": {k: list(v) for k, v in self.categories.items()},
                "checks": dict(self.checks),
                "settings": dict(self.settings),
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_markdown(self, *, digits: int = 3) -> str:
        pct = f"{self.settings['level'] * 100:g}%"
        out = [f"| Rank | Model | Score | {pct} CI | n |", "| ---: | --- | ---: | --- | ---: |"]
        for r in self.rows:
            out.append(
                f"| {r['rank']} | {r['model']} | {r['score']:.{digits}f} | "
                f"{r['ci_low']:.{digits}f}–{r['ci_high']:.{digits}f} | {r['n']} |"
            )
        return "\n".join(out)

    def to_latex(self, *, digits: int = 3, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        from .core.result import _latex_escape, _latex_table

        pct = f"{self.settings['level'] * 100:g}\\%"
        rows = [
            [
                str(r["rank"]),
                _latex_escape(r["model"]),
                f"{r['score']:.{digits}f}",
                f"{r['ci_low']:.{digits}f}--{r['ci_high']:.{digits}f}",
                str(r["n"]),
            ]
            for r in self.rows
        ]
        return _latex_table(["Rank", "Model", "Score", f"{pct} CI", "n"], rows, caption=caption, label=label)

    def to_csv(self) -> str:
        from .core.export import csv_text

        return csv_text(
            ["rank", "model", "score", "ci_low", "ci_high", "n"],
            [[r["rank"], r["model"], r["score"], r["ci_low"], r["ci_high"], r["n"]] for r in self.rows],
        )

    def to_html(self, *, digits: int = 3, full: bool = False) -> str:
        import html

        body = (
            "<table><thead><tr><th>Rank</th><th>Model</th><th>Score</th><th>CI</th><th>n</th></tr></thead><tbody>"
        )
        for r in self.rows:
            body += (
                f"<tr><td>{r['rank']}</td><td>{html.escape(r['model'])}</td><td>{r['score']:.{digits}f}</td>"
                f"<td>{r['ci_low']:.{digits}f}–{r['ci_high']:.{digits}f}</td><td>{r['n']}</td></tr>"
            )
        body += "</tbody></table>"
        if not full:
            return body
        head = "<!doctype html><html><head><meta charset='utf-8'><title>EvalSuite leaderboard</title></head>"
        return f"{head}<body>{body}</body></html>"


def leaderboard(
    runs: Mapping[str, Records],
    *,
    scorer: Optional[Union[str, Callable[[str, str], Any]]] = None,
    references: Optional[Records] = None,
    benchmark: Optional[str] = None,
    level: float = 0.95,
    alpha: float = 0.05,
    correction: str = "holm",
    train_corpus: Optional[Iterable[str]] = None,
    ngram: int = 13,
    n_resamples: int = 2000,
    random_state: Optional[int] = 0,
) -> Leaderboard:
    """Score several models' outputs and rank them with intervals and paired tests (see the module docs).

    ``runs`` maps model names to records (a JSONL / JSON path, a list of dicts or a DataFrame). ``references``
    optionally supplies the gold answers separately (records with ``id`` and ``reference``). ``scorer`` is one
    of "exact_match" (SQuAD-normalised), "contains", "gsm8k", "boxed", "choice", "last_line", "numeric", or a
    callable (prediction, reference) -> score in [0, 1]; records with a ``correct`` or ``score`` field are used
    as given. ``train_corpus`` (iterable of texts) enables the contamination check.
    ``benchmark`` (``"gsm8k"``, ``"mmlu"``, ``"arc"``, ``"arc-easy"``, ``"truthfulqa"``, ``"humaneval"``) picks
    that
    benchmark's scorer (unless ``scorer`` is given); pass ``references=es.load_benchmark(name)`` for its gold
    answers and categories. For HumanEval, records carry ``passed`` (or ``n_samples`` / ``n_correct``) from your
    own sandboxed execution; the score is pass@1.
    """
    from . import adjust_pvalues, mcnemar_test, position_consistency, proportion_ci

    if not isinstance(runs, Mapping) or len(runs) < 1:
        raise InputValidationError("runs must map model names to their records.")
    if not 0 < level < 1 or not 0 < alpha < 1:
        raise InputValidationError("level and alpha must be in (0, 1).")
    bench = None
    if benchmark is not None:
        from .llm_benchmarks import spec

        bench = spec(benchmark)
    scorer_label: Optional[str] = None
    if scorer is None:
        if bench is not None and bench.scorer == "pass":
            scorer, scorer_label = "exact_match", "pass@1"
        else:
            scorer = bench.scorer if bench is not None else "exact_match"
    gold: dict[Any, Any] = {}
    gold_category: dict[Any, str] = {}
    if references is not None:
        for r in _load(references, "references"):
            gold[r["id"]] = r.get("reference", r.get("answer"))
            if r.get("category") is not None:
                gold_category[r["id"]] = str(r["category"])
    score_fn = _scorer(scorer)
    scores: dict[str, dict[Any, float]] = {}
    failures: dict[str, int] = {}
    meta: dict[Any, dict[str, Any]] = {}
    judge: dict[str, tuple[list[Any], list[Any]]] = {}
    binary = True
    for model, src in runs.items():
        recs = _load(src, f"runs[{model!r}]")
        s: dict[Any, float] = {}
        fails = 0
        orig, swapped = [], []
        for r in recs:
            key = r["id"]
            if key in s:
                raise InputValidationError(f"runs[{model!r}] has item id {key!r} more than once.")
            if "correct" in r and r["correct"] is not None:
                v = float(bool(r["correct"]))
            elif "passed" in r and r["passed"] is not None:
                v = float(bool(r["passed"]))
            elif r.get("n_samples") is not None and r.get("n_correct") is not None:
                ns, nc = int(r["n_samples"]), int(r["n_correct"])
                if ns < 1 or not 0 <= nc <= ns:
                    raise InputValidationError(
                        f"runs[{model!r}] item {key!r}: need 0 <= n_correct <= n_samples and n_samples >= 1."
                    )
                v = nc / ns  # unbiased pass@1
            elif "score" in r and r["score"] is not None:
                v = float(r["score"])
                if not 0.0 <= v <= 1.0:
                    raise InputValidationError(f"runs[{model!r}] item {key!r}: score must be in [0, 1].")
            else:
                ref = r.get("reference", r.get("answer", gold.get(key)))
                if ref is None or ("prediction" not in r and "output" not in r):
                    raise InputValidationError(
                        f"runs[{model!r}] item {key!r} needs 'correct', 'score', "
                        "or 'prediction' with a 'reference'."
                    )
                v, ok = score_fn(str(r.get("prediction", r.get("output", ""))), str(ref))
                fails += int(not ok)
            binary &= v in (0.0, 1.0)
            s[key] = v
            m = meta.setdefault(key, {})
            if r.get("category") is not None:
                m.setdefault("category", str(r["category"]))
            elif key in gold_category:
                m.setdefault("category", gold_category[key])
            for f in ("prompt", "reference"):
                if r.get(f) is not None:
                    m.setdefault(f, str(r[f]))
            if r.get("verdict") is not None and r.get("verdict_swapped") is not None:
                orig.append(r["verdict"])
                swapped.append(r["verdict_swapped"])
        scores[model] = s
        failures[model] = fails
        if orig:
            judge[model] = (orig, swapped)
    all_ids = set().union(*(s.keys() for s in scores.values()))
    shared = set.intersection(*(set(s) for s in scores.values()))
    rng = np.random.default_rng(random_state)

    def interval(vals: np.ndarray) -> tuple[float, float]:
        if binary:
            ci = proportion_ci(int(vals.sum()), int(vals.size), level=level)
            return ci.low, ci.high
        idx = rng.integers(0, vals.size, (n_resamples, vals.size))
        means = vals[idx].mean(axis=1)
        a = (1 - level) / 2
        lo, hi = np.quantile(means, [a, 1 - a])
        return float(lo), float(hi)

    rows: list[dict[str, Any]] = []
    for model, s in scores.items():
        arr = np.array(list(s.values()), dtype=float)
        lo, hi = interval(arr)
        rows.append({"model": model, "score": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n": int(arr.size)})
    models = list(scores)
    tests: list[dict[str, Any]] = []
    order = sorted(shared, key=str)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            xa = np.array([scores[a][k] for k in order])
            xb = np.array([scores[b][k] for k in order])
            if not order:
                continue
            if binary:
                mc = mcnemar_test(np.ones(len(order)), xa, xb)
                test, p = mc.test, mc.p_value
            else:
                d = xa - xb
                obs = float(d.mean())
                idx = rng.integers(0, d.size, (n_resamples, d.size))
                boot = d[idx].mean(axis=1)
                p = float((1 + np.sum(np.abs(boot - obs) >= abs(obs))) / (n_resamples + 1))
                test = "paired-bootstrap"
            tests.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "difference": float(xa.mean() - xb.mean()),
                    "test": test,
                    "p_value": float(p),
                    "n_shared": len(order),
                }
            )
    if tests:
        adj = adjust_pvalues([float(t["p_value"]) for t in tests], method=correction)  # type: ignore[arg-type]
        for t, p in zip(tests, adj):
            t["p_adjusted"] = float(p)
            t["significant"] = bool(p < alpha)
    better: dict[str, int] = dict.fromkeys(models, 0)
    for t in tests:
        if t["significant"]:
            loser = t["model_b"] if t["difference"] > 0 else t["model_a"]
            better[loser] += 1
    for r in rows:
        r["rank"] = 1 + better[r["model"]]
    rows.sort(key=lambda r: (r["rank"], -r["score"], r["model"]))
    categories: dict[str, list[dict[str, Any]]] = {}
    cats = sorted({m["category"] for m in meta.values() if "category" in m})
    for c in cats:
        block = []
        for model, s in scores.items():
            cv = np.array([x for k, x in s.items() if meta.get(k, {}).get("category") == c], dtype=float)
            if cv.size:
                lo, hi = interval(cv)
                block.append(
                    {"model": model, "score": float(cv.mean()), "ci_low": lo, "ci_high": hi, "n": int(cv.size)}
                )
        categories[c] = block
    checks: dict[str, Any] = {
        "n_models": len(models),
        "unmatched_items": len(all_ids - shared),
        "extraction_failures": failures,
    }
    if train_corpus is not None:
        corpus: set[tuple[str, ...]] = set()
        for text in train_corpus:
            corpus |= _ngrams(str(text), ngram)
        flagged = set()
        for k, m in meta.items():
            text = " ".join(m.get(f, "") for f in ("prompt", "reference"))
            if _ngrams(text, ngram) & corpus:
                flagged.add(k)
        per_model = {}
        for model, s in scores.items():
            clean = [x for k, x in s.items() if k not in flagged]
            dirty = [x for k, x in s.items() if k in flagged]
            per_model[model] = {
                "clean_score": float(np.mean(clean)) if clean else math.nan,
                "contaminated_score": float(np.mean(dirty)) if dirty else math.nan,
            }
        checks["contamination"] = {
            "ngram": ngram,
            "n_items": len(meta),
            "contaminated_items": len(flagged),
            "contaminated_ids": sorted(flagged, key=str),
            "by_model": per_model,
        }
    if judge:
        checks["position_consistency"] = {
            m: {"value": float(position_consistency(o, s).value), "n": len(o)} for m, (o, s) in judge.items()
        }
    return Leaderboard(
        rows=tuple(rows),
        tests=tuple(tests),
        categories={k: tuple(v) for k, v in categories.items()},
        checks=checks,
        settings={
            "scorer": scorer_label
            or (scorer if isinstance(scorer, str) else getattr(scorer, "__name__", "custom")),
            "benchmark": bench.name if bench is not None else None,
            "level": level,
            "alpha": alpha,
            "correction": correction,
            "n_items": len(all_ids),
            "binary": binary,
            "random_state": random_state,
        },
    )
