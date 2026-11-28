"""Run language models and evaluate LLM systems: cached model calls, agent runs, conversations, quality per cost,
and benchmark contamination.

>>> import evalsuite as es
>>> runs = es.run_model(prompts, "openai:gpt-4o-mini")         # or "anthropic:claude-…", or any callable
>>> es.agent_report(episodes)                                  # success, steps, tool use, cost, with CIs
>>> es.conversation_report(dialogues)                          # per-turn and per-dialogue scores
>>> es.cost_quality_report({"a": runs_a, "b": runs_b})          # quality, cost and latency per model
>>> es.contamination_check(benchmark_items, training_texts)     # n-gram overlap with training data

``run_model`` calls the provider's official SDK (``pip install openai`` / ``anthropic``; keys come from the usual
environment variables and are never stored) or any Python callable, with retries, a small thread pool and an
on-disk cache in EvalSuite's cache folder (``evalsuite cache info``), so re-running an evaluation costs nothing.
Responses record latency and token counts for cost reports. Intervals in the reports are Wilson intervals for
rates and cluster-bootstrap percentile intervals (by dialogue / episode) for means.
"""

from __future__ import annotations

# ruff: noqa: E501  (long f-strings of report text)
import contextlib
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np

from .core.exceptions import InputValidationError, OptionalDependencyError

__all__ = [
    "run_model",
    "agent_report",
    "AgentReport",
    "conversation_report",
    "ConversationReport",
    "cost_quality_report",
    "CostQualityReport",
    "contamination_check",
    "ContaminationReport",
]


# ------------------------------------------------------------------------------------------------ model runs


def _cache_path(key: str) -> Any:
    from .cache import cache_dir

    folder = cache_dir(create=True) / "llm" / key[:2]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{key}.json"


def _call_openai(model: str, prompt: str, system: Optional[str], params: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import openai  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise OptionalDependencyError("openai", "openai", "run_model") from exc
    client = openai.OpenAI()
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(model=model, messages=messages, **params)
    usage = getattr(resp, "usage", None)
    return {
        "output": resp.choices[0].message.content or "",
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
    }


def _call_anthropic(model: str, prompt: str, system: Optional[str], params: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import anthropic  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise OptionalDependencyError("anthropic", "anthropic", "run_model") from exc
    client = anthropic.Anthropic()
    kw = dict(params)
    kw.setdefault("max_tokens", 1024)
    if system:
        kw["system"] = system
    resp = client.messages.create(model=model, messages=[{"role": "user", "content": prompt}], **kw)
    text = "".join(getattr(b, "text", "") for b in resp.content)
    usage = getattr(resp, "usage", None)
    return {
        "output": text,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }


def run_model(
    prompts: Sequence[Any],
    model: Union[str, Callable[[str], Any]],
    *,
    system: Optional[str] = None,
    cache: bool = True,
    concurrency: int = 4,
    retries: int = 3,
    ids: Optional[Sequence[Any]] = None,
    **params: Any,
) -> list[dict[str, Any]]:
    """Run ``model`` on every prompt and return one record per prompt: ``id``, ``prompt``, ``prediction`` (the
    output text), ``latency_s``, ``input_tokens``, ``output_tokens`` and ``cached``. ``model`` is
    ``"openai:<model>"``, ``"anthropic:<model>"``, or a callable taking a prompt and returning text (or a dict
    with ``output`` and optional token counts). ``prompts`` may be strings or records with ``prompt`` (and ``id``)
    such as :func:`load_benchmark` returns. ``params`` (``temperature``, ``max_tokens``, …) go to the provider and
    are part of the cache key."""
    items = []
    for i, p in enumerate(prompts):
        if isinstance(p, Mapping):
            if "prompt" not in p:
                raise InputValidationError(f"Record {i} has no 'prompt'.")
            items.append((p.get("id", i if ids is None else ids[i]), str(p["prompt"])))
        else:
            items.append((i if ids is None else ids[i], str(p)))
    if isinstance(model, str):
        provider, _, name = model.partition(":")
        if provider not in ("openai", "anthropic") or not name:
            raise InputValidationError("model must be 'openai:<name>', 'anthropic:<name>' or a callable.")
        caller: Callable[[str], dict[str, Any]] = (
            (lambda pr: _call_openai(name, pr, system, params))
            if provider == "openai"
            else (lambda pr: _call_anthropic(name, pr, system, params))
        )
        label = model
    elif callable(model):
        fn = model
        label = getattr(model, "__qualname__", "callable")
        cache = cache and getattr(model, "cacheable", False)  # only cache callables that opt in

        def caller(pr: str) -> dict[str, Any]:
            out = fn(pr)
            return dict(out) if isinstance(out, Mapping) else {"output": str(out)}
    else:
        raise InputValidationError("model must be a provider string or a callable.")
    if concurrency < 1 or retries < 0:
        raise InputValidationError("concurrency must be ≥ 1 and retries ≥ 0.")

    def one(item: tuple[Any, str]) -> dict[str, Any]:
        rid, prompt = item
        key = hashlib.sha256(
            json.dumps([label, system, prompt, sorted(params.items())], default=str).encode()
        ).hexdigest()
        path = _cache_path(key) if cache else None
        if path is not None and path.exists():
            try:
                rec = json.loads(path.read_text())
                rec.update(id=rid, cached=True)
                return dict(rec)
            except (OSError, ValueError):
                pass
        last: Optional[BaseException] = None
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                out = dict(caller(prompt))
                break
            except (OptionalDependencyError, InputValidationError):
                raise
            except Exception as exc:
                last = exc
                if attempt < retries:
                    time.sleep(min(30.0, 0.5 * 2**attempt))
        else:
            return {
                "id": rid,
                "prompt": prompt,
                "prediction": None,
                "error": f"{type(last).__name__}: {last}",
                "cached": False,
            }
        rec = {
            "id": rid,
            "prompt": prompt,
            "prediction": str(out.get("output", "")),
            "latency_s": time.perf_counter() - t0,
            "input_tokens": out.get("input_tokens"),
            "output_tokens": out.get("output_tokens"),
            "model": label,
            "cached": False,
        }
        if path is not None:
            with contextlib.suppress(OSError):
                path.write_text(json.dumps(rec))
        return rec

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, items))


# ------------------------------------------------------------------------------------------------ helpers


def _wilson(k: int, n: int, level: float) -> tuple[float, float]:
    from .stats.intervals import proportion_ci

    if n == 0:
        return (math.nan, math.nan)
    ci = proportion_ci(k, n, level=level)
    return float(ci.low), float(ci.high)


def _mean_ci(
    x: np.ndarray, level: float, rng: np.random.Generator, n_resamples: int = 2000
) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (math.nan, math.nan)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    m = x[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


class _Report:
    rows: Any
    settings: Any

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe({"rows": list(self.rows), "settings": dict(self.settings)})
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        keys = [k for k in self.rows[0] if not k.endswith(("_low", "_high"))]
        head = "| " + " | ".join(keys) + " |"
        rows = [head, "| " + " | ".join("---" for _ in keys) + " |"]
        for r in self.rows:
            cells = []
            for k in keys:
                v = r[k]
                if isinstance(v, float):
                    cell = f"{v:.{d}g}" if v and abs(v) < 0.01 else f"{v:.{d}f}"
                    if f"{k}_low" in r and r[f"{k}_low"] is not None and math.isfinite(r[f"{k}_low"]):
                        cell += f" ({r[f'{k}_low']:.{d}f}–{r[f'{k}_high']:.{d}f})"
                else:
                    cell = str(v)
                cells.append(cell)
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))

    def summary(self, *, digits: int = 3) -> str:
        return self.to_markdown(digits=digits)

    def __repr__(self) -> str:
        return self.summary()


# ------------------------------------------------------------------------------------------------ agents


@dataclass(frozen=True, repr=False)
class AgentReport(_Report):
    """Agent evaluation per system: success rate, steps, tool use, cost and latency with intervals."""

    rows: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        lines = [f"Agent report ({self.settings['level'] * 100:g}% CIs)"]
        for r in self.rows:
            line = f"  {r['system']:<16} success {r['success_rate']:.{d}f} ({r['success_rate_low']:.{d}f}–{r['success_rate_high']:.{d}f}) on {r['episodes']} episodes"
            if "mean_steps" in r:
                line += f"; steps {r['mean_steps']:.1f}"
            if "cost_per_success" in r:
                line += f"; cost per success {r['cost_per_success']:.4g}"
            lines.append(line)
        return "\n".join(lines)


def agent_report(
    episodes: Union[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    level: float = 0.95,
    random_state: Optional[int] = 0,
) -> AgentReport:
    """Summarise agent episodes, one system (a list of episodes) or several (mapping name → list). Each episode
    is a dict with ``success`` (bool) and optionally ``steps`` (int), ``tool_calls`` / ``expected_tool_calls``
    (lists of ``{"name", "arguments"}`` for tool-call F1), ``invalid_tool_calls`` (int), ``cost`` and
    ``latency_s``."""
    from .text.structured import tool_call_f1

    systems = episodes if isinstance(episodes, Mapping) else {"agent": episodes}
    rng = np.random.default_rng(random_state)
    rows = []
    for name, eps in systems.items():
        eps = list(eps)
        if not eps:
            raise InputValidationError(f"System {name!r} has no episodes.")
        if any("success" not in e for e in eps):
            raise InputValidationError("Every episode needs a 'success' field.")
        succ = np.array([bool(e["success"]) for e in eps])
        lo, hi = _wilson(int(succ.sum()), succ.size, level)
        row: dict[str, Any] = {
            "system": str(name),
            "episodes": int(succ.size),
            "success_rate": float(succ.mean()),
            "success_rate_low": lo,
            "success_rate_high": hi,
        }
        if all("steps" in e for e in eps):
            steps = np.array([float(e["steps"]) for e in eps])
            row["mean_steps"] = float(steps.mean())
            row["mean_steps_low"], row["mean_steps_high"] = _mean_ci(steps, level, rng)
            row["steps_successful"] = float(steps[succ].mean()) if succ.any() else math.nan
        with_tools = [e for e in eps if "tool_calls" in e and "expected_tool_calls" in e]
        if with_tools:
            row["tool_call_f1"] = float(
                tool_call_f1(
                    [e["expected_tool_calls"] for e in with_tools], [e["tool_calls"] for e in with_tools]
                ).value
            )
        if all("invalid_tool_calls" in e for e in eps):
            row["invalid_tool_calls_per_episode"] = float(np.mean([e["invalid_tool_calls"] for e in eps]))
        if all("cost" in e for e in eps):
            cost = np.array([float(e["cost"]) for e in eps])
            row["total_cost"] = float(cost.sum())
            row["cost_per_success"] = float(cost.sum()) / int(succ.sum()) if succ.any() else math.inf
        if all("latency_s" in e for e in eps):
            lat = np.array([float(e["latency_s"]) for e in eps])
            row["median_latency_s"] = float(np.median(lat))
        rows.append(row)
    if len(rows) > 1:
        from .stats.effect import adjust_pvalues

        base = rows[0]
        ps = []
        for r in rows[1:]:
            ps.append(_two_prop_p(r["success_rate"], r["episodes"], base["success_rate"], base["episodes"]))
        for r, p in zip(rows[1:], adjust_pvalues(ps, method="holm")):
            r["p_vs_first_holm"] = float(p)
    return AgentReport(tuple(rows), {"level": level, "systems": list(systems)})


def _two_prop_p(p1: float, n1: int, p2: float, n2: int) -> float:
    from scipy import stats

    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if pool in (0.0, 1.0):
        return 1.0
    z = (p1 - p2) / math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    return float(2 * stats.norm.sf(abs(z)))


# ------------------------------------------------------------------------------------------------ conversations


@dataclass(frozen=True, repr=False)
class ConversationReport(_Report):
    """Per-turn and per-dialogue scores with dialogue-level (cluster) bootstrap intervals."""

    rows: tuple[dict[str, Any], ...]
    by_turn: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        s = self.settings
        lines = [
            f"Conversation report: {s['dialogues']} dialogues, {s['turns']} turns ({s['level'] * 100:g}% CIs, resampling dialogues)"
        ]
        for r in self.rows:
            lines.append(
                f"  {r['measure']:<28} {r['value']:.{d}f} ({r['value_low']:.{d}f}–{r['value_high']:.{d}f})"
            )
        if self.by_turn:
            lines.append(
                "  By turn position: " + ", ".join(f"{t['turn']}: {t['mean']:.{d}f}" for t in self.by_turn[:12])
            )
        return "\n".join(lines)


def conversation_report(
    dialogues: Sequence[Mapping[str, Any]],
    *,
    level: float = 0.95,
    n_resamples: int = 2000,
    random_state: Optional[int] = 0,
) -> ConversationReport:
    """Evaluate multi-turn conversations. Each dialogue is a dict with ``turns``: a list of per-turn scores
    (numbers in [0, 1], e.g. judge or rubric scores of each assistant reply) or of dicts with ``score``; and
    optionally ``success`` (did the whole conversation achieve its goal). Reports the mean turn score, the share of
    dialogues with every turn acceptable (score ≥ 0.5), the worst-turn mean, goal success, and the mean score by
    turn position (does quality drift over long conversations?)."""
    if not dialogues:
        raise InputValidationError("No dialogues given.")
    turn_scores: list[np.ndarray] = []
    success = []
    for i, d in enumerate(dialogues):
        if "turns" not in d or not len(d["turns"]):
            raise InputValidationError(f"Dialogue {i} has no turns.")
        vals = [t["score"] if isinstance(t, Mapping) else t for t in d["turns"]]
        try:
            arr = np.asarray(vals, dtype=float)
        except (TypeError, ValueError):
            raise InputValidationError(f"Dialogue {i}: turn scores must be numbers.") from None
        if not np.all(np.isfinite(arr)):
            raise InputValidationError(f"Dialogue {i}: turn scores must be finite.")
        turn_scores.append(arr)
        if "success" in d:
            success.append(bool(d["success"]))
    rng = np.random.default_rng(random_state)
    nd = len(turn_scores)

    def stats_of(sel: np.ndarray) -> dict[str, float]:
        chosen = [turn_scores[j] for j in sel]
        allt = np.concatenate(chosen)
        out = {
            "Mean turn score": float(allt.mean()),
            "Mean of worst turn": float(np.mean([c.min() for c in chosen])),
            "All turns acceptable (≥ 0.5)": float(np.mean([np.all(c >= 0.5) for c in chosen])),
        }
        if len(success) == nd:
            out["Goal success"] = float(np.mean([success[j] for j in sel]))
        return out

    point = stats_of(np.arange(nd))
    draws = {k: np.empty(n_resamples) for k in point}
    for b in range(n_resamples):
        got = stats_of(rng.integers(0, nd, nd))
        for k in point:
            draws[k][b] = got[k]
    a = (1 - level) / 2
    rows = tuple(
        {
            "measure": k,
            "value": v,
            "value_low": float(np.quantile(draws[k], a)),
            "value_high": float(np.quantile(draws[k], 1 - a)),
        }
        for k, v in point.items()
    )
    longest = max(len(t) for t in turn_scores)
    by_turn = tuple(
        {
            "turn": j + 1,
            "mean": float(np.mean([t[j] for t in turn_scores if len(t) > j])),
            "n": sum(len(t) > j for t in turn_scores),
        }
        for j in range(longest)
    )
    return ConversationReport(
        rows, by_turn, {"dialogues": nd, "turns": int(sum(len(t) for t in turn_scores)), "level": level}
    )


# ------------------------------------------------------------------------------------------------ cost / quality


@dataclass(frozen=True, repr=False)
class CostQualityReport(_Report):
    """Quality, cost and latency per model, and quality per unit cost."""

    rows: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def pareto(self) -> list[str]:
        """Models not beaten on both quality and cost by another model."""
        out = []
        for r in self.rows:
            dominated = any(
                o["quality"] >= r["quality"]
                and o["cost_per_item"] <= r["cost_per_item"]
                and o is not r
                and (o["quality"] > r["quality"] or o["cost_per_item"] < r["cost_per_item"])
                for o in self.rows
            )
            if not dominated:
                out.append(r["model"])
        return out


def cost_quality_report(
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    prices: Optional[Mapping[str, tuple[float, float]]] = None,
    level: float = 0.95,
    random_state: Optional[int] = 0,
) -> CostQualityReport:
    """Per model: mean quality (each record's ``correct`` (bool) or ``score``), cost per item and latency, with
    intervals. Cost comes from each record's ``cost``, or from ``input_tokens`` / ``output_tokens`` and
    ``prices[model] = (price per 1M input tokens, price per 1M output tokens)``."""
    rng = np.random.default_rng(random_state)
    rows = []
    for model, recs in runs.items():
        recs = list(recs)
        if not recs:
            raise InputValidationError(f"Model {model!r} has no records.")
        if all("correct" in r for r in recs):
            q = np.array([float(bool(r["correct"])) for r in recs])
        elif all("score" in r for r in recs):
            q = np.array([float(r["score"]) for r in recs])
        else:
            raise InputValidationError(f"Records of {model!r} need 'correct' or 'score'.")
        if all("cost" in r for r in recs):
            cost = np.array([float(r["cost"]) for r in recs])
        elif prices and model in prices and all(r.get("input_tokens") is not None for r in recs):
            pin, pout = prices[model]
            cost = np.array([(r["input_tokens"] * pin + (r.get("output_tokens") or 0) * pout) / 1e6 for r in recs])
        else:
            cost = np.full(len(recs), math.nan)
        lat = np.array([float(r["latency_s"]) if r.get("latency_s") is not None else math.nan for r in recs])
        row: dict[str, Any] = {"model": str(model), "n": len(recs), "quality": float(q.mean())}
        row["quality_low"], row["quality_high"] = _mean_ci(q, level, rng)
        row["cost_per_item"] = float(np.nanmean(cost)) if np.isfinite(cost).any() else math.nan
        row["quality_per_dollar"] = (
            row["quality"] / row["cost_per_item"]
            if row["cost_per_item"] and math.isfinite(row["cost_per_item"])
            else math.nan
        )
        row["median_latency_s"] = float(np.nanmedian(lat)) if np.isfinite(lat).any() else math.nan
        row["p95_latency_s"] = float(np.nanpercentile(lat, 95)) if np.isfinite(lat).any() else math.nan
        rows.append(row)
    return CostQualityReport(tuple(rows), {"level": level})


# ------------------------------------------------------------------------------------------------ contamination


@dataclass(frozen=True, repr=False)
class ContaminationReport(_Report):
    """Benchmark items whose text overlaps a training corpus."""

    rows: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    @property
    def contaminated(self) -> list[Any]:
        return [r["id"] for r in self.rows if r["contaminated"]]

    def summary(self, *, digits: int = 3) -> str:
        s = self.settings
        k = len(self.contaminated)
        return (
            f"Contamination check: {k} of {len(self.rows)} benchmark items ({k / max(1, len(self.rows)):.1%}) share at "
            f"least {s['threshold']:.0%} of their {s['n']}-grams with the corpus "
            f"(Wilson {s['level'] * 100:g}% CI {s['rate_low']:.1%}–{s['rate_high']:.1%})."
        )

    def to_markdown(self, *, digits: int = 3) -> str:
        rows = ["| Item | Overlap | Contaminated |", "| --- | ---: | :---: |"]
        rows += [
            f"| {r['id']} | {r['overlap']:.{digits}f} | {'yes' if r['contaminated'] else 'no'} |"
            for r in self.rows
        ]
        return "\n".join(rows) + "\n"


def _ngrams(text: str, n: int) -> set[int]:
    words = re.findall(r"\w+", str(text).lower())
    return {hash(" ".join(words[i : i + n])) for i in range(max(0, len(words) - n + 1))}


def contamination_check(
    items: Sequence[Any],
    corpus: Union[Sequence[str], Any],
    *,
    n: int = 13,
    threshold: float = 0.5,
    level: float = 0.95,
) -> ContaminationReport:
    """Share of each benchmark item's word ``n``-grams that also occur in ``corpus`` (an iterable of training
    texts; streamed, so it can be large). Items are strings or records with ``prompt`` / ``question`` /
    ``text`` (and ``id``). 13-grams follow the GPT-3 / GPT-4 contamination analyses; items shorter than n words are
    checked as a whole. An item is flagged when the overlap is at least ``threshold``."""
    if n < 1 or not 0 < threshold <= 1:
        raise InputValidationError("n must be ≥ 1 and threshold in (0, 1].")
    texts, ids = [], []
    for i, it in enumerate(items):
        if isinstance(it, Mapping):
            t = it.get("prompt") or it.get("question") or it.get("text")
            if t is None:
                raise InputValidationError(f"Item {i} has no 'prompt', 'question' or 'text'.")
            texts.append(str(t))
            ids.append(it.get("id", i))
        else:
            texts.append(str(it))
            ids.append(i)
    grams = []
    for t in texts:
        k = min(n, max(1, len(re.findall(r"\w+", t.lower()))))
        grams.append((k, _ngrams(t, k)))
    needed: dict[int, set[int]] = {}
    for k, g in grams:
        needed.setdefault(k, set()).update(g)
    seen: dict[int, set[int]] = {k: set() for k in needed}
    for doc in corpus:
        for k, want in needed.items():
            seen[k].update(_ngrams(str(doc), k) & want)
    rows = []
    for rid, (k, g) in zip(ids, grams):
        overlap = len(g & seen[k]) / len(g) if g else 0.0
        rows.append({"id": rid, "overlap": float(overlap), "contaminated": bool(g) and overlap >= threshold})
    k_c = sum(r["contaminated"] for r in rows)
    lo, hi = _wilson(k_c, len(rows), level)
    return ContaminationReport(
        tuple(rows), {"n": n, "threshold": threshold, "level": level, "rate_low": lo, "rate_high": hi}
    )
