"""End-to-end RAG evaluation: retrieval, grounding and answer quality in one report, with every failure traced
to the stage that caused it.

``rag_report(records, k=5)`` takes one record per question:

* ``retrieved`` — ranked document ids; ``relevant`` — the ids that answer the question (optional);
* ``answer`` — the system's answer; ``reference`` — the gold answer (a string or a list of acceptable answers);
* ``grounded`` (true / false) or ``support`` (per-claim 0 / 1 or scores in [0, 1], e.g. from an NLI model or an
  LLM judge) — whether the answer is supported by the retrieved context (optional).

It reports hit rate, recall and MRR at ``k``, exact match and token F1, the grounded rate, each with an
interval, and classifies every question:

* ``correct`` — right answer, supported by the context (or no grounding data given);
* ``retrieval_failure`` — wrong answer and no relevant document in the top ``k``: fix the retriever;
* ``generation_failure`` — wrong answer although a relevant document was retrieved: fix the prompt / model;
* ``unsupported_correct`` — right answer that the context does not support (memorised or lucky);
* ``wrong_unknown_cause`` — wrong answer without relevance labels to tell retrieval from generation.

The breakdown (with Wilson intervals) says where to invest, and ``examples`` lists the ids in each class.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError

__all__ = ["RAGReport", "rag_report"]

CLASSES = ("correct", "retrieval_failure", "generation_failure", "unsupported_correct", "wrong_unknown_cause")
_ADVICE = {
    "retrieval_failure": "improve retrieval (recall, chunking, query rewriting)",
    "generation_failure": "improve generation (prompting, context use, the model)",
    "unsupported_correct": "check grounding: answers not supported by the retrieved context",
    "wrong_unknown_cause": "add relevance labels to separate retrieval from generation failures",
}


@dataclass(frozen=True, eq=False)
class RAGReport:
    """Result of :func:`rag_report`."""

    metrics: Mapping[str, Mapping[str, Any]]
    breakdown: Mapping[str, Mapping[str, Any]]
    examples: Mapping[str, tuple[Any, ...]]
    settings: Mapping[str, Any]

    def summary(self, *, digits: int = 3) -> str:
        d = digits
        pct = f"{self.settings['level'] * 100:g}%"
        n = self.settings["n"]
        lines = [f"EvalSuite RAG report ({n} question{'s' * (n != 1)}, k={self.settings['k']}, {pct} CIs)"]
        for m in self.metrics.values():
            lines.append(f"  {m['name']:<22} {m['value']:.{d}f} ({m['low']:.{d}f}–{m['high']:.{d}f})")
        lines.append("  Outcome per question:")
        for k, b in self.breakdown.items():
            if b["count"]:
                lines.append(
                    f"    {k:<22} {b['count']:5d}  {b['share']:.{d}f} ({b['low']:.{d}f}–{b['high']:.{d}f})"
                )
        top = self.biggest_problem()
        if top:
            lines.append(f"  Biggest problem: {top} → {_ADVICE[top]}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def biggest_problem(self) -> Optional[str]:
        bad = {k: b["count"] for k, b in self.breakdown.items() if k != "correct" and b["count"]}
        return max(bad, key=lambda k: bad[k]) if bad else None

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "metrics": {k: dict(v) for k, v in self.metrics.items()},
                "breakdown": {k: dict(v) for k, v in self.breakdown.items()},
                "examples": {k: list(v) for k, v in self.examples.items()},
                "settings": dict(self.settings),
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_markdown(self, *, digits: int = 3) -> str:
        d = digits
        rows = ["| Measure | Value | CI |", "| --- | ---: | --- |"]
        rows += [
            f"| {m['name']} | {m['value']:.{d}f} | {m['low']:.{d}f}–{m['high']:.{d}f} |"
            for m in self.metrics.values()
        ]
        rows += ["", "| Outcome | Questions | Share (CI) |", "| --- | ---: | --- |"]
        rows += [
            f"| {k} | {b['count']} | {b['share']:.{d}f} ({b['low']:.{d}f}–{b['high']:.{d}f}) |"
            for k, b in self.breakdown.items()
        ]
        return "\n".join(rows) + "\n"


def _ids(v: Any, what: str, i: int) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (str, int)):
        return [str(v)]
    try:
        return [str(x) for x in v]
    except TypeError as exc:
        raise InputValidationError(f"Record {i}: {what} must be a list of document ids.") from exc


def rag_report(
    records: Any,
    *,
    k: int = 5,
    answer_threshold: float = 0.5,
    support_threshold: float = 0.5,
    level: float = 0.95,
) -> RAGReport:
    """See the module documentation. An answer counts as correct when its exact match is 1 or its token F1 is at
    least ``answer_threshold``; with ``support`` scores, it is grounded when their mean is at least
    ``support_threshold``."""
    from . import proportion_ci
    from .text.qa import exact_match, token_f1

    if hasattr(records, "to_dict") and hasattr(records, "columns"):
        rows = records.to_dict("records")
    else:
        rows = list(records)
    if not rows:
        raise InputValidationError("rag_report needs at least one record.")
    if k < 1:
        raise InputValidationError("k must be a positive integer.")
    n = len(rows)
    hit, rec, rr, em, f1, correct, grounded = [], [], [], [], [], [], []
    have_rel = have_ground = True
    ids = []
    for i, r in enumerate(rows):
        if not isinstance(r, Mapping):
            raise InputValidationError(f"Record {i} is not a mapping.")
        if "answer" not in r or "reference" not in r:
            raise InputValidationError(f"Record {i} needs 'answer' and 'reference'.")
        ids.append(r.get("id", i))
        refs = r["reference"] if isinstance(r["reference"], (list, tuple)) else [r["reference"]]
        e = float(exact_match([list(map(str, refs))], [str(r["answer"])]).value)
        f = float(token_f1([list(map(str, refs))], [str(r["answer"])]).value)
        em.append(e)
        f1.append(f)
        correct.append(e == 1.0 or f >= answer_threshold)
        rel = set(_ids(r.get("relevant"), "relevant", i))
        got = _ids(r.get("retrieved"), "retrieved", i)[:k]
        if rel:
            found = [j for j, d in enumerate(got) if d in rel]
            hit.append(1.0 if found else 0.0)
            rec.append(len(set(got) & rel) / len(rel))
            rr.append(1.0 / (found[0] + 1) if found else 0.0)
        else:
            have_rel = False
            hit.append(math.nan)
        if r.get("grounded") is not None:
            grounded.append(bool(r["grounded"]))
        elif r.get("support") is not None:
            s = np.atleast_1d(np.asarray(r["support"], dtype=float))
            if s.size == 0 or not np.all((s >= 0) & (s <= 1)):
                raise InputValidationError(f"Record {i}: support must hold scores in [0, 1].")
            grounded.append(bool(s.mean() >= support_threshold))
        else:
            have_ground = False
            grounded.append(True)

    classes = []
    for i in range(n):
        if correct[i]:
            classes.append("correct" if grounded[i] else "unsupported_correct")
        elif math.isnan(hit[i]):
            classes.append("wrong_unknown_cause")
        elif hit[i] == 0:
            classes.append("retrieval_failure")
        else:
            classes.append("generation_failure")

    def prop(name: str, vals: list[float]) -> dict[str, Any]:
        v = [x for x in vals if not math.isnan(x)]
        m = len(v)
        if not m:
            return {"name": name, "value": math.nan, "low": math.nan, "high": math.nan, "n": 0}
        if all(x in (0.0, 1.0) for x in v):
            ci = proportion_ci(int(sum(v)), m, level=level)
            lo, hi = ci.low, ci.high
        else:
            arr = np.asarray(v)
            from scipy import stats

            half = (
                float(stats.t.ppf(1 - (1 - level) / 2, m - 1) * arr.std(ddof=1) / math.sqrt(m))
                if m > 1
                else math.nan
            )
            lo, hi = max(0.0, arr.mean() - half), min(1.0, arr.mean() + half)
        return {"name": name, "value": float(np.mean(v)), "low": float(lo), "high": float(hi), "n": m}

    metrics: dict[str, dict[str, Any]] = {}
    if any(not math.isnan(h) for h in hit):
        metrics["hit_rate"] = prop(f"Hit rate@{k}", hit)
        metrics["recall"] = prop(f"Recall@{k}", rec)
        metrics["mrr"] = prop(f"MRR@{k}", rr)
    metrics["exact_match"] = prop("Exact match", em)
    metrics["token_f1"] = prop("Token F1", f1)
    metrics["answer_correct"] = prop("Answer correct", [float(c) for c in correct])
    if have_ground:
        metrics["grounded"] = prop("Grounded", [float(g) for g in grounded])
    breakdown = {}
    for c in CLASSES:
        cnt = classes.count(c)
        ci = proportion_ci(cnt, n, level=level)
        breakdown[c] = {"count": cnt, "share": cnt / n, "low": ci.low, "high": ci.high}
    examples = {c: tuple(ids[i] for i in range(n) if classes[i] == c) for c in CLASSES}
    return RAGReport(
        metrics=metrics,
        breakdown=breakdown,
        examples=examples,
        settings={
            "n": n,
            "k": k,
            "level": level,
            "answer_threshold": answer_threshold,
            "support_threshold": support_threshold,
            "relevance_labels": have_rel,
            "grounding": have_ground,
        },
    )
