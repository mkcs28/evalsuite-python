"""Reporting standards: the evaluation sections that journals and model cards ask for, generated from data.

``report_card(y_true, y_prob, standard="tripod-ai")`` computes what TRIPOD+AI (Collins et al., BMJ 2024) asks a
prediction-model paper to report about performance (sample size and outcome events; discrimination; calibration
in the large, slope and plot-free summaries; classification at a threshold; clinical utility by net benefit;
performance in subgroups) with confidence intervals, writes it up as a results paragraph and a table, and lists
which checklist items are covered and which need your own text. ``standard="stard"`` (STARD 2015, diagnostic
accuracy studies) and ``standard="consort-ai"`` (CONSORT-AI, trials of AI interventions) fill in those checklists'
results items the same way. ``standard="model-card"`` produces the sections of
a model card (Mitchell et al., FAT* 2019) with the quantitative analyses filled in and the narrative sections
taken from ``context``.

Export with ``to_markdown()``, ``to_latex()``, ``to_html()`` or ``to_docx(path)`` (python-docx).
"""

from __future__ import annotations

# ruff: noqa: E501  (long f-strings of report text)
import contextlib
import html as _html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .core.exceptions import EvalSuiteError, InputValidationError, OptionalDependencyError

_TRIPOD_ITEMS = [
    ("23a", "Sample size and number of outcome events in the evaluation data", "n"),
    ("23b", "Discrimination with confidence interval (e.g. ROC AUC / c-statistic)", "roc_auc"),
    (
        "23c",
        "Calibration: calibration-in-the-large (intercept) and calibration slope, with confidence intervals",
        "calibration",
    ),
    ("23d", "Overall performance (e.g. Brier score)", "brier_score"),
    ("23e", "Classification measures at the chosen threshold (sensitivity, specificity, PPV, NPV)", "threshold"),
    ("23f", "Clinical utility (net benefit / decision curve)", "net_benefit"),
    ("23g", "Performance in relevant subgroups (fairness)", "subgroups"),
    ("12", "Description of how missing data were handled", None),
    ("8", "Definition and timing of the outcome", None),
]

_STARD_ITEMS = [
    ("20", "Baseline characteristics: numbers with and without the target condition", "n"),
    ("23", "Cross tabulation of the index test results by the reference standard (2×2 table)", "threshold"),
    (
        "24",
        "Estimates of diagnostic accuracy and their precision (sensitivity, specificity, 95% CIs)",
        "threshold",
    ),
    ("24b", "Discrimination across thresholds (ROC AUC) with confidence interval", "roc_auc"),
    ("24c", "Accuracy in clinically relevant subgroups", "subgroups"),
    ("19", "Flow of participants (diagram)", None),
    ("21a", "Distribution of severity of disease in those with the target condition", None),
    ("25", "Adverse events from performing the index test or the reference standard", None),
]

_CONSORT_AI_ITEMS = [
    ("17a", "Results for each outcome with the estimated effect size and its precision (95% CI)", "roc_auc"),
    ("17a-ii", "Classification performance at the operating threshold used in the trial", "threshold"),
    ("17a-iii", "Calibration of the AI system's risk estimates", "calibration"),
    ("19-AI", "Analysis of performance errors and how they were identified (e.g. by subgroup)", "subgroups"),
    ("5-AI(iii)", "How poor quality or unavailable input data were assessed and handled", None),
    ("5-AI(vi)", "Human–AI interaction and the expertise required of users", None),
    ("25-AI", "Whether and how the AI intervention and its code can be accessed", None),
]

_STANDARDS = {
    "tripod-ai": ("Model performance (TRIPOD+AI)", _TRIPOD_ITEMS),
    "stard": ("Diagnostic accuracy (STARD 2015)", _STARD_ITEMS),
    "consort-ai": ("AI intervention performance (CONSORT-AI)", _CONSORT_AI_ITEMS),
}

_CARD_SECTIONS = [
    ("model_details", "Model details"),
    ("intended_use", "Intended use"),
    ("factors", "Factors"),
    ("metrics", "Metrics"),
    ("evaluation_data", "Evaluation data"),
    ("training_data", "Training data"),
    ("quantitative_analyses", "Quantitative analyses"),
    ("ethical_considerations", "Ethical considerations"),
    ("caveats", "Caveats and recommendations"),
]


def _ci(est: float, lo: float, hi: float, d: int) -> str:
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), (lo, hi))):
        return f"{est:.{d}f}"
    return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


@dataclass(frozen=True, eq=False)
class ReportCard:
    """Computed performance results plus the text of a reporting-standard section."""

    standard: str
    rows: tuple[dict[str, Any], ...]
    subgroups: tuple[dict[str, Any], ...]
    checklist: tuple[dict[str, Any], ...]
    sections: Mapping[str, str]
    settings: Mapping[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------------------------------------- text

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def paragraph(self, *, digits: int = 3) -> str:
        """A results paragraph in the style of a methods/results section."""
        s, d = self.settings, digits
        get = {r["key"]: r for r in self.rows}
        parts = [
            f"The model was evaluated on {s['n']} observations with {s['events']} outcome events "
            f"({s['events'] / s['n']:.1%})."
        ]
        if "roc_auc" in get:
            r = get["roc_auc"]
            parts.append(
                f"Discrimination was {'good' if r['value'] >= 0.8 else 'moderate' if r['value'] >= 0.7 else 'limited'}: "
                f"the c-statistic (ROC AUC) was {_ci(r['value'], r['low'], r['high'], d)}."
            )
        if "calibration_intercept" in get and "calibration_slope" in get:
            a, b = get["calibration_intercept"], get["calibration_slope"]
            parts.append(
                f"Calibration-in-the-large was {_ci(a['value'], a['low'], a['high'], d)} (ideal 0) and the calibration "
                f"slope {_ci(b['value'], b['low'], b['high'], d)} (ideal 1)."
            )
        if "brier_score" in get:
            r = get["brier_score"]
            parts.append(f"The Brier score was {_ci(r['value'], r['low'], r['high'], d)}.")
        if "sensitivity" in get:
            se, sp = get["sensitivity"], get["specificity"]
            parts.append(
                f"At a risk threshold of {s['threshold']:g}, sensitivity was {_ci(se['value'], se['low'], se['high'], d)} "
                f"and specificity {_ci(sp['value'], sp['low'], sp['high'], d)}."
            )
        if "net_benefit" in get:
            r = get["net_benefit"]
            parts.append(
                f"The net benefit at that threshold was {r['value']:.{d}f}, against {get['net_benefit_all']['value']:.{d}f} "
                "for treating everyone and 0 for treating no one."
            )
        if self.subgroups:
            vals = [g for g in self.subgroups if not math.isnan(g["roc_auc"])]
            if vals:
                lo, hi = min(vals, key=lambda g: g["roc_auc"]), max(vals, key=lambda g: g["roc_auc"])
                parts.append(
                    f"Across subgroups ({s['group_name']}), the c-statistic ranged from {lo['roc_auc']:.{d}f} "
                    f"({lo['group']}) to {hi['roc_auc']:.{d}f} ({hi['group']})."
                )
        return " ".join(parts)

    def summary(self, *, digits: int = 3) -> str:
        pct = f"{self.settings['level'] * 100:g}%"
        width = max(len(r["name"]) for r in self.rows)
        lines = [f"EvalSuite report card ({self.standard}, n={self.settings['n']}, {pct} CIs)"]
        for r in self.rows:
            lines.append(f"  {r['name']:<{width}}  {_ci(r['value'], r['low'], r['high'], digits)}")
        for item in self.checklist:
            mark = "x" if item["covered"] else " "
            lines.append(f"  [{mark}] {item['item']}: {item['text']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # -------------------------------------------------------------------------------------------- exports

    def to_markdown(self, *, digits: int = 3) -> str:
        pct = f"{self.settings['level'] * 100:g}%"
        out = []
        if self.standard == "model-card":
            for key, title in _CARD_SECTIONS:
                out.append(f"## {title}\n")
                if key == "quantitative_analyses":
                    out.append(self._md_table(digits, pct))
                    if self.subgroups:
                        out.append("\n" + self._md_subgroups(digits))
                    out.append("")
                else:
                    out.append(self.sections.get(key, "_To be completed._") + "\n")
            return "\n".join(out).rstrip() + "\n"
        out.append(f"## {_STANDARDS[self.standard][0]}\n")
        out.append(self.paragraph(digits=digits) + "\n")
        out.append(self._md_table(digits, pct))
        conf = self.settings.get("confusion")
        if self.standard == "stard" and conf:
            out.append(
                "\n| Index test \\ Reference | Positive | Negative | Total |\n| --- | ---: | ---: | ---: |\n"
                f"| Positive | {conf['tp']} | {conf['fp']} | {conf['tp'] + conf['fp']} |\n"
                f"| Negative | {conf['fn']} | {conf['tn']} | {conf['fn'] + conf['tn']} |\n"
                f"| Total | {conf['tp'] + conf['fn']} | {conf['fp'] + conf['tn']} | {sum(conf.values())} |"
            )
        if self.subgroups:
            out.append("\n" + self._md_subgroups(digits))
        out.append("\n### Checklist\n")
        for item in self.checklist:
            out.append(f"- [{'x' if item['covered'] else ' '}] **{item['item']}** {item['text']}")
        return "\n".join(out) + "\n"

    def _md_table(self, digits: int, pct: str) -> str:
        rows = [f"| Measure | Estimate ({pct} CI) |", "| --- | --- |"]
        rows += [f"| {r['name']} | {_ci(r['value'], r['low'], r['high'], digits)} |" for r in self.rows]
        return "\n".join(rows)

    def _md_subgroups(self, digits: int) -> str:
        rows = [
            f"| {self.settings['group_name']} | n | Events | ROC AUC | Calibration-in-the-large |",
            "| --- | ---: | ---: | --- | --- |",
        ]
        for g in self.subgroups:
            rows.append(
                f"| {g['group']} | {g['n']} | {g['events']} | {_ci(g['roc_auc'], g['roc_auc_low'], g['roc_auc_high'], digits)} | "
                f"{g['calibration_in_the_large']:.{digits}f} |"
            )
        return "\n".join(rows)

    def to_latex(self, *, digits: int = 3, caption: Optional[str] = None, label: Optional[str] = None) -> str:
        from .core.result import _latex_escape, _latex_table

        rows = [
            [_latex_escape(r["name"]), _ci(r["value"], r["low"], r["high"], digits).replace("–", "--")]
            for r in self.rows
        ]
        pct = f"{self.settings['level'] * 100:g}\\%"
        table = _latex_table(["Measure", f"Estimate ({pct} CI)"], rows, caption=caption, label=label)
        text = _latex_escape(self.paragraph(digits=digits)).replace("–", "--")
        return text + "\n\n" + table

    def to_html(self, *, digits: int = 3, full: bool = False) -> str:
        body = f"<p>{_html.escape(self.paragraph(digits=digits))}</p><table><tr><th>Measure</th><th>Estimate</th></tr>"
        for r in self.rows:
            body += f"<tr><td>{_html.escape(r['name'])}</td><td>{_ci(r['value'], r['low'], r['high'], digits)}</td></tr>"
        body += "</table>"
        if not full:
            return body
        return f"<!doctype html><html><head><meta charset='utf-8'><title>Report card</title></head><body>{body}</body></html>"

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "standard": self.standard,
                "rows": list(self.rows),
                "subgroups": list(self.subgroups),
                "checklist": list(self.checklist),
                "sections": dict(self.sections),
                "settings": dict(self.settings),
                "paragraph": self.paragraph(),
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def to_csv(self) -> str:
        from .core.export import csv_text

        return csv_text(
            ["measure", "estimate", "ci_low", "ci_high"],
            [[r["name"], r["value"], r["low"], r["high"]] for r in self.rows],
        )

    def to_docx(self, path: str, *, digits: int = 3) -> None:
        """Write a Word document with the paragraph, the table and (model card) the narrative sections."""
        try:
            import docx  # type: ignore[import-not-found, unused-ignore]
        except ImportError as exc:
            raise OptionalDependencyError("python-docx", "docx", "ReportCard.to_docx") from exc
        doc = docx.Document()
        doc.add_heading("Model card" if self.standard == "model-card" else _STANDARDS[self.standard][0], level=1)
        if self.standard == "model-card":
            for key, title in _CARD_SECTIONS:
                if key == "quantitative_analyses":
                    continue
                doc.add_heading(title, level=2)
                doc.add_paragraph(self.sections.get(key, "To be completed."))
            doc.add_heading("Quantitative analyses", level=2)
        doc.add_paragraph(self.paragraph(digits=digits))
        t = doc.add_table(rows=1, cols=2)
        t.rows[0].cells[0].text, t.rows[0].cells[1].text = (
            "Measure",
            f"Estimate ({self.settings['level'] * 100:g}% CI)",
        )
        for r in self.rows:
            c = t.add_row().cells
            c[0].text, c[1].text = r["name"], _ci(r["value"], r["low"], r["high"], digits)
        doc.save(path)


def report_card(
    y_true: Any,
    y_prob: Any,
    *,
    y_pred: Any = None,
    standard: str = "tripod-ai",
    threshold: float = 0.5,
    groups: Optional[Sequence[Any]] = None,
    group_name: str = "subgroup",
    context: Optional[Mapping[str, str]] = None,
    pos_label: Any = None,
    level: float = 0.95,
    n_resamples: int = 1000,
    random_state: Optional[int] = 0,
) -> ReportCard:
    """Performance section for a reporting standard (``"tripod-ai"``, ``"stard"`` for diagnostic accuracy studies,
    ``"consort-ai"`` for trials of AI interventions, or ``"model-card"``) from a binary outcome
    and predicted risks. ``groups`` adds a subgroup (fairness) table; ``context`` supplies the narrative sections
    of a model card (keys: model_details, intended_use, factors, metrics, evaluation_data, training_data,
    ethical_considerations, caveats). Intervals: DeLong for ROC AUC, Wilson for proportions, bootstrap (seeded)
    for calibration and the Brier score."""
    from . import (
        bootstrap_ci,
        brier_score,
        calibration_intercept,
        calibration_slope,
        expected_calibration_error,
        net_benefit,
        proportion_ci,
        roc_auc,
        roc_auc_ci,
    )

    if standard not in (*_STANDARDS, "model-card"):
        raise InputValidationError("standard must be 'tripod-ai', 'stard', 'consort-ai' or 'model-card'.")
    if not 0 < threshold < 1:
        raise InputValidationError("threshold must be in (0, 1).")
    yt = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if yt.ndim != 1 or p.shape != yt.shape:
        raise InputValidationError(
            "y_true and y_prob must be one-dimensional and the same length (binary outcome)."
        )
    labels = np.unique(yt)
    if labels.size != 2:
        raise InputValidationError("report_card needs a binary outcome with both classes present.")
    pos = labels[-1] if pos_label is None else pos_label
    y = (yt == pos).astype(int)
    pred = (p >= threshold).astype(int) if y_pred is None else (np.asarray(y_pred) == pos).astype(int)
    n, events = int(y.size), int(y.sum())
    rows: list[dict[str, Any]] = []

    def add(
        key: str, name: str, value: float, low: float = math.nan, high: float = math.nan, method: str = ""
    ) -> None:
        rows.append(
            {
                "key": key,
                "name": name,
                "value": float(value),
                "low": float(low),
                "high": float(high),
                "method": method,
            }
        )

    auc = roc_auc_ci(y, p, level=level)
    add("roc_auc", "c-statistic (ROC AUC)", auc.estimate, auc.low, auc.high, "DeLong")
    for key, fn, name in (
        ("calibration_intercept", calibration_intercept, "Calibration-in-the-large (intercept)"),
        ("calibration_slope", calibration_slope, "Calibration slope"),
        ("brier_score", brier_score, "Brier score"),
        ("ece", expected_calibration_error, "Expected calibration error"),
    ):
        try:
            ci = bootstrap_ci(
                fn,
                y,
                y_prob=p,
                level=level,
                n_resamples=n_resamples,
                random_state=random_state,
                method="percentile",
            )
            add(key, name, ci.estimate, ci.low, ci.high, "bootstrap")
        except EvalSuiteError:
            with contextlib.suppress(EvalSuiteError):
                add(key, name, float(fn(y, y_prob=p)))
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn_ = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    for key, name, k, m in (
        ("sensitivity", "Sensitivity", tp, tp + fn_),
        ("specificity", "Specificity", tn, tn + fp),
        ("ppv", "Positive predictive value", tp, tp + fp),
        ("npv", "Negative predictive value", tn, tn + fn_),
    ):
        if m:
            ci = proportion_ci(k, m, level=level)
            add(key, name, ci.estimate, ci.low, ci.high, "Wilson")
    nb = float(net_benefit(y, y_prob=p, threshold=threshold))
    prev = events / n
    add("net_benefit", f"Net benefit at threshold {threshold:g}", nb)
    add("net_benefit_all", "Net benefit, treat all", prev - (1 - prev) * threshold / (1 - threshold))
    subgroups: list[dict[str, Any]] = []
    if groups is not None:
        g = np.asarray(list(groups))
        if g.shape != y.shape:
            raise InputValidationError("groups must have one label per observation.")
        for lab in sorted(set(g.tolist()), key=str):
            m = g == lab
            row: dict[str, Any] = {
                "group": str(lab),
                "n": int(m.sum()),
                "events": int(y[m].sum()),
                "calibration_in_the_large": float(y[m].mean() - p[m].mean()),
            }
            if 0 < row["events"] < row["n"]:
                ci = roc_auc_ci(y[m], p[m], level=level)
                row.update(roc_auc=ci.estimate, roc_auc_low=ci.low, roc_auc_high=ci.high)
            else:
                row.update(roc_auc=math.nan, roc_auc_low=math.nan, roc_auc_high=math.nan)
            subgroups.append(row)
    keys = {r["key"] for r in rows}
    covered = {
        "n": True,
        "roc_auc": "roc_auc" in keys,
        "calibration": {"calibration_intercept", "calibration_slope"} <= keys,
        "brier_score": "brier_score" in keys,
        "threshold": "sensitivity" in keys,
        "net_benefit": "net_benefit" in keys,
        "subgroups": bool(subgroups),
    }
    checklist = [
        {
            "item": item,
            "text": text,
            "covered": bool(covered.get(k, False)) if k else False,
            "needs_text": k is None,
        }
        for item, text, k in _STANDARDS.get(standard, _STANDARDS["tripod-ai"])[1]
    ]
    ctx = dict(context or {})
    sections = {key: ctx[key] for key, _ in _CARD_SECTIONS if key in ctx}
    if standard == "model-card":
        sections.setdefault(
            "metrics",
            "Discrimination (c-statistic), calibration (intercept, slope, ECE), Brier score, sensitivity / "
            f"specificity / PPV / NPV at a risk threshold of {threshold:g}, and net benefit; {level * 100:g}% "
            "confidence intervals throughout (EvalSuite).",
        )
    unknown = set(ctx) - {k for k, _ in _CARD_SECTIONS}
    if unknown:
        raise InputValidationError(f"Unknown context section(s): {sorted(unknown)}.")
    return ReportCard(
        standard=standard,
        rows=tuple(rows),
        subgroups=tuple(subgroups),
        checklist=tuple(checklist),
        sections=sections,
        settings={
            "n": n,
            "events": events,
            "threshold": threshold,
            "level": level,
            "group_name": group_name,
            "n_resamples": n_resamples,
            "random_state": random_state,
            "roc_auc": float(roc_auc(y, y_prob=p)),
            "confusion": {"tp": tp, "fp": fp, "fn": fn_, "tn": tn},
        },
    )
