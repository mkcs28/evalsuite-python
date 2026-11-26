"""One self-contained HTML file for a whole evaluation: tables, intervals, plots and narrative, to attach to a
paper, a pull request or an e-mail.

>>> page = es.html_report(
...     {"Test set": result, "Leaderboard": board, "Fairness": fairness, "Calibration": es.plot.calibration(...)},
...     title="Sepsis model v3 evaluation", path="report.html",
... )

Each section can be any EvalSuite result (``evaluate``, ``compare``, ``leaderboard``, ``report_card``,
``fairness_report``, ``judge_report``, ``rag_report``, sample-size plans, metric results), a Matplotlib figure
(embedded as a PNG), or text (paragraphs; Markdown tables are rendered as tables). The file has inline CSS, no
scripts and no external requests, so it opens anywhere and is safe to share; all text is escaped.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import platform
from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

from .core.exceptions import InputValidationError

__all__ = ["html_report"]

_EXTRA_CSS = """
section{margin:0 0 2rem}h2{font-size:1.05rem;border-bottom:1px solid #d1d9e0;padding-bottom:.25rem}
table{border-collapse:collapse;margin:0 0 1rem;font-variant-numeric:tabular-nums}
td,th{padding:.3rem .7rem;border-bottom:1px solid #e6e9ed;text-align:left}thead th{border-bottom:2px solid #1f2328}
pre{background:#f6f8fa;padding:.75rem;overflow-x:auto;font-size:.85rem}img{max-width:100%;height:auto}
.toc a{margin-right:1rem}footer{color:#59636e;font-size:.8rem;margin-top:3rem}
@media (prefers-color-scheme:dark){body{background:#0d1117;color:#e6edf3}pre{background:#161b22}
td,th{border-color:#30363d}thead th{border-color:#e6edf3}}
"""


def _markdown_tables(text: str) -> str:
    """Paragraphs and pipe tables of a Markdown string as HTML (everything escaped)."""
    out: list[str] = []
    for block in text.strip().split("\n\n"):
        lines = [x for x in block.splitlines() if x.strip()]
        if lines and all(x.strip().startswith("|") for x in lines):
            body = [x for x in lines if not set(x.replace("|", "").strip()) <= set("-: ")]
            cells = [[html.escape(c.strip()) for c in x.strip().strip("|").split("|")] for x in body]
            head = "".join(f"<th>{c}</th>" for c in cells[0])
            rows = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        elif lines and lines[0].startswith("#"):
            level = min(len(lines[0]) - len(lines[0].lstrip("#")) + 2, 6)
            out.append(f"<h{level}>{html.escape(lines[0].lstrip('#').strip())}</h{level}>")
            if len(lines) > 1:
                out.append(_markdown_tables("\n".join(lines[1:])))
        elif lines and all(x.lstrip().startswith(("- ", "* ")) for x in lines):
            items = "".join(f"<li>{html.escape(x.lstrip()[2:])}</li>" for x in lines)
            out.append(f"<ul>{items}</ul>")
        elif lines:
            out.append(f"<p>{html.escape(' '.join(lines))}</p>")
    return "\n".join(out)


def _figure(fig: Any) -> Optional[str]:
    savefig = getattr(fig, "savefig", None)
    if savefig is None and hasattr(fig, "figure") and hasattr(fig.figure, "savefig"):  # an Axes
        savefig = fig.figure.savefig
    if savefig is None:
        return None
    buf = io.BytesIO()
    savefig(buf, format="png", dpi=144, bbox_inches="tight")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img alt="figure" src="data:image/png;base64,{data}">'


def _render(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return _markdown_tables(item)
    img = _figure(item)
    if img is not None:
        return img
    if isinstance(item, (list, tuple)):
        return "\n".join(_render(x) for x in item)
    # results with their own HTML: prefer a table without page chrome
    for attr in ("to_html",):
        fn = getattr(item, attr, None)
        if callable(fn):
            try:
                return str(fn())
            except TypeError:
                pass
    for attr in ("to_markdown",):
        fn = getattr(item, attr, None)
        if callable(fn):
            return _markdown_tables(str(fn()))
    summary = getattr(item, "summary", None)
    text = summary() if callable(summary) else repr(item)
    return f"<pre>{html.escape(str(text))}</pre>"


def html_report(
    sections: Union[Mapping[str, Any], Sequence[Any]],
    *,
    title: str = "Evaluation report",
    intro: Optional[str] = None,
    path: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build the report and return the HTML (also written to ``path`` when given). ``sections`` maps headings to
    content, or is a list (headings are then taken from each result's kind). ``intro`` is Markdown text placed
    under the title; ``metadata`` adds rows to the provenance footer (data set, model version, ...)."""
    from .version import __version__

    if isinstance(sections, Mapping):
        items = [(str(k), v) for k, v in sections.items()]
    elif isinstance(sections, Sequence) and not isinstance(sections, str):
        items = [
            (type(v).__name__.replace("Result", "").replace("Report", " report") or "Section", v) for v in sections
        ]
    else:
        raise InputValidationError("sections must be a mapping of heading → content or a list of results.")
    if not items:
        raise InputValidationError("html_report needs at least one section.")
    body = [f"<h1>{html.escape(title)}</h1>"]
    if intro:
        body.append(_markdown_tables(intro))
    if len(items) > 2:
        links = "".join(f'<a href="#s{i}">{html.escape(h)}</a>' for i, (h, _) in enumerate(items))
        body.append(f'<nav class="toc">{links}</nav>')
    for i, (heading, content) in enumerate(items):
        body.append(f'<section id="s{i}"><h2>{html.escape(heading)}</h2>\n{_render(content)}</section>')
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = {"Generated": now, "EvalSuite": __version__, "Python": platform.python_version(), **(metadata or {})}
    foot = " · ".join(f"{html.escape(str(k))}: {html.escape(str(v))}" for k, v in meta.items())
    body.append(f"<footer>{foot}</footer>")
    from .core.export import _CSS

    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}{_EXTRA_CSS}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )
    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(page)
    return page
