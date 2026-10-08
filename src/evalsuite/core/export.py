"""Shared HTML, CSV and file-saving helpers for every result type."""

from __future__ import annotations

import csv
import html
import io
import os
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

from .exceptions import InputValidationError

__all__ = ["csv_text", "html_document", "html_table", "save_as"]

PathLike = Union[str, "os.PathLike[str]"]

_CSS = """
body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:2rem;color:#1f2328}
h1{font-size:1.25rem;margin:0 0 .25rem}p.meta{color:#59636e;margin:0 0 1rem;font-size:.875rem}
table.evalsuite{border-collapse:collapse;margin:0 0 1.5rem;font-variant-numeric:tabular-nums}
table.evalsuite caption{text-align:left;font-weight:600;padding:0 0 .5rem}
table.evalsuite th,table.evalsuite td{padding:.35rem .75rem;border-bottom:1px solid #d1d9e0}
table.evalsuite thead th{border-bottom:2px solid #1f2328;text-align:left}
table.evalsuite td.num,table.evalsuite th.num{text-align:right}
table.evalsuite tr.avg td{background:#f6f8fa}table.evalsuite .best{font-weight:700}
"""


def html_table(
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    caption: Optional[str] = None,
    numeric: Optional[Sequence[bool]] = None,
    row_classes: Optional[Sequence[str]] = None,
    bold: Optional[set[tuple[int, int]]] = None,
) -> str:
    """An escaped HTML table with class ``evalsuite``. ``bold`` marks (row, column) cells."""
    numeric = list(numeric) if numeric is not None else [i > 0 for i in range(len(header))]
    out = ['<table class="evalsuite">']
    if caption:
        out.append(f"<caption>{html.escape(caption)}</caption>")
    out.append(
        "<thead><tr>"
        + "".join(
            f'<th{" class=num" if numeric[i] else ""} scope="col">{html.escape(str(h))}</th>'.replace(
                " class=num", ' class="num"'
            )
            for i, h in enumerate(header)
        )
        + "</tr></thead><tbody>"
    )
    for r, row in enumerate(rows):
        cls = f' class="{html.escape(row_classes[r])}"' if row_classes and row_classes[r] else ""
        cells = []
        for c, value in enumerate(row):
            classes = [n for n, on in (("num", numeric[c]), ("best", bool(bold and (r, c) in bold))) if on]
            attr = f' class="{" ".join(classes)}"' if classes else ""
            tag = "th" if c == 0 else "td"
            scope = ' scope="row"' if c == 0 else ""
            cells.append(f"<{tag}{attr}{scope}>{html.escape(str(value))}</{tag}>")
        out.append(f"<tr{cls}>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def html_document(title: str, body: str, meta: Optional[str] = None) -> str:
    """A standalone, dependency-free HTML page (inline CSS, no scripts)."""
    meta_html = f'<p class="meta">{html.escape(meta)}</p>' if meta else ""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n{meta_html}\n{body}\n</body>\n</html>\n"
    )


def csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """RFC 4180 CSV. Floats keep full precision; NaN is written as an empty field."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if isinstance(v, float) and v != v else v for v in row])
    return buf.getvalue()


def save_as(path: PathLike, writers: dict[str, Callable[[], str]]) -> str:
    """Write the format chosen by the file extension; returns the path written."""
    p = os.fspath(path)
    ext = os.path.splitext(p)[1].lower().lstrip(".")
    aliases = {"tex": "latex", "md": "markdown", "htm": "html", "txt": "text"}
    fmt = aliases.get(ext, ext)
    if fmt not in writers:
        exts = ", ".join(sorted({"." + e for e in (*writers, *aliases) if aliases.get(e, e) in writers}))
        raise InputValidationError(f"Cannot save as '.{ext}'. Supported extensions: {exts}.")
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(writers[fmt]())
    return p
