"""Turn a per-metric benchmark (``run_benchmarks(suite="metrics")`` JSON) into BENCHMARKS.md text and,
optionally, the website's data file.

    python scripts/bench_metrics_report.py metrics.json > every_metric.md
    python scripts/bench_metrics_report.py metrics.json --site SITE/src/data/benchmarks/metrics.generated.ts

Rows are split by what EvalSuite is compared with: a reference library, an independent textbook formula
(NumPy or the standard library, which skips input validation and so sets a lower bound on time), or nothing
(learned, judge-dependent or randomised procedures, timed alone). Speed-up summaries use library rows only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict

from evalsuite.benchmarks import MATCH_TOLERANCE, benchmark_group

FORMULA = {"NumPy formula", "Python stdlib", "SQuAD formula"}


def kind(ref: str | None) -> str:
    if not ref:
        return "none"
    return "formula" if ref in FORMULA else "library"


def geomean(v: list[float]) -> float | None:
    v = [x for x in v if x and x > 0]
    return math.exp(sum(map(math.log, v)) / len(v)) if v else None


def load(path: str) -> tuple[dict, list[dict]]:
    data = json.load(open(path, encoding="utf-8"))  # noqa: SIM115
    by_case: dict[str, dict] = {}
    for r in data["rows"]:
        e = by_case.setdefault(
            r["case"],
            {
                "metric": r["case"],
                "group": benchmark_group(r["case"]),
                "reference": r["reference"] or "",
                "sizes": {},
            },
        )
        e["sizes"][r["n"]] = r
    rows = sorted(by_case.values(), key=lambda e: e["metric"])
    for e in rows:
        e["kind"] = kind(e["reference"])
        diffs = [s["max_abs_diff"] for s in e["sizes"].values() if s["max_abs_diff"] is not None]
        e["max_diff"] = max(diffs) if diffs else None
        e["match"] = None if not diffs else max(diffs) < MATCH_TOLERANCE
    return data["environment"], rows


def summary(rows: list[dict], sizes: list[int]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in rows:
        groups[e["group"]].append(e)
    out = []
    for name in [*sorted(groups), "All metrics"]:
        g = rows if name == "All metrics" else groups[name]
        lib = [e for e in g if e["kind"] == "library"]
        sp = [e["sizes"][n]["speedup"] for e in lib for n in sizes if e["sizes"][n]["speedup"]]
        out.append(
            {
                "group": name,
                "metrics": len(g),
                "library": len(lib),
                "formula": sum(e["kind"] == "formula" for e in g),
                "alone": sum(e["kind"] == "none" for e in g),
                "matching": sum(bool(e["match"]) for e in g),
                "compared": sum(e["match"] is not None for e in g),
                "faster": sum(s >= 1 for s in sp),
                "measured": len(sp),
                "geomean": geomean(sp),
                "min": min(sp) if sp else None,
                "max": max(sp) if sp else None,
            }
        )
    return out


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.3f}" if v < 100 else f"{v:,.1f}"


def fmt_x(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}×"


def markdown(env: dict, rows: list[dict]) -> str:
    sizes = sorted({n for e in rows for n in e["sizes"]})
    lines = [
        "## Every metric",
        "",
        f"One row for each of the {len(rows)} registered metrics and statistics functions "
        f"(`evalsuite benchmark --suite metrics --sizes {' '.join(map(str, sizes))}`; "
        f"EvalSuite {env['evalsuite']}, "
        f"Python {env['python']}, NumPy {env['numpy']}, SciPy {env['scipy']}, scikit-learn {env['sklearn']}, "
        f"{env['machine']}, fastest of {env['repeat']} runs). n is observations for classification, regression, "
        "clinical and statistics; pixels for segmentation; n / 1000 images for detection; n / 100 examples for "
        "text, retrieval, RAG, judge and structured-output metrics; n / 1000 examples for BERTScore, MoverScore "
        "and MAUVE; n / 100 examples for the v0.5.0 LLM-systems metrics (n / 10 for uncertainty, n / 1000 "
        "sentence pairs for bitext mining).",
        "",
        "Each metric is compared with a reference library where one exists, otherwise with an independent "
        "implementation of its textbook formula in NumPy or the Python standard library. A bare formula skips "
        "input validation, so it is a lower bound on time rather than a competitor; speed-ups are summarised "
        "over library comparisons only. Metrics with neither (learned or judge-dependent scores, randomised "
        "resampling) are timed alone.",
        "",
        "| Suite | Metrics | vs library | vs formula | Timed alone | Match | Faster than library | "
        "Geo-mean speed-up (library) | Range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summary(rows, sizes):
        rng = "–" if s["min"] is None else f"{s['min']:.2f}×–{s['max']:.2f}×"
        name = f"**{s['group']}**" if s["group"] == "All metrics" else s["group"]
        lines.append(
            f"| {name} | {s['metrics']} | {s['library']} | {s['formula']} | {s['alone']} | "
            f"{s['matching']}/{s['compared']} | {s['faster']}/{s['measured']} | {fmt_x(s['geomean'])} | {rng} |"
        )
    head = ["Metric", "Compared with"]
    for n in sizes:
        head += [f"EvalSuite ms (n={n:,})", f"Reference ms (n={n:,})", f"Speed-up (n={n:,})"]
    head += ["Max |difference|"]
    lines += ["", "| " + " | ".join(head) + " |", "| --- | --- |" + " ---: |" * (3 * len(sizes) + 1)]
    for e in rows:
        cells = [f"`{e['metric']}`", e["reference"] or "timed alone"]
        for n in sizes:
            r = e["sizes"].get(n, {})
            cells += [fmt_ms(r.get("evalsuite_ms")), fmt_ms(r.get("reference_ms")), fmt_x(r.get("speedup"))]
        cells.append("–" if e["max_diff"] is None else f"{e['max_diff']:.1e}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def site(env: dict, rows: list[dict]) -> str:
    sizes = sorted({n for e in rows for n in e["sizes"]})

    def r3(v: float | None) -> float | None:
        return None if v is None else float(f"{v:.4g}")

    payload = {
        "environment": env,
        "sizes": sizes,
        "summary": [
            {k: (r3(v) if isinstance(v, float) else v) for k, v in s.items()} for s in summary(rows, sizes)
        ],
        "rows": [
            {
                "metric": e["metric"],
                "group": e["group"],
                "reference": e["reference"],
                "kind": e["kind"],
                "maxDiff": None if e["max_diff"] is None else float(f"{e['max_diff']:.2g}"),
                "sizes": {
                    str(n): {
                        "es": r3(e["sizes"][n]["evalsuite_ms"]),
                        "ref": r3(e["sizes"][n]["reference_ms"]),
                        "speedup": r3(e["sizes"][n]["speedup"]),
                        "esMb": r3(e["sizes"][n]["evalsuite_peak_mb"]),
                    }
                    for n in sizes
                },
            }
            for e in rows
        ],
    }
    return (
        "// Generated from the evalsuite-python per-metric benchmark by scripts/bench_metrics_report.py\n"
        "// in the package repository. Do not edit by hand.\n"
        'import type { MetricBenchmarks } from "./types";\n\n'
        f"export const METRIC_BENCHMARKS: MetricBenchmarks = {json.dumps(payload, indent=2)};\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--site", help="write the website data file to this path")
    a = ap.parse_args()
    env, rows = load(a.json)
    if a.site:
        with open(a.site, "w", encoding="utf-8") as fh:
            fh.write(site(env, rows))
    else:
        sys.stdout.write(markdown(env, rows))


if __name__ == "__main__":
    main()
