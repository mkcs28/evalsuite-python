"""``evalsuite``: evaluate, compare, report, plot and benchmark from the command line.

Examples::

    evalsuite evaluate predictions.csv --y-true label --y-pred pred --y-prob prob
    evalsuite report predictions.csv --y-true label --y-pred pred --format markdown
    evalsuite compare predictions.csv --y-true label --pred lr=pred_lr --pred rf=pred_rf \\
        --prob lr=p_lr --prob rf=p_rf
    evalsuite plot roc predictions.csv --y-true label --y-prob prob --output roc.png
    evalsuite metrics --category regression
    evalsuite info classification.mcc
    evalsuite benchmark --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from typing import Any, Optional

from ..core.exceptions import EvalSuiteError
from ..version import __version__

FORMATS = ("text", "json", "csv", "markdown", "latex", "html")
_EXT = {
    ".json": "json",
    ".csv": "csv",
    ".md": "markdown",
    ".tex": "latex",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
}


class CLIError(Exception):
    """A user-facing error: printed without a traceback, exit code 2."""


# ---- input ---------------------------------------------------------------------------------------
def read_table(path: str) -> Any:
    import pandas as pd

    if path == "-":
        return pd.read_csv(sys.stdin)
    if not os.path.exists(path):
        raise CLIError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tsv", ".tab"):
            return pd.read_csv(path, sep="\t")
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext in (".json", ".jsonl"):
            return pd.read_json(path, lines=ext == ".jsonl")
        return pd.read_csv(path)
    except ImportError as exc:
        raise CLIError(f"Reading {ext} files needs an extra package: {exc}") from exc
    except Exception as exc:
        raise CLIError(f"Could not read {path}: {exc}") from exc


def column(df: Any, name: str, role: str) -> Any:
    if name not in df.columns:
        raise CLIError(
            f"Column '{name}' ({role}) not found. Columns in the file: {', '.join(map(str, df.columns))}."
        )
    return df[name].to_numpy()


def columns(df: Any, names: Sequence[str], role: str) -> Any:
    import numpy as np

    cols = [column(df, n, role) for n in names]
    return cols[0] if len(cols) == 1 else np.column_stack(cols)


def pairs(items: Optional[Sequence[str]], flag: str) -> dict[str, list[str]]:
    """``name=col`` or ``name=col1,col2,col3`` (multiclass probabilities) -> {name: [cols]}."""
    out: dict[str, list[str]] = {}
    for item in items or []:
        name, sep, cols = item.partition("=")
        if not sep or not name or not cols:
            raise CLIError(f"{flag} expects NAME=COLUMN (or NAME=COL1,COL2,... for multiclass); got '{item}'.")
        out[name] = [c for c in cols.split(",") if c]
    return out


def metric_list(text: Optional[str]) -> Optional[list[str]]:
    return [m.strip() for m in text.split(",") if m.strip()] if text else None


def label_value(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return text


# ---- output --------------------------------------------------------------------------------------
def render(obj: Any, fmt: str, digits: int) -> str:
    if fmt == "text":
        return obj.summary(digits=digits) if hasattr(obj, "summary") else str(obj)
    if fmt == "json":
        return str(obj.to_json())
    if fmt == "csv":
        return str(obj.to_csv())
    if fmt == "markdown":
        return str(obj.to_markdown(digits=digits))
    if fmt == "latex":
        return str(obj.to_latex(digits=digits))
    return str(obj.to_html(digits=digits, full=True))


def emit(obj: Any, args: argparse.Namespace) -> None:
    fmt = args.format
    if fmt is None:
        fmt = _EXT.get(os.path.splitext(args.output)[1].lower(), "text") if args.output else "text"
    text = render(obj, fmt, args.digits)
    if not text.endswith("\n"):
        text += "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print(f"Wrote {fmt} to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def add_output(p: argparse.ArgumentParser, digits: int = 4) -> None:
    p.add_argument(
        "--format",
        "-f",
        choices=FORMATS,
        default=None,
        help="output format (default: from --output extension, else text)",
    )
    p.add_argument("--output", "-o", help="write to this file instead of standard output")
    p.add_argument("--digits", type=int, default=digits, help=f"decimal places (default {digits})")


def add_targets(p: argparse.ArgumentParser, pred_required: bool = False) -> None:
    p.add_argument(
        "file", help="CSV/TSV/Parquet/JSON file with one row per observation ('-' reads CSV from stdin)"
    )
    p.add_argument("--y-true", required=True, metavar="COL", help="column with the true labels or values")
    p.add_argument(
        "--y-pred", required=pred_required, metavar="COL", help="column with predicted labels or values"
    )
    p.add_argument(
        "--y-prob",
        nargs="+",
        metavar="COL",
        help="probability column(s): one for P(positive), or one per class in label order",
    )
    p.add_argument("--weight", metavar="COL", help="column with sample weights")


# ---- commands ------------------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    import evalsuite as es

    df = read_table(args.file)
    if not args.y_pred and not args.y_prob:
        raise CLIError("Give --y-pred, --y-prob, or both.")
    result = es.evaluate(
        column(df, args.y_true, "--y-true"),
        column(df, args.y_pred, "--y-pred") if args.y_pred else None,
        y_prob=columns(df, args.y_prob, "--y-prob") if args.y_prob else None,
        task=args.task,
        metrics=metric_list(args.metrics),
        average=args.average,
        pos_label=label_value(args.pos_label),
        sample_weight=column(df, args.weight, "--weight") if args.weight else None,
    )
    emit(result, args)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    import evalsuite as es

    df = read_table(args.file)
    report = es.classification_report(
        column(df, args.y_true, "--y-true"),
        column(df, args.y_pred, "--y-pred"),
        sample_weight=column(df, args.weight, "--weight") if args.weight else None,
    )
    emit(report, args)
    return 0


def cmd_diagnostic(args: argparse.Namespace) -> int:
    import evalsuite as es

    df = read_table(args.file)
    report = es.diagnostic_report(
        column(df, args.y_true, "--y-true"),
        column(df, args.y_pred, "--y-pred"),
        pos_label=label_value(args.pos_label),
        level=args.level,
    )
    emit(report, args)
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    import evalsuite as es

    df = read_table(args.file)
    if not args.y_prob or len(args.y_prob) != 1:
        raise CLIError("calibration needs exactly one --y-prob column (predicted risk of the positive class).")
    report = es.calibration_report(
        column(df, args.y_true, "--y-true"),
        column(df, args.y_prob[0], "--y-prob"),
        n_bins=args.bins,
        strategy=args.strategy,
        n_groups=args.groups,
        pos_label=label_value(args.pos_label),
    )
    emit(report, args)
    return 0


def read_masks(path: str, name: str) -> Any:
    """Masks from .npy / .npz (first array, or the array named like the file stem) or a folder of image
    files (PNG, TIFF, ...; read with Pillow, sorted by file name)."""
    import numpy as np

    if not os.path.exists(path):
        raise CLIError(f"{name}: no such file or folder: {path}")
    if os.path.isdir(path):
        try:
            from PIL import Image
        except ImportError as exc:
            raise CLIError(f"{name}: reading image folders needs Pillow (pip install pillow).") from exc
        files = sorted(
            f for f in os.listdir(path) if f.lower().endswith((".png", ".tif", ".tiff", ".bmp", ".gif", ".jpg"))
        )
        if not files:
            raise CLIError(f"{name}: no image files in {path}.")
        return [np.asarray(Image.open(os.path.join(path, f))) for f in files]
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if not data.files:
                raise CLIError(f"{name}: {path} contains no arrays.")
            return data[data.files[0]]
    raise CLIError(f"{name}: use a .npy or .npz file, or a folder of mask images.")


def cmd_segmentation(args: argparse.Namespace) -> int:
    import evalsuite as es

    spacing = tuple(float(x) for x in args.spacing.split(",")) if args.spacing else None
    report = es.segmentation_report(
        read_masks(args.y_true, "y_true"),
        read_masks(args.y_pred, "y_pred"),
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        spacing=spacing,
        aggregate=args.aggregate,
        include_background=not args.no_background,
    )
    emit(report, args)
    if args.plot:
        _save_figure(lambda ax: es.plot.per_class(_report_iou(report), ax=ax), args.plot)
    return 0


def _report_iou(report: Any) -> Any:
    import numpy as np

    from ..core.result import MetricResult

    return MetricResult(
        "iou", "IoU", np.array([r["iou"] for r in report.rows]), labels=tuple(r["name"] for r in report.rows)
    )


def cmd_detection(args: argparse.Namespace) -> int:
    import evalsuite as es

    y_true, y_pred = es.from_coco(args.ground_truth, args.results)
    report = es.detection_report(y_true, y_pred, box_format="xywh", max_detections=args.max_detections)
    emit(report, args)
    if args.plot:
        _save_figure(
            lambda ax: es.plot.detection_pr(y_true, y_pred, ax=ax, box_format="xywh", iou_threshold=0.5), args.plot
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import evalsuite as es

    df = read_table(args.file)
    preds = {k: columns(df, v, "--pred") for k, v in pairs(args.pred, "--pred").items()}
    probs = {k: columns(df, v, "--prob") for k, v in pairs(args.prob, "--prob").items()}
    if len(set(preds) | set(probs)) < 2:
        raise CLIError(
            "compare needs at least two models: --pred NAME=COL (and/or --prob NAME=COL), twice or more."
        )
    result = es.compare(
        column(df, args.y_true, "--y-true"),
        preds or None,
        probabilities=probs or None,
        metrics=metric_list(args.metrics),
        baseline=args.baseline,
        level=args.level,
        alpha=args.alpha,
        n_resamples=args.resamples,
        correction=args.correction,
        random_state=args.seed,
    )
    emit(result, args)
    if args.plot:
        _save_figure(lambda ax: es.plot.comparison(result, ax=ax), args.plot)
    return 0


def _save_figure(draw: Any, path: str) -> None:
    from ..plot import _plt

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 4.4), layout="constrained")
    draw(ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Wrote figure to {path}", file=sys.stderr)


def cmd_plot(args: argparse.Namespace) -> int:
    import evalsuite as es

    df = read_table(args.file)
    y = column(df, args.y_true, "--y-true")
    kind = args.kind
    if kind in ("roc", "pr", "calibration", "decision"):
        if not args.y_prob:
            raise CLIError(f"'{kind}' needs --y-prob.")
        prob = columns(df, args.y_prob, "--y-prob")
        fn: Any = {
            "roc": es.plot.roc,
            "pr": es.plot.pr,
            "calibration": es.plot.calibration,
            "decision": es.plot.decision_curve,
        }[kind]
        _save_figure(lambda ax: fn(y, prob, ax=ax), args.output)
    else:
        if not args.y_pred:
            raise CLIError(f"'{kind}' needs --y-pred.")
        pred = column(df, args.y_pred, "--y-pred")
        if kind == "confusion":
            _save_figure(
                lambda ax: es.plot.confusion_matrix(y, pred, ax=ax, normalize=args.normalize), args.output
            )
        else:
            _save_figure(
                lambda ax: es.plot.residuals(
                    y, pred, ax=ax, kind="predicted" if kind == "predicted" else "residuals"
                ),
                args.output,
            )
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    import evalsuite as es

    for mid in es.list_metrics(args.category):
        info = es.metric_info(mid)
        print(f"{mid:<45} {info.name}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import evalsuite as es

    try:
        info = es.metric_info(args.metric)
    except KeyError as exc:
        raise CLIError(str(exc).strip("'\"")) from exc
    better = {True: "higher is better", False: "lower is better", None: "closer to 0 is better"}
    print(f"{info.name}  ({info.id})\n")
    print(f"Definition:  {info.definition}")
    print(f"Formula:     {info.formula}")
    print(f"Range:       {info.range} ({better[info.higher_is_better]})")
    print(f"Task:        {info.task}")
    print(f"Needs:       {', '.join(info.input_requirements)}")
    print(f"Python:      es.{info.id.split('.')[-1]}(...)")
    print("References:")
    for ref in info.references:
        print(f"  - {ref}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from ..benchmarks import run_benchmarks

    sizes = (1_000, 10_000) if args.quick else tuple(args.sizes)
    result = run_benchmarks(
        sizes=sizes,
        repeat=args.repeat,
        compare_sklearn=not args.no_sklearn,
        random_state=args.seed,
        suite=args.suite,
    )
    emit(result, args)
    return 0


# ---- parser --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalsuite",
        description="EvalSuite: unified, reproducible evaluation for machine learning and research.",
        epilog="Run 'evalsuite COMMAND --help' for the options of each command.",
    )
    parser.add_argument("--version", action="version", version=f"evalsuite-python {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("evaluate", help="evaluate predictions in a file with a standard or chosen set of metrics")
    add_targets(p)
    p.add_argument("--task", choices=("classification", "regression"), help="default: inferred from the data")
    p.add_argument("--metrics", help="comma-separated metric names, e.g. accuracy,f1,mcc")
    p.add_argument("--average", default="auto", help="binary, micro, macro, weighted, samples (default: auto)")
    p.add_argument("--pos-label", help="positive class for binary metrics (default: 1)")
    add_output(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser(
        "report", help="per-class classification report (precision, recall, F1, specificity, support)"
    )
    add_targets(p, pred_required=True)
    add_output(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="compare models on the same test set with CIs and paired tests")
    p.add_argument("file", help="CSV/TSV/Parquet/JSON file with one row per observation")
    p.add_argument("--y-true", required=True, metavar="COL")
    p.add_argument("--pred", action="append", metavar="NAME=COL", help="a model's predictions (repeat per model)")
    p.add_argument(
        "--prob", action="append", metavar="NAME=COL[,COL...]", help="a model's probabilities (repeat per model)"
    )
    p.add_argument("--metrics", help="comma-separated metric names")
    p.add_argument("--baseline", help="compare every model with this one instead of all pairs")
    p.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
    p.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    p.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples (default 1000)")
    p.add_argument("--correction", default="holm", choices=("holm", "bonferroni", "hochberg", "bh", "by"))
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0, for reproducibility)")
    p.add_argument("--plot", metavar="PATH", help="also save a forest plot (needs matplotlib)")
    add_output(p, digits=3)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser(
        "diagnostic",
        help="diagnostic accuracy of a binary test: sensitivity, specificity, PPV, NPV, LR+, LR−, DOR with CIs",
    )
    add_targets(p, pred_required=True)
    p.add_argument("--pos-label", help="positive class (default: 1)")
    p.add_argument("--level", type=float, default=0.95, help="confidence level (default 0.95)")
    add_output(p, digits=3)
    p.set_defaults(func=cmd_diagnostic)

    p = sub.add_parser(
        "calibration", help="calibration of predicted risks: Brier, ECE, MCE, intercept, slope, Hosmer–Lemeshow"
    )
    add_targets(p)
    p.add_argument("--bins", type=int, default=10, help="bins for the calibration curve, ECE and MCE")
    p.add_argument("--strategy", choices=("uniform", "quantile"), default="uniform")
    p.add_argument("--groups", type=int, default=10, help="Hosmer–Lemeshow risk groups (default 10)")
    p.add_argument("--pos-label", help="positive class (default: 1)")
    add_output(p)
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser(
        "segmentation",
        help="segmentation masks: Dice, IoU, mIoU, Boundary IoU, HD95, ASSD per class",
    )
    p.add_argument("y_true", help="true masks: .npy/.npz (images on the first axis) or a folder of mask images")
    p.add_argument("y_pred", help="predicted masks, same layout as y_true")
    p.add_argument("--num-classes", type=int, help="number of classes (default: largest label + 1)")
    p.add_argument("--ignore-index", type=int, help="label to ignore, e.g. 255")
    p.add_argument("--spacing", help="pixel spacing per axis for surface distances, e.g. 0.8,0.8")
    p.add_argument("--aggregate", choices=("dataset", "image"), default="dataset")
    p.add_argument("--no-background", action="store_true", help="leave class 0 out")
    p.add_argument("--plot", metavar="PATH", help="also save a per-class IoU bar chart")
    add_output(p)
    p.set_defaults(func=cmd_segmentation)

    p = sub.add_parser(
        "detection", help="object detection, COCO protocol: mAP, mAP@.50, mAP@.75, mAR, AP per class"
    )
    p.add_argument("ground_truth", help="COCO ground-truth JSON (images, annotations, categories)")
    p.add_argument("results", help="COCO results JSON (image_id, category_id, bbox, score)")
    p.add_argument("--max-detections", type=int, default=100)
    p.add_argument("--plot", metavar="PATH", help="also save precision-recall curves per class (IoU 0.5)")
    add_output(p, digits=3)
    p.set_defaults(func=cmd_detection)

    p = sub.add_parser(
        "plot", help="save a ROC, PR, calibration, decision-curve, confusion-matrix or residual plot"
    )
    p.add_argument("kind", choices=("roc", "pr", "calibration", "decision", "confusion", "residuals", "predicted"))
    add_targets(p)
    p.add_argument("--normalize", choices=("true", "pred", "all"), help="confusion matrix normalisation")
    p.add_argument("--output", "-o", required=True, help="image file (.png, .pdf, .svg)")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("metrics", help="list available metrics")
    p.add_argument("--category", choices=("classification", "regression", "clinical", "calibration"))
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("info", help="show a metric's definition, formula, range and references")
    p.add_argument("metric", help="metric id, e.g. classification.mcc")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("benchmark", help="time and memory benchmarks (against scikit-learn if installed)")
    p.add_argument("--sizes", type=int, nargs="+", default=[1_000, 100_000, 1_000_000])
    p.add_argument("--quick", action="store_true", help="small sizes only (1k and 10k)")
    p.add_argument("--repeat", type=int, default=5, help="timed repetitions per case; the fastest is reported")
    p.add_argument(
        "--suite",
        choices=("all", "core", "clinical", "vision"),
        default="all",
        help="core: classification/regression; clinical: clinical, calibration, tests; vision: segmentation, "
        "detection (default all)",
    )
    p.add_argument("--no-sklearn", action="store_true", help="time EvalSuite only (skip reference libraries)")
    p.add_argument("--seed", type=int, default=0)
    add_output(p, digits=3)
    p.set_defaults(func=cmd_benchmark)
    return parser


def _unicode_safe_streams() -> None:
    """Results contain characters such as −, χ², ≥ and ×. On a stream whose encoding cannot represent them
    (for example cp1252 when output is piped on Windows), write UTF-8 to files and pipes, and replace
    unrepresentable characters on interactive consoles, instead of failing."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if encoding == "utf8" or reconfigure is None:
            continue
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8")
        except (OSError, ValueError):  # pragma: no cover - exotic streams
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _unicode_safe_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (CLIError, EvalSuiteError, ValueError) as exc:
        print(f"evalsuite {args.command}: error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:  # e.g. piped into `head`
        return 0
