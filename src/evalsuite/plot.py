"""Publication-ready plots (optional dependency: ``pip install "evalsuite-python[plot]"``).

Every function draws on ``ax`` (or a new figure), returns the matplotlib ``Axes`` and computes its numbers
with EvalSuite's own metrics, so the plot and the reported values always agree. Several models can be passed
as a dict ``{name: values}``; they get distinct colours *and* line styles, so plots stay readable in
greyscale. Matplotlib is imported only when a plot function is called.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np

from .core.exceptions import InputValidationError, OptionalDependencyError
from .core.types import ArrayLike

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .stats.compare import ComparisonResult

__all__ = ["calibration", "comparison", "confusion_matrix", "decision_curve", "pr", "residuals", "roc"]

Scores = Union[ArrayLike, Mapping[str, ArrayLike]]
_STYLES = ("-", "--", "-.", ":")
_MARKERS = ("o", "s", "^", "D", "v", "P")


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise OptionalDependencyError("matplotlib", "plot", "Plotting") from exc
    return plt


def _axes(ax: Optional[Axes], figsize: tuple[float, float] = (5.0, 4.2)) -> Axes:
    if ax is not None:
        return ax
    _, new_ax = _plt().subplots(figsize=figsize, layout="constrained")
    return cast("Axes", new_ax)


def _models(values: Scores, label: Optional[str]) -> list[tuple[Optional[str], Any]]:
    if isinstance(values, Mapping):
        if not values:
            raise InputValidationError("Pass at least one model.")
        return [(str(k), v) for k, v in values.items()]
    return [(label, values)]


def _style(i: int) -> dict[str, Any]:
    return {"linestyle": _STYLES[i % len(_STYLES)], "linewidth": 1.8}


def roc(
    y_true: ArrayLike,
    y_prob: Scores,
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    chance: bool = True,
) -> Axes:
    """ROC curve(s) for binary probabilities, with the AUC in the legend.

    ``y_prob``: P(positive) for one model, or ``{name: probabilities}`` for several.
    """
    from .classification.metrics import roc_auc, roc_curve

    ax = _axes(ax)
    for i, (name, prob) in enumerate(_models(y_prob, label)):
        fpr, tpr, _ = roc_curve(y_true, prob, pos_label=pos_label, sample_weight=sample_weight)
        auc = float(roc_auc(y_true, prob, pos_label=pos_label, sample_weight=sample_weight))
        ax.plot(
            fpr, tpr, drawstyle="steps-post", label=f"{name + ': ' if name else ''}AUC = {auc:.3f}", **_style(i)
        )
    if chance:
        ax.plot([0, 1], [0, 1], color="0.6", linewidth=1, linestyle=":", label="chance")
    ax.set(
        xlabel="False positive rate (1 − specificity)",
        ylabel="True positive rate (sensitivity)",
        title="ROC curve",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
    )
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False)
    return ax


def pr(
    y_true: ArrayLike,
    y_prob: Scores,
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    chance: bool = True,
) -> Axes:
    """Precision-recall curve(s) with average precision in the legend; the chance line is the prevalence."""
    from .classification.metrics import _binary_target, _ctx, average_precision, pr_curve

    ax = _axes(ax)
    for i, (name, prob) in enumerate(_models(y_prob, label)):
        prec, rec, _ = pr_curve(y_true, prob, pos_label=pos_label, sample_weight=sample_weight)
        ap = float(average_precision(y_true, prob, pos_label=pos_label, sample_weight=sample_weight))
        ax.plot(
            rec, prec, drawstyle="steps-post", label=f"{name + ': ' if name else ''}AP = {ap:.3f}", **_style(i)
        )
    if chance:
        first = _models(y_prob, label)[0][1]
        ctx = _ctx(y_true, None, y_prob=first, sample_weight=sample_weight)
        prevalence = float(np.average(_binary_target(ctx, pos_label), weights=ctx.weights))
        ax.axhline(prevalence, color="0.6", linewidth=1, linestyle=":", label=f"chance ({prevalence:.2f})")
    ax.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision (PPV)",
        title="Precision-recall curve",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.03),
    )
    ax.legend(loc="lower left", frameon=False)
    return ax


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    ax: Optional[Axes] = None,
    labels: Optional[ArrayLike] = None,
    normalize: Optional[str] = None,
    sample_weight: Optional[ArrayLike] = None,
    cmap: str = "Blues",
    colorbar: bool = True,
) -> Axes:
    """Annotated confusion matrix (rows: true, columns: predicted). ``normalize``: None, "true", "pred", "all"."""
    from .classification.metrics import confusion_matrix as cm_fn
    from .core.validation import resolve_labels, to_numpy

    cm = np.asarray(cm_fn(y_true, y_pred, labels=labels, sample_weight=sample_weight, normalize=normalize))  # type: ignore[arg-type]
    names = (
        to_numpy(labels, "labels")
        if labels is not None
        else resolve_labels(to_numpy(y_true, "y_true"), to_numpy(y_pred, "y_pred"), None)
    ).tolist()
    k = cm.shape[0]
    ax = _axes(ax, (max(3.6, 0.7 * k + 2.4), max(3.2, 0.7 * k + 1.8)))
    image = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1 if normalize else None)
    threshold = (cm.max() + cm.min()) / 2
    integral = normalize is None and np.all(np.mod(cm, 1) == 0)
    for i in range(k):
        for j in range(k):
            text = f"{int(cm[i, j])}" if integral else f"{cm[i, j]:.2f}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > threshold else "black",
            )
    ax.set(
        xticks=range(k),
        yticks=range(k),
        xticklabels=names,
        yticklabels=names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix" + (f" (normalised by {normalize})" if normalize else ""),
    )
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def calibration(
    y_true: ArrayLike,
    y_prob: Scores,
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    legend_loc: str = "below",
) -> Axes:
    """Reliability diagram: observed frequency against mean predicted probability per bin, with ECE and Brier
    score in the legend; the diagonal is perfect calibration.

    ``legend_loc="below"`` (default) puts the legend under the axes so it never covers the curves; any
    matplotlib location (e.g. ``"upper left"``) places it inside instead."""
    from .classification.metrics import brier_score, calibration_curve, expected_calibration_error

    ax = _axes(ax)
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=1, linestyle=":", label="perfect calibration")
    kw = {"n_bins": n_bins, "strategy": strategy, "pos_label": pos_label, "sample_weight": sample_weight}
    for i, (name, prob) in enumerate(_models(y_prob, label)):
        frac, mean_p, _ = calibration_curve(y_true, prob, **kw)
        ece = float(expected_calibration_error(y_true, prob, **kw))
        brier = float(brier_score(y_true, prob, pos_label=pos_label, sample_weight=sample_weight))
        ax.plot(
            mean_p,
            frac,
            marker=_MARKERS[i % len(_MARKERS)],
            markersize=4,
            label=f"{name + ': ' if name else ''}ECE = {ece:.3f}, Brier = {brier:.3f}",
            **_style(i),
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed frequency of positives",
        title="Calibration",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
    )
    ax.set_aspect("equal")
    if legend_loc == "below":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize="small")
    else:
        ax.legend(loc=legend_loc, frameon=False)  # type: ignore[call-overload]
    return ax


def residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    ax: Optional[Axes] = None,
    kind: str = "residuals",
) -> Axes:
    """Regression diagnostics. ``kind="residuals"``: residual (predicted − true) against predicted value;
    ``kind="predicted"``: predicted against true with the identity line. RMSE and R² in the title."""
    from .regression.metrics import _Inputs, r2, rmse

    inp = _Inputs(y_true, y_pred, None)
    if inp.multi:
        raise InputValidationError("residuals() plots single-output targets; pass one output column at a time.")
    yt, yp = inp.y_true[:, 0], inp.y_pred[:, 0]
    ax = _axes(ax)
    stats = f"RMSE = {float(rmse(yt, yp)):.3g}, R² = {float(r2(yt, yp)):.3f}"
    if kind == "residuals":
        ax.scatter(yp, yp - yt, s=14, alpha=0.7, edgecolors="none")
        ax.axhline(0, color="0.4", linewidth=1)
        ax.set(xlabel="Predicted value", ylabel="Residual (predicted − true)", title=f"Residuals ({stats})")
    elif kind == "predicted":
        lo, hi = float(min(np.min(yt), np.min(yp))), float(max(np.max(yt), np.max(yp)))
        ax.scatter(yt, yp, s=14, alpha=0.7, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=1, linestyle=":", label="y = x")
        ax.set(xlabel="True value", ylabel="Predicted value", title=f"Predicted vs true ({stats})")
        ax.legend(loc="upper left", frameon=False)
    else:
        raise InputValidationError("kind must be 'residuals' or 'predicted'.")
    return ax


def comparison(
    result: ComparisonResult,
    *,
    metrics: Optional[list[str]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """Forest plot of an :func:`evalsuite.compare` result: each model's estimate with its confidence interval,
    grouped by metric; the best model per metric is drawn filled."""
    names = list(metrics or result.metrics)
    unknown = [m for m in names if m not in result.metrics]
    if unknown:
        raise InputValidationError(f"Metric(s) {unknown} were not compared. Compared: {list(result.metrics)}.")
    rows = [(m, model) for m in names for model in result.models]
    ax = _axes(ax, (6.0, max(2.5, 0.38 * len(rows) + 1.2)))
    for y_pos, (metric, model) in enumerate(rows):
        est = result.estimate(model, metric)
        best = result.best(metric) == model
        ax.errorbar(
            est["estimate"],
            y_pos,
            xerr=[[est["estimate"] - est["low"]], [est["high"] - est["estimate"]]],
            fmt="o",
            capsize=3,
            color="C0",
            markerfacecolor="C0" if best else "white",
            markersize=6,
        )
    ax.set_yticks(range(len(rows)), [f"{metric} · {model}" for metric, model in rows])
    ax.invert_yaxis()
    level = round(result.settings["level"] * 100, 6)
    ax.set(xlabel=f"Estimate with {level:g}% CI (filled: best per metric)", title="Model comparison")
    ax.grid(axis="x", color="0.9")
    return ax


def decision_curve(
    y_true: ArrayLike,
    y_prob: Scores,
    *,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    thresholds: Optional[ArrayLike] = None,
    pos_label: Any = None,
    sample_weight: Optional[ArrayLike] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Axes:
    """Decision curve: net benefit against threshold probability for each model, with *treat all* and
    *treat none* references (Vickers & Elkin 2006). Values come from :func:`evalsuite.decision_curve`.

    By default the y-axis runs from a little below 0 to a little above the prevalence, where the
    clinically meaningful part of the curve lies; pass ``ylim`` to change it."""
    from .clinical.report import decision_curve as _dc

    models = dict(_models(y_prob, label))
    names = [n if n is not None else "model" for n in models]
    dc = _dc(
        y_true,
        dict(zip(names, models.values())),
        thresholds=thresholds,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )
    ax = _axes(ax)
    t = dc.thresholds
    for i, name in enumerate(dc.net_benefit):
        ax.plot(t, dc.net_benefit[name], label=name, **_style(i))
    ax.plot(t, dc.treat_all, color="0.45", linewidth=1.2, linestyle="--", label="treat all")
    ax.plot(t, dc.treat_none, color="0.2", linewidth=1.2, linestyle=":", label="treat none")
    top = max(dc.prevalence, max(float(np.nanmax(v)) for v in dc.net_benefit.values()))
    ax.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title="Decision curve",
        xlim=(float(t[0]), float(t[-1])),
        ylim=ylim if ylim is not None else (-0.05 * max(top, 0.05), top * 1.1 + 0.01),
    )
    ax.legend(loc="upper right", frameon=False)
    return ax
