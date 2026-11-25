"""Use EvalSuite inside the tools you already train and track with.

* :func:`sklearn_scorer`: any EvalSuite metric as a scikit-learn scorer (``cross_val_score``,
  ``GridSearchCV``), with the right direction and probability handling.
* :func:`hf_compute_metrics`: a ``compute_metrics`` function for the Hugging Face ``Trainer``.
* :func:`lightning_callback` / :func:`keras_callback`: evaluate on a validation set at the end of every epoch
  and log the metrics (with confidence intervals) to the framework's logger.
* :func:`log_mlflow` / :func:`log_wandb`: log an :func:`evalsuite.evaluate` result, intervals and baseline
  tests included, plus the full JSON report as an artifact.

None of these frameworks is a dependency: each is imported only when its helper is used.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np

from .core.exceptions import InputValidationError, OptionalDependencyError


def _import(module: str, extra: str, what: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise OptionalDependencyError(module, extra, what) from exc


def _resolve(metric: str) -> tuple[Callable[..., Any], bool, bool]:
    """(function, needs probabilities, higher is better) for a metric name."""
    from .core.registry import list_metrics, metric_info
    from .stats._resolve import resolve_metric

    fn, name = resolve_metric(metric)
    ids = [m for m in list_metrics() if m.split(".")[-1] == name]
    info = metric_info(ids[0]) if ids else None
    needs_prob = info is not None and "y_prob" in info.input_requirements
    higher = True if info is None else bool(info.higher_is_better)
    return fn, needs_prob, higher


def flatten(result: Any, prefix: str = "") -> dict[str, float]:
    """Scalar metrics of an :func:`evaluate` result as a flat dict (``f1``, ``f1_ci_low``, ``f1_ci_high``,
    ``f1_vs_baseline``, ``f1_p_value``), the form experiment trackers log."""
    out: dict[str, float] = {}
    metrics = result.metrics if hasattr(result, "metrics") else {result.metric: result}
    for key, m in metrics.items():
        if isinstance(m.value, np.ndarray):
            for lab, v in zip(m.labels or range(len(m.value)), m.value):
                out[f"{prefix}{key}_{lab}"] = float(v)
            continue
        out[f"{prefix}{key}"] = float(m.value)
        p = m.params
        for src, dst in (
            ("ci_low", "ci_low"),
            ("ci_high", "ci_high"),
            ("baseline_difference", "vs_baseline"),
            ("baseline_p_adjusted", "p_value"),
        ):
            if src in p:
                out[f"{prefix}{key}_{dst}"] = float(p[src])
    return out


# ------------------------------------------------------------------------------------------- scikit-learn


def sklearn_scorer(metric: str, **metric_kwargs: Any) -> Any:
    """A scikit-learn scorer for an EvalSuite metric, e.g.
    ``cross_val_score(model, X, y, scoring=sklearn_scorer("mcc"))``.

    Probability metrics (ROC AUC, log loss, Brier, ...) use ``predict_proba`` (the positive-class column for
    binary problems); lower-is-better metrics are negated, as scikit-learn expects."""
    metrics_mod = _import("sklearn.metrics", "sklearn", "sklearn_scorer")
    fn, needs_prob, higher = _resolve(metric)

    def score(y_true: Any, y: Any, **_: Any) -> float:
        y = np.asarray(y)
        if needs_prob and y.ndim == 2 and y.shape[1] == 2:
            y = y[:, 1]
        r = fn(y_true, y_prob=y, **metric_kwargs) if needs_prob else fn(y_true, y, **metric_kwargs)
        return float(r)

    method = "predict_proba" if needs_prob else "predict"
    try:
        return metrics_mod.make_scorer(score, greater_is_better=higher, response_method=method)
    except TypeError:  # scikit-learn < 1.4
        return metrics_mod.make_scorer(score, greater_is_better=higher, needs_proba=needs_prob)


# ------------------------------------------------------------------------------------------- Hugging Face


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def hf_compute_metrics(
    metrics: Optional[Sequence[str]] = None, *, ci: Any = False, **evaluate_kwargs: Any
) -> Callable[[Any], dict[str, float]]:
    """``Trainer(compute_metrics=hf_compute_metrics())``: evaluate classification logits with EvalSuite.

    Logits become labels (argmax) and probabilities (softmax; the positive column for two classes). Returns
    the flat dict the Trainer logs; ``ci="analytic"`` adds ``*_ci_low`` / ``*_ci_high`` entries."""
    from .api import evaluate

    def compute(eval_pred: Any) -> dict[str, float]:
        logits, labels = (
            (eval_pred.predictions, eval_pred.label_ids) if hasattr(eval_pred, "predictions") else eval_pred
        )
        logits = np.asarray(logits[0] if isinstance(logits, tuple) else logits, dtype=float)
        labels = np.asarray(labels)
        if logits.ndim != 2:
            raise InputValidationError("hf_compute_metrics expects (n_examples, n_classes) logits.")
        pred = logits.argmax(axis=-1)
        prob = _softmax(logits)
        if prob.shape[1] == 2:
            prob = prob[:, 1]
        result = evaluate(labels, pred, y_prob=prob, metrics=metrics, ci=ci, **evaluate_kwargs)
        return flatten(result)

    return compute


# ------------------------------------------------------------------------------------------- callbacks


def _predict(model: Any, x: Any) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Labels and (binary: positive-class) probabilities from a model's raw outputs."""
    out = np.asarray(
        model(x) if callable(model) and not hasattr(model, "predict") else model.predict(x, verbose=0)
    )
    if out.ndim == 2 and out.shape[1] > 1:
        prob = out if np.allclose(out.sum(axis=1), 1.0, atol=1e-4) and out.min() >= 0 else _softmax(out)
        return prob.argmax(axis=1), (prob[:, 1] if prob.shape[1] == 2 else prob)
    p = out.reshape(-1)
    if p.min() < 0 or p.max() > 1:
        p = 1 / (1 + np.exp(-p))
    return (p >= 0.5).astype(int), p


def keras_callback(
    x_val: Any, y_val: Any, *, metrics: Optional[Sequence[str]] = None, ci: Any = "analytic", prefix: str = "val_"
) -> Any:
    """A Keras callback that evaluates ``(x_val, y_val)`` with EvalSuite after every epoch and adds the metrics
    (``val_f1``, ``val_f1_ci_low``, ...) to the epoch logs, where History, TensorBoard and other callbacks see
    them."""
    keras = _import("keras", "keras", "keras_callback")
    from .api import evaluate

    class EvalSuiteCallback(keras.callbacks.Callback):  # type: ignore[misc,name-defined]
        def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
            pred, prob = _predict(self.model, x_val)
            result = evaluate(y_val, pred, y_prob=prob, metrics=metrics, ci=ci)
            self.last_result = result
            if logs is not None:
                logs.update(flatten(result, prefix))

    return EvalSuiteCallback()


def lightning_callback(
    dataloader_fn: Callable[[Any], tuple[Any, Any, Any]],
    *,
    metrics: Optional[Sequence[str]] = None,
    ci: Any = "analytic",
    prefix: str = "val_",
) -> Any:
    """A PyTorch Lightning callback: after each validation epoch, ``dataloader_fn(pl_module)`` returns
    ``(y_true, y_pred, y_prob)`` (``y_pred`` or ``y_prob`` may be None) and the EvalSuite metrics are logged with
    ``pl_module.log_dict`` (and kept as ``callback.last_result``)."""
    try:
        pl = importlib.import_module("lightning.pytorch")
    except ImportError:
        pl = _import("pytorch_lightning", "lightning", "lightning_callback")
    from .api import evaluate

    class EvalSuiteCallback(pl.Callback):  # type: ignore[misc,name-defined]
        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            y_true, y_pred, y_prob = dataloader_fn(pl_module)
            result = evaluate(y_true, y_pred, y_prob=y_prob, metrics=metrics, ci=ci)
            self.last_result = result
            pl_module.log_dict(flatten(result, prefix))

    return EvalSuiteCallback()


# ------------------------------------------------------------------------------------------- trackers


def log_mlflow(
    result: Any, *, prefix: str = "", step: Optional[int] = None, artifact: Optional[str] = "evalsuite.json"
) -> None:
    """Log an EvalSuite result to the active MLflow run: every scalar metric with its interval and baseline
    test, and (``artifact``) the full JSON report."""
    mlflow = _import("mlflow", "mlflow", "log_mlflow")
    mlflow.log_metrics(flatten(result, prefix), step=step)
    if artifact and hasattr(result, "to_dict"):
        mlflow.log_dict(result.to_dict(), artifact)


def log_wandb(result: Any, *, prefix: str = "", run: Any = None, table: bool = True) -> None:
    """Log an EvalSuite result to Weights & Biases (the current run, or ``run``): the flat metrics, and
    (``table``) a table with each metric's value and interval."""
    wandb = _import("wandb", "wandb", "log_wandb")
    target = run if run is not None else wandb
    data: dict[str, Any] = dict(flatten(result, prefix))
    if table and hasattr(result, "metrics"):
        rows = [
            [m.name, float(m.value), m.params.get("ci_low"), m.params.get("ci_high")]
            for m in result.metrics.values()
            if not isinstance(m.value, np.ndarray)
        ]
        data[f"{prefix}evalsuite"] = wandb.Table(columns=["metric", "value", "ci_low", "ci_high"], data=rows)
    target.log(data)


__all__ = [
    "flatten",
    "hf_compute_metrics",
    "keras_callback",
    "lightning_callback",
    "log_mlflow",
    "log_wandb",
    "sklearn_scorer",
]
