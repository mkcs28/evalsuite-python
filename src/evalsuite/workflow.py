"""Workflow integrations: log results to MLflow or Weights & Biases, keep a history of evaluations and flag
regressions, and serve EvalSuite over HTTP.

>>> import evalsuite as es
>>> es.log_to_mlflow(result)                 # inside an active MLflow run (or pass run_id=)
>>> es.log_to_wandb(result)                  # inside wandb.init()
>>> es.record_history(result, model="v3", dataset="test-2024")   # appends to .evalsuite/history.jsonl
>>> es.check_regression(result, model="v3", dataset="test-2024") # against the latest recorded result
>>> # shell: evalsuite history list | evalsuite serve --port 8000

Logging writes each metric's value and interval bounds as scalar metrics (``accuracy``, ``accuracy_ci_low``,
…), the full result as JSON, and its HTML table, so dashboards show intervals rather than bare numbers. The
history is a plain JSON-lines file (one evaluation per line) that can live in the repository next to the code.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from collections.abc import Mapping
from typing import Any, Optional

from .core.exceptions import InputValidationError, OptionalDependencyError

__all__ = [
    "metrics_dict",
    "log_to_mlflow",
    "log_to_wandb",
    "record_history",
    "load_history",
    "check_regression",
    "serve",
]


def metrics_dict(result: Any, *, prefix: str = "") -> dict[str, float]:
    """Flat ``{name: value, name_ci_low: …, name_ci_high: …}`` of an EvalSuite result (evaluation, metric, report
    with ``metrics``); non-finite values are left out."""
    out: dict[str, float] = {}

    def put(name: str, v: Any) -> None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            out[f"{prefix}{name}"] = f

    metrics = getattr(result, "metrics", None)
    if isinstance(metrics, Mapping):
        for key, m in metrics.items():
            if isinstance(m, Mapping):  # report-style {value, low, high}
                put(key, m.get("value"))
                put(f"{key}_ci_low", m.get("low"))
                put(f"{key}_ci_high", m.get("high"))
                continue
            name = getattr(m, "metric", None) or key
            put(name, getattr(m, "value", None))
            params = getattr(m, "params", {}) or {}
            put(f"{name}_ci_low", params.get("ci_low"))
            put(f"{name}_ci_high", params.get("ci_high"))
    elif hasattr(result, "value"):
        name = getattr(result, "metric", None) or "value"
        put(name, result.value)
        params = getattr(result, "params", {}) or {}
        put(f"{name}_ci_low", params.get("ci_low"))
        put(f"{name}_ci_high", params.get("ci_high"))
    if not out:
        raise InputValidationError("No numeric metrics found in this result.")
    return out


def log_to_mlflow(
    result: Any, *, prefix: str = "", run_id: Optional[str] = None, artifact: str = "evalsuite"
) -> dict[str, float]:
    """Log a result to MLflow: scalar metrics (with interval bounds), ``<artifact>.json`` and ``<artifact>.html``.
    Uses the active run, or ``run_id``. Returns the logged metrics."""
    try:
        import mlflow  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise OptionalDependencyError("mlflow", "mlflow", "log_to_mlflow") from exc
    flat = metrics_dict(result, prefix=prefix)
    client_run = mlflow.start_run(run_id=run_id) if run_id else None
    try:
        mlflow.log_metrics(flat)
        if hasattr(result, "to_dict"):
            mlflow.log_dict(result.to_dict(), f"{artifact}.json")
        if hasattr(result, "to_html"):
            mlflow.log_text(str(result.to_html()), f"{artifact}.html")
    finally:
        if client_run is not None:
            mlflow.end_run()
    return flat


def log_to_wandb(result: Any, *, prefix: str = "", table: str = "evalsuite") -> dict[str, float]:
    """Log a result to the active Weights & Biases run: scalar metrics (with interval bounds) and a table of the
    metrics. Returns the logged metrics."""
    try:
        import wandb  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:
        raise OptionalDependencyError("wandb", "wandb", "log_to_wandb") from exc
    if getattr(wandb, "run", None) is None:
        raise InputValidationError("No active Weights & Biases run; call wandb.init() first.")
    flat = metrics_dict(result, prefix=prefix)
    rows = [[k, v] for k, v in flat.items()]
    wandb.log({**flat, table: wandb.Table(columns=["metric", "value"], data=rows)})
    return flat


# ----------------------------------------------------------------------------------------------- history


def _history_path(path: Optional[str]) -> str:
    return path or os.environ.get("EVALSUITE_HISTORY") or os.path.join(".evalsuite", "history.jsonl")


def record_history(
    result: Any,
    *,
    model: str,
    dataset: str = "default",
    path: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Append one evaluation (model version, data set, timestamp, metrics with intervals) to the history file
    (``.evalsuite/history.jsonl`` by default, or ``$EVALSUITE_HISTORY``)."""
    from .version import __version__

    entry = {
        "time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": str(model),
        "dataset": str(dataset),
        "metrics": metrics_dict(result),
        "evalsuite": __version__,
    }
    if note:
        entry["note"] = note
    p = _history_path(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def load_history(
    *, model: Optional[str] = None, dataset: Optional[str] = None, path: Optional[str] = None
) -> list[dict[str, Any]]:
    """Recorded evaluations, oldest first, optionally for one model and / or data set."""
    p = _history_path(path)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except ValueError:
                raise InputValidationError(f"{p} line {i + 1} is not valid JSON.") from None
            if (model is None or e.get("model") == model) and (dataset is None or e.get("dataset") == dataset):
                out.append(e)
    return out


def _higher_is_better(name: str) -> bool:
    from .stats.compare import _info

    info = _info(name)
    return True if info is None or info.higher_is_better is None else bool(info.higher_is_better)


def check_regression(
    result: Any,
    *,
    model: Optional[str] = None,
    dataset: str = "default",
    baseline: Optional[Mapping[str, Any]] = None,
    path: Optional[str] = None,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare ``result`` with the latest recorded evaluation of ``model`` (any model when None) on ``dataset``,
    or with ``baseline`` (a history entry). A metric regressed when it got worse by more than ``tolerance`` and the
    new confidence interval lies entirely on the worse side of the old estimate. Returns ``{"regressed": [...],
    "improved": [...], "baseline": entry}``."""
    if baseline is None:
        hist = load_history(model=model, dataset=dataset, path=path)
        if not hist:
            return {"regressed": [], "improved": [], "baseline": None}
        baseline = hist[-1]
    old = baseline["metrics"]
    new = metrics_dict(result)
    regressed, improved = [], []
    for name, v in new.items():
        if name.endswith(("_ci_low", "_ci_high")) or name not in old:
            continue
        hib = _higher_is_better(name)
        diff = v - old[name]
        worse = -diff if hib else diff
        lo, hi = new.get(f"{name}_ci_low"), new.get(f"{name}_ci_high")
        clear = (hi is not None and hi < old[name]) if hib else (lo is not None and lo > old[name])
        better_clear = (lo is not None and lo > old[name]) if hib else (hi is not None and hi < old[name])
        row = {"metric": name, "previous": old[name], "current": v, "difference": diff}
        if worse > tolerance and clear:
            regressed.append(row)
        elif better_clear:
            improved.append(row)
    return {"regressed": regressed, "improved": improved, "baseline": baseline}


# ----------------------------------------------------------------------------------------------- HTTP service


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover - blocking server loop
    """A small JSON HTTP service (standard library only): ``GET /health``, ``GET /metrics`` (the catalogue) and
    ``POST /evaluate`` with ``{"y_true": [...], "y_pred": [...], "y_prob": [...], ...}`` (any
    :func:`evalsuite.evaluate` keyword) returning the result as JSON. Binds to localhost by default."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    server = ThreadingHTTPServer((host, port), _handler_class(BaseHTTPRequestHandler))
    print(f"EvalSuite service on http://{host}:{port} (POST /evaluate, GET /metrics, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


_MAX_BODY = 50 * 1024 * 1024


def _handler_class(base: Any) -> Any:
    import evalsuite as es

    from .version import __version__

    class Handler(base):  # type: ignore[misc, valid-type, unused-ignore]
        server_version = f"EvalSuite/{__version__}"

        def _send(self, code: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            return  # quiet by default

        def do_GET(self) -> None:  # noqa: N802 - name required by http.server
            if self.path == "/health":
                self._send(200, {"status": "ok", "version": __version__})
            elif self.path == "/metrics":
                self._send(200, {"metrics": es.list_metrics()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - name required by http.server
            if self.path != "/evaluate":
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n <= 0 or n > _MAX_BODY:
                self._send(413 if n > _MAX_BODY else 400, {"error": "missing or too large request body"})
                return
            try:
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict) or "y_true" not in req:
                    raise InputValidationError("Body must be a JSON object with y_true.")
                allowed = {
                    "y_pred",
                    "y_prob",
                    "task",
                    "metrics",
                    "average",
                    "pos_label",
                    "sample_weight",
                    "clusters",
                    "ci",
                    "level",
                    "n_resamples",
                    "random_state",
                    "baseline_pred",
                    "baseline_prob",
                }
                unknown = set(req) - allowed - {"y_true"}
                if unknown:
                    raise InputValidationError(f"Unknown field(s): {', '.join(sorted(unknown))}.")
                y_pred = req.pop("y_pred", None)
                result = es.evaluate(req.pop("y_true"), y_pred, **req)
                self._send(200, result.to_dict())
            except (ValueError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler
