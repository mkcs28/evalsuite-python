"""Metric registry: programmatic discovery of every metric and its documentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, TypeVar

__all__ = ["MetricInfo", "list_metrics", "metric_info", "register"]

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class MetricInfo:
    """Documentation for one metric. ``id`` is ``"<category>.<name>"``, e.g. ``"classification.f1"``."""

    id: str
    name: str
    category: str
    task: str
    definition: str
    formula: str
    range: str
    input_requirements: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    higher_is_better: Optional[bool] = True
    function: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("function", None)
        out["input_requirements"] = list(self.input_requirements)
        out["references"] = list(self.references)
        out["api"] = f"evalsuite.{self.id}"
        return out


_REGISTRY: dict[str, MetricInfo] = {}

# Packages whose import registers metrics. EvalSuite loads lazily, so the catalogue imports them on first use.
_METRIC_PACKAGES = (
    "evalsuite.classification",
    "evalsuite.regression",
    "evalsuite.clinical",
    "evalsuite.calibration",
    "evalsuite.text",
    "evalsuite.llmsys",
    "evalsuite.vision",
)
_loaded = False


def _load_all() -> None:
    global _loaded
    if not _loaded:
        import importlib

        for name in _METRIC_PACKAGES:
            importlib.import_module(name)
        _loaded = True


def register(
    *,
    category: str,
    task: str,
    name: str,
    definition: str,
    formula: str,
    range: str,
    input_requirements: tuple[str, ...] = (),
    references: tuple[str, ...] = (),
    higher_is_better: Optional[bool] = True,
) -> Callable[[F], F]:
    """Decorator registering a public metric function under ``<category>.<function name>``."""

    def deco(fn: F) -> F:
        metric_id = f"{category}.{fn.__name__}"
        if metric_id in _REGISTRY:
            raise RuntimeError(f"Metric {metric_id} is registered twice.")
        _REGISTRY[metric_id] = MetricInfo(
            id=metric_id,
            name=name,
            category=category,
            task=task,
            definition=definition,
            formula=formula,
            range=range,
            input_requirements=input_requirements,
            references=references,
            higher_is_better=higher_is_better,
            function=fn,
        )
        return fn

    return deco


def list_metrics(category: Optional[str] = None) -> list[str]:
    """Sorted metric ids, optionally filtered by category (``"classification"``, ``"regression"``)."""
    _load_all()
    return sorted(k for k, v in _REGISTRY.items() if category is None or v.category == category)


def metric_info(metric_id: str) -> MetricInfo:
    """Documentation for one metric, e.g. ``metric_info("classification.mcc")``."""
    _load_all()
    try:
        return _REGISTRY[metric_id]
    except KeyError:
        close = [k for k in _REGISTRY if metric_id.split(".")[-1] in k]
        hint = f" Did you mean: {', '.join(close)}?" if close else " Use evalsuite.list_metrics() to see all ids."
        raise KeyError(f"Unknown metric '{metric_id}'.{hint}") from None
