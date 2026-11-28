"""Exception hierarchy.

Every message says what failed, why, and how to fix it.
"""

from __future__ import annotations

__all__ = [
    "EvalSuiteDeprecationWarning",
    "EvalSuiteError",
    "InputValidationError",
    "MetricInputError",
    "OptionalDependencyError",
    "StatisticalTestError",
    "UndefinedMetricWarning",
    "UnsupportedTaskError",
]


class EvalSuiteError(Exception):
    """Base class for all EvalSuite errors."""


class InputValidationError(EvalSuiteError, ValueError):
    """Inputs have the wrong shape, length, type or values."""


class MetricInputError(EvalSuiteError, ValueError):
    """Inputs are valid arrays but outside the domain a metric is defined on."""


class UnsupportedTaskError(EvalSuiteError, ValueError):
    """The requested task or averaging mode is not supported for this metric."""


class OptionalDependencyError(EvalSuiteError, ImportError):
    """A feature needs an optional dependency that is not installed."""

    def __init__(self, package: str, extra: str, feature: str) -> None:
        super().__init__(
            f"{feature} requires the optional dependency '{package}'. "
            f'Install it with: pip install "evalsuite-python[{extra}]"'
        )
        self.package = package
        self.extra = extra


class StatisticalTestError(EvalSuiteError, ValueError):
    """A statistical procedure cannot be applied to the given data."""


class UndefinedMetricWarning(UserWarning):
    """A metric is undefined for the input (for example a zero denominator)."""


class EvalSuiteDeprecationWarning(DeprecationWarning):
    """A public name or argument that will be removed in the next major version (see the deprecation policy)."""


def deprecated(old: str, new: str, *, since: str, stacklevel: int = 3) -> None:
    """Warn that ``old`` is deprecated in favour of ``new`` (used by EvalSuite itself; part of the policy in
    ``docs/deprecation.md``)."""
    import warnings

    warnings.warn(
        f"{old} is deprecated since EvalSuite {since} and will be removed in the next major version; use {new}.",
        EvalSuiteDeprecationWarning,
        stacklevel=stacklevel,
    )
