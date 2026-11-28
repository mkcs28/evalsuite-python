"""``df.evalsuite`` on pandas and Polars DataFrames.

>>> import evalsuite as es
>>> es.register_accessors()                       # once; or: import evalsuite.dataframe
>>> df.evalsuite.evaluate("label", "pred", y_prob="risk")
>>> df.evalsuite.compare("label", {"lr": "pred_lr", "rf": "pred_rf"})
>>> df.evalsuite.report()                          # automatic HTML report
>>> train.evalsuite.drift(test); test.evalsuite.card(target="label"); train.evalsuite.leakage(test)

Column arguments are column names. Registration is explicit (importing EvalSuite never imports pandas or
Polars); importing this module registers the accessor for whichever of the two is installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

__all__ = ["register_accessors", "EvalSuiteAccessor"]


class EvalSuiteAccessor:
    """The ``.evalsuite`` namespace of a DataFrame."""

    def __init__(self, df: Any) -> None:
        self._df = df

    def _col(self, name: Optional[str]) -> Any:
        if name is None:
            return None
        try:
            return self._df[name].to_numpy()
        except Exception as exc:
            from .core.exceptions import InputValidationError

            raise InputValidationError(f"No column {name!r} in this DataFrame.") from exc

    def _cols(self, names: Union[str, Sequence[str], None]) -> Any:
        if names is None or isinstance(names, str):
            return self._col(names)
        import numpy as np

        return np.column_stack([self._col(n) for n in names])

    def evaluate(
        self,
        y_true: str,
        y_pred: Optional[str] = None,
        *,
        y_prob: Union[str, Sequence[str], None] = None,
        **kwargs: Any,
    ) -> Any:
        """:func:`evalsuite.evaluate` on columns (``clusters=`` / ``sample_weight=`` may also be column names)."""
        import evalsuite as es

        for key in ("clusters", "sample_weight", "baseline_pred"):
            if isinstance(kwargs.get(key), str):
                kwargs[key] = self._col(kwargs[key])
        return es.evaluate(self._col(y_true), self._col(y_pred), y_prob=self._cols(y_prob), **kwargs)

    def compare(
        self,
        y_true: str,
        predictions: Optional[Mapping[str, str]] = None,
        *,
        probabilities: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
        **kwargs: Any,
    ) -> Any:
        """:func:`evalsuite.compare` with model name → column name mappings."""
        import evalsuite as es

        preds = {m: self._col(c) for m, c in (predictions or {}).items()} or None
        probs = {m: self._cols(c) for m, c in (probabilities or {}).items()} or None
        return es.compare(self._col(y_true), preds, probabilities=probs, **kwargs)

    def report(self, **kwargs: Any) -> Any:
        """:func:`evalsuite.auto_report` on this DataFrame."""
        import evalsuite as es

        return es.auto_report(self._df, **kwargs)

    def drift(self, current: Any, **kwargs: Any) -> Any:
        """:func:`evalsuite.drift_report` with this DataFrame as the reference."""
        import evalsuite as es

        return es.drift_report(self._df, current, **kwargs)

    def leakage(self, test: Any, **kwargs: Any) -> Any:
        """:func:`evalsuite.leakage_report` with this DataFrame as the training set."""
        import evalsuite as es

        return es.leakage_report(self._df, test, **kwargs)

    def card(self, **kwargs: Any) -> Any:
        """:func:`evalsuite.dataset_card` of this DataFrame."""
        import evalsuite as es

        return es.dataset_card(self._df, **kwargs)


def register_accessors() -> list[str]:
    """Register ``df.evalsuite`` for pandas and / or Polars (whichever is installed); returns which."""
    done = []
    try:
        import pandas as pd

        if not hasattr(pd.DataFrame, "evalsuite"):
            pd.api.extensions.register_dataframe_accessor("evalsuite")(EvalSuiteAccessor)
        done.append("pandas")
    except ImportError:
        pass
    try:
        import polars as pl

        if not hasattr(pl.DataFrame, "evalsuite"):
            pl.api.register_dataframe_namespace("evalsuite")(EvalSuiteAccessor)
        done.append("polars")
    except ImportError:
        pass
    return done


register_accessors()
