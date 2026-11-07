"""Deferred imports for heavy optional-at-import-time dependencies (SciPy submodules).

``import evalsuite`` should not pay for SciPy until a function that needs it runs.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class LazyModule(ModuleType):
    """Module proxy that imports the real module on first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__dict__["_mod"] = None

    def _load(self) -> ModuleType:
        mod = self.__dict__["_mod"]
        if mod is None:
            mod = importlib.import_module(self.__name__)
            self.__dict__["_mod"] = mod
        return mod  # type: ignore[no-any-return]

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)


def lazy_module(name: str) -> Any:
    return LazyModule(name)
