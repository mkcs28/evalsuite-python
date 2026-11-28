"""EvalSuite's cache folder: where downloaded benchmark data and the update-check record live.

>>> import evalsuite as es
>>> es.cache_info()                       # location, size, entries
>>> es.clear_cache(older_than="30d")      # or clear_cache() for everything

The folder is ``$EVALSUITE_CACHE`` if set, else ``$XDG_CACHE_HOME/evalsuite``, ``%LOCALAPPDATA%\\evalsuite`` on
Windows, or ``~/.cache/evalsuite``. From the command line: ``evalsuite cache info`` and ``evalsuite cache clear``.

Automatic clearing: at most once a day, in the background after import, entries not used for
``EVALSUITE_CACHE_MAX_AGE`` (default 90 days) are removed, then the least recently used entries until the cache is
below ``EVALSUITE_CACHE_MAX_SIZE`` (default 2 GB). Set ``EVALSUITE_CACHE_AUTO_CLEAN=0`` to turn it off. Only
folders EvalSuite created (they carry a marker file) are ever touched, and only whole entries (one data set
download, one record) are removed, never part of one.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .core.exceptions import InputValidationError

__all__ = ["CacheInfo", "cache_dir", "cache_info", "clear_cache"]

MARKER = ".evalsuite-cache"
_STAMP = ".last-clean"
_KEEP = {MARKER, _STAMP}
_DEFAULT_MAX_AGE = 90 * 86400.0
_DEFAULT_MAX_SIZE = 2 * 1024**3


def cache_dir(*, create: bool = False) -> Path:
    """The cache folder (see the module docstring); ``create=True`` makes it, with its marker file."""
    from ._update import _cache_root

    path = _cache_root()
    if create:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / MARKER
        if not marker.exists():
            marker.write_text("Created by EvalSuite. Everything in this folder can be deleted safely.\n")
    return path


def _parse_age(value: Union[str, float, int, None]) -> Optional[float]:
    """Seconds from 3600, "30d", "12h", "45m", "90s", "2w" (None stays None)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise InputValidationError("Ages must be non-negative.")
        return float(value)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhdw]?)\s*", str(value).lower())
    if not m:
        raise InputValidationError(f"Cannot read the age {value!r}; use e.g. '30d', '12h', '2w' or seconds.")
    unit = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}[m.group(2)]
    return float(m.group(1)) * unit


def _parse_size(value: Union[str, int, None]) -> Optional[int]:
    """Bytes from 1048576, "500MB", "2GB", "1.5G" (None stays None)."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?\s*", str(value).lower())
    if not m:
        raise InputValidationError(f"Cannot read the size {value!r}; use e.g. '500MB' or '2GB'.")
    return int(float(m.group(1)) * 1024 ** "_kmgt".index(m.group(2) or "_"))


def _entries(root: Path) -> list[tuple[Path, int, float]]:
    """Whole entries — a top-level file, or one item inside a top-level folder (one data set download) — with
    their size and last use (newest access or modification time of anything in them)."""
    out: list[tuple[Path, int, float]] = []
    if not root.is_dir():
        return out
    for top in root.iterdir():
        if top.name in _KEEP or top.name.startswith(".trash-"):
            continue
        units = list(top.iterdir()) if top.is_dir() and not top.is_symlink() else [top]
        for unit in units:
            size, last = 0, 0.0
            files = [unit] if not unit.is_dir() or unit.is_symlink() else [p for p in unit.rglob("*")]
            for f in files:
                if f.is_dir() and not f.is_symlink():
                    continue  # listing a folder updates its access time, so only files say when it was used
                try:
                    st = f.lstat()
                except OSError:
                    continue
                size += st.st_size
                last = max(last, st.st_atime, st.st_mtime)
            if last == 0.0:
                with contextlib.suppress(OSError):
                    last = unit.lstat().st_mtime
            out.append((unit, size, last))
    return out


@dataclass(frozen=True)
class CacheInfo:
    """What the cache holds."""

    path: str
    exists: bool
    size_bytes: int
    entries: tuple[dict[str, Any], ...]
    auto_clean: bool
    max_age_days: float
    max_size_bytes: int

    def summary(self) -> str:
        if not self.exists:
            return f"EvalSuite cache: {self.path} (empty; nothing downloaded yet)"
        lines = [
            f"EvalSuite cache: {self.path}",
            f"  {_human(self.size_bytes)} in {len(self.entries)} entr{'y' if len(self.entries) == 1 else 'ies'}",
            "  Automatic clearing: "
            + (
                f"on (unused for {self.max_age_days:g} days, or beyond {_human(self.max_size_bytes)})"
                if self.auto_clean
                else "off"
            ),
        ]
        for e in sorted(self.entries, key=lambda e: -e["size_bytes"])[:15]:
            lines.append(f"  {_human(e['size_bytes']):>9}  {e['days_unused']:6.1f} days unused  {e['name']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"  # pragma: no cover


def _settings() -> tuple[bool, float, int]:
    auto = os.environ.get("EVALSUITE_CACHE_AUTO_CLEAN", "1").strip().lower() not in ("0", "false", "no", "off")
    try:
        age = _parse_age(os.environ.get("EVALSUITE_CACHE_MAX_AGE")) or _DEFAULT_MAX_AGE
        size = _parse_size(os.environ.get("EVALSUITE_CACHE_MAX_SIZE")) or _DEFAULT_MAX_SIZE
    except InputValidationError:
        age, size = _DEFAULT_MAX_AGE, _DEFAULT_MAX_SIZE
    return auto, age, size


def cache_info() -> CacheInfo:
    """Location, size and entries of the cache, and the automatic-clearing settings."""
    root = cache_dir()
    auto, age, size = _settings()
    now = time.time()
    entries: tuple[dict[str, Any], ...] = tuple(
        {"name": p.relative_to(root).as_posix(), "size_bytes": s, "days_unused": max(0.0, (now - last) / 86400)}
        for p, s, last in _entries(root)
    )
    return CacheInfo(
        str(root), root.is_dir(), sum(int(e["size_bytes"]) for e in entries), entries, auto, age / 86400, size
    )


def _remove(path: Path, root: Path) -> None:
    """Delete one entry: rename it into the cache's trash first (atomic), so an interrupted deletion never leaves
    half an entry where a reader would find it; the trash is emptied now and on later runs."""
    import uuid

    trash = root / f".trash-{uuid.uuid4().hex}"
    try:
        path.rename(trash)
    except OSError:
        trash = path
    if trash.is_dir() and not trash.is_symlink():
        shutil.rmtree(trash, ignore_errors=True)
    else:
        with contextlib.suppress(OSError):
            trash.unlink()


def _owned(root: Path) -> bool:
    return (root / MARKER).is_file()


def clear_cache(
    *,
    older_than: Union[str, float, None] = None,
    max_size: Union[str, int, None] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete cache entries: everything, or those unused for ``older_than`` (e.g. ``"30d"``), then the least
    recently used until the cache is below ``max_size`` (e.g. ``"1GB"``). Returns what was (or with
    ``dry_run=True`` would be) removed. Only a folder EvalSuite created is touched."""
    root = cache_dir()
    if not root.is_dir():
        return {"path": str(root), "removed": [], "freed_bytes": 0, "dry_run": dry_run}
    if not _owned(root):
        raise InputValidationError(
            f"{root} was not created by EvalSuite (no {MARKER} file); refusing to delete anything in it."
        )
    age = _parse_age(older_than)
    limit = _parse_size(max_size)
    now = time.time()
    entries = sorted(_entries(root), key=lambda e: e[2])  # least recently used first
    doomed: list[tuple[Path, int]] = []
    if age is None and limit is None:
        doomed = [(p, s) for p, s, _ in entries]
    else:
        keep = []
        for p, s, last in entries:
            (doomed.append((p, s)) if age is not None and now - last > age else keep.append((p, s)))
        if limit is not None:
            total = sum(s for _, s in keep)
            for p, s in keep:
                if total <= limit:
                    break
                doomed.append((p, s))
                total -= s
    if not dry_run:
        for p, _ in doomed:
            _remove(p, root)
        for leftover in root.glob(".trash-*"):  # from an interrupted earlier run
            _remove(leftover, root)
        for top in root.iterdir():  # drop section folders left empty
            if top.is_dir() and top.name not in _KEEP and not top.name.startswith(".") and not any(top.iterdir()):
                top.rmdir()
    return {
        "path": str(root),
        "removed": [p.relative_to(root).as_posix() for p, _ in doomed],
        "freed_bytes": sum(s for _, s in doomed),
        "dry_run": dry_run,
    }


def auto_clean(*, force: bool = False) -> Optional[dict[str, Any]]:
    """The automatic clearing (see the module docstring): runs at most once a day unless ``force``; never
    raises."""
    try:
        auto, age, size = _settings()
        root = cache_dir()
        if not (auto or force) or not root.is_dir() or not _owned(root):
            return None
        stamp = root / _STAMP
        if not force and stamp.exists() and time.time() - stamp.stat().st_mtime < 86400:
            return None
        stamp.write_text(str(time.time()))
        return clear_cache(older_than=age, max_size=size)
    except Exception:
        return None
