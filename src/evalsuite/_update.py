"""Staying current: a once-a-day notice when a newer EvalSuite is on PyPI, and a one-command upgrade.

pip itself never replaces an installed package unless asked (``pip install -U``), and no package can change
that. EvalSuite therefore tells you when a newer release exists and upgrades on request:

* On import, if the last check (cached for a day in the user cache directory) found a newer release, one line
  is printed to stderr. The check itself runs in a background thread, so importing never waits on the network.
* ``evalsuite.upgrade()`` or ``evalsuite upgrade`` installs the latest release into the running interpreter
  (``python -m pip install --upgrade evalsuite-python``).
* ``evalsuite.check_for_update()`` returns the installed and latest versions.

Set ``EVALSUITE_NO_UPDATE_CHECK=1`` to turn the check off; it is also off in CI (``CI`` set) and under pytest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PACKAGE = "evalsuite-python"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE}/json"
DAY = 24 * 3600


@dataclass(frozen=True)
class UpdateInfo:
    """Installed and latest released versions."""

    installed: str
    latest: Optional[str]

    @property
    def update_available(self) -> bool:
        return self.latest is not None and _key(self.latest) > _key(self.installed)

    def __repr__(self) -> str:
        if self.latest is None:
            return f"UpdateInfo(installed={self.installed}, latest=unknown)"
        state = "update available" if self.update_available else "up to date"
        return f"UpdateInfo(installed={self.installed}, latest={self.latest}, {state})"


def _key(version: str) -> tuple[int, ...]:
    """Release tuple of a version string ('1.1.2' -> (1, 1, 2)); pre-release suffixes sort below the release."""
    parts: list[int] = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
        if len(digits) < len(piece):  # e.g. "2rc1": a pre-release of 2
            parts[-1] -= 1
            break
    return tuple(parts)


def _installed() -> str:
    from .version import __version__

    return __version__


def _cache_file() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "evalsuite")
        if sys.platform == "win32" and os.environ.get("LOCALAPPDATA")
        else os.path.join(os.path.expanduser("~"), ".cache")
    )
    return Path(base) / "evalsuite" / "update-check.json"


def _read_cache() -> dict[str, Any]:
    try:
        data = json.loads(_cache_file().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(data: dict[str, Any]) -> None:
    try:
        path = _cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except OSError:
        pass


def _fetch_latest(timeout: float = 3.0) -> Optional[str]:
    """Latest version on PyPI, or None when PyPI cannot be reached."""
    from urllib.request import Request, urlopen

    try:
        req = Request(PYPI_URL, headers={"Accept": "application/json", "User-Agent": f"evalsuite/{_installed()}"})
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            version = json.load(resp)["info"]["version"]
        return str(version)
    except Exception:  # offline, proxy, firewall, malformed response: never an error for the user
        return None


def check_for_update(*, timeout: float = 3.0) -> UpdateInfo:
    """Ask PyPI for the latest release now (refreshes the daily cache)."""
    latest = _fetch_latest(timeout)
    if latest is not None:
        data = _read_cache()
        data.update({"checked": time.time(), "latest": latest})
        _write_cache(data)
    return UpdateInfo(_installed(), latest)


def upgrade(*, quiet: bool = False) -> int:
    """Install the latest release into this Python environment (``python -m pip install --upgrade``).

    Returns pip's exit code. In a virtual environment, conda environment or user install, the upgrade goes
    exactly where the running ``evalsuite`` came from, because it uses this interpreter's pip."""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE]
    if quiet:
        cmd.append("--quiet")
    return subprocess.call(cmd)  # noqa: S603 - fixed arguments


def _disabled() -> bool:
    env = os.environ
    return bool(
        env.get("EVALSUITE_NO_UPDATE_CHECK")
        or env.get("CI")
        or env.get("PYTEST_CURRENT_TEST")
        or "pytest" in sys.modules
    )


def _background_refresh() -> None:
    latest = _fetch_latest()
    if latest is not None:
        data = _read_cache()
        data.update({"checked": time.time(), "latest": latest})
        _write_cache(data)


def notify_on_import() -> None:
    """Print the cached notice (at most once a day) and refresh the cache in the background when stale."""
    if _disabled():
        return
    data = _read_cache()
    now = time.time()
    latest = data.get("latest")
    installed = _installed()
    if isinstance(latest, str) and _key(latest) > _key(installed) and now - float(data.get("notified", 0)) > DAY:
        print(
            f"EvalSuite {latest} is available (you have {installed}). Upgrade with: evalsuite upgrade "
            f"(or pip install -U {PACKAGE}). Silence with EVALSUITE_NO_UPDATE_CHECK=1.",
            file=sys.stderr,
        )
        data["notified"] = now
        _write_cache(data)
    if now - float(data.get("checked", 0)) > DAY:
        threading.Thread(target=_background_refresh, name="evalsuite-update-check", daemon=True).start()


__all__ = ["UpdateInfo", "check_for_update", "upgrade"]
