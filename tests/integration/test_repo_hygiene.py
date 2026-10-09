"""Security and maintenance checks on the repository itself (run in CI on every push)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "evalsuite"

SECRET = re.compile(
    r"(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|pypi-AgEI[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{10,}|sk-[A-Za-z0-9]{32,})"
)


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [ROOT / line for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):  # sdist without .git: scan what is there
        return [p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts]


def test_no_credentials_in_repository() -> None:
    leaks = []
    for path in _tracked_files():
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SECRET.search(text):
            leaks.append(str(path.relative_to(ROOT)))
    assert not leaks, f"possible credentials committed: {leaks}"


@pytest.mark.parametrize(
    "pattern",
    [
        r"\bpickle\b",
        r"(?<![\w.])eval\(",
        r"(?<![\w.])exec\(",
        r"\bsubprocess\b",
        r"shell\s*=\s*True",
        r"yaml\.load\(",
        r"allow_pickle\s*=\s*True",
        r"__import__\(",
    ],
)
def test_no_unsafe_calls_in_package(pattern: str) -> None:
    hits = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in SRC.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(pattern, line.split("#", 1)[0])
    ]
    assert not hits, f"unsafe call {pattern!r} in {hits}"


def test_numpy_loads_never_allow_pickle() -> None:
    for p in SRC.rglob("*.py"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if "np.load(" in line:
                assert "allow_pickle=False" in line, f"{p.name}: {line.strip()}"


def test_workflows_use_least_privilege() -> None:
    workflows = list((ROOT / ".github" / "workflows").glob("*.yml"))
    if not workflows:
        pytest.skip("workflows are not shipped in the sdist")
    for wf in workflows:
        text = wf.read_text(encoding="utf-8")
        assert re.search(r"^permissions:\s*\n\s+contents: read", text, re.M), (
            f"{wf.name}: default must be read-only"
        )
        assert "pull_request_target" not in text, (
            f"{wf.name}: pull_request_target runs untrusted code with secrets"
        )
        assert "password:" not in text and "PYPI_API_TOKEN" not in text, f"{wf.name}: use Trusted Publishing"


def test_single_version_source() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject and 'path = "src/evalsuite/version.py"' in pyproject
    import evalsuite

    version = re.search(r'__version__ = "(.+)"', (SRC / "version.py").read_text()).group(1)
    assert evalsuite.__version__ == version
    assert f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
