"""Check the built wheel and sdist before they are published.

    python scripts/check_dist.py dist/

Fails (exit 1) with a message for each problem: wrong version, missing metadata, files that must not be
shipped (caches, secrets, local data, VCS files), or a wheel and sdist that disagree about the package.
"""

from __future__ import annotations

import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

FORBIDDEN = re.compile(
    r"(__pycache__|\.pyc$|\.pyo$|\.env$|\.env\.|\.pem$|\.key$|\.p12$|id_rsa|\.pypirc$|\.netrc$|"
    r"credentials|secrets?\.|\.ipynb_checkpoints|\.DS_Store|\.git/|\.npy$|\.npz$|\.csv$|\.parquet$|\.pkl$|"
    r"\.pickle$|\.coverage|htmlcov/|\.whl$|\.tar\.gz$)",
    re.IGNORECASE,
)
SECRET = re.compile(
    rb"(github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|pypi-AgEI[A-Za-z0-9_-]{20,}|"
    rb"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
REQUIRED_META = ["Name", "Version", "Summary", "Requires-Python", "License-Expression", "Author"]


def version_of_source() -> str:
    text = Path("src/evalsuite/version.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "(.+)"', text).group(1)


def main(dist: Path) -> int:
    problems: list[str] = []
    version = version_of_source()
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print(f"Expected one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}.")
        return 1
    wheel, sdist = wheels[0], sdists[0]
    for path in (wheel, sdist):
        if f"-{version}" not in path.name:
            problems.append(f"{path.name} does not carry version {version} from src/evalsuite/version.py")

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        meta = Parser().parsestr(zf.read(meta_name).decode("utf-8"))
        wheel_files = {n for n in names if n.startswith("evalsuite/")}
        blobs = {n: zf.read(n) for n in names}
    with tarfile.open(sdist) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        sdist_names = [m.name.split("/", 1)[1] for m in members]
        sdist_pkg = {n[len("src/") :] for n in sdist_names if n.startswith("src/evalsuite/")}
        blobs.update({m.name: tf.extractfile(m).read() for m in members})

    for key in REQUIRED_META:
        if not meta.get(key):
            problems.append(f"wheel METADATA is missing {key}")
    if meta.get("Version") != version:
        problems.append(f"wheel METADATA version {meta.get('Version')} != {version}")
    requires = meta.get_all("Requires-Dist") or []
    for dep in ("numpy", "scipy", "pandas"):
        if not any(r.startswith(dep) and "extra ==" not in r for r in requires):
            problems.append(f"runtime dependency {dep} missing from Requires-Dist")
    if any(
        r.split(";")[0].strip().startswith(("matplotlib", "scikit-learn", "pytest")) and "extra ==" not in r
        for r in requires
    ):
        problems.append("an optional or development tool is a required runtime dependency")

    for name in [*names, *sdist_names]:
        if FORBIDDEN.search(name):
            problems.append(f"file must not be shipped: {name}")
    for name, data in blobs.items():
        if SECRET.search(data):
            problems.append(f"possible credential in {name}")
    if wheel_files != sdist_pkg:
        problems.append(f"wheel and sdist package files differ: {sorted(wheel_files ^ sdist_pkg)[:5]}")

    if problems:
        print("Distribution check failed:\n  " + "\n  ".join(problems))
        return 1
    print(
        f"Distribution check passed: {wheel.name} ({len(wheel_files)} package files), {sdist.name}; "
        f"Requires-Python {meta['Requires-Python']}, license {meta['License-Expression']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist")))
