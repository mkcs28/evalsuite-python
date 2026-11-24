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


CORE = re.compile(r"^evalsuite/_core(\.[^/]*)?\.(so|pyd)$")


def check_wheel(wheel: Path, version: str, sdist_pkg: set[str], problems: list[str]) -> tuple[int, dict[str, str]]:
    """Check one wheel; returns its number of package files and its METADATA."""
    pure = wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        meta = Parser().parsestr(zf.read(meta_name).decode("utf-8"))
        cores = [n for n in names if CORE.match(n)]
        wheel_files = {n for n in names if n.startswith("evalsuite/") and not CORE.match(n)}
        blobs = {n: zf.read(n) for n in names if not CORE.match(n)}
    tag = wheel.name
    if f"-{version}-" not in tag:
        problems.append(f"{tag} does not carry version {version} from src/evalsuite/version.py")
    if pure and cores:
        problems.append(f"{tag}: the pure-Python wheel must not contain the compiled core ({cores})")
    if not pure and len(cores) != 1:
        problems.append(f"{tag}: a binary wheel must contain exactly one compiled core, found {cores}")
    if not pure and "-cp39-abi3-" not in tag:
        problems.append(f"{tag}: binary wheels must target the stable ABI for Python >= 3.9 (cp39-abi3)")
    for key in REQUIRED_META:
        if not meta.get(key):
            problems.append(f"{tag}: METADATA is missing {key}")
    if meta.get("Version") != version:
        problems.append(f"{tag}: METADATA version {meta.get('Version')} != {version}")
    if meta.get("License-Expression") != "MIT OR Apache-2.0":
        problems.append(f"{tag}: License-Expression is {meta.get('License-Expression')!r}")
    requires = meta.get_all("Requires-Dist") or []
    for dep in ("numpy", "scipy", "pandas"):
        if not any(r.startswith(dep) and "extra ==" not in r for r in requires):
            problems.append(f"{tag}: runtime dependency {dep} missing from Requires-Dist")
    if any(
        r.split(";")[0].strip().startswith(("matplotlib", "scikit-learn", "pytest")) and "extra ==" not in r
        for r in requires
    ):
        problems.append(f"{tag}: an optional or development tool is a required runtime dependency")
    for name in names:
        if FORBIDDEN.search(name):
            problems.append(f"{tag}: file must not be shipped: {name}")
    for name, data in blobs.items():
        if SECRET.search(data):
            problems.append(f"{tag}: possible credential in {name}")
    if wheel_files != sdist_pkg:
        problems.append(f"{tag}: wheel and sdist package files differ: {sorted(wheel_files ^ sdist_pkg)[:5]}")
    return len(wheel_files), dict(meta.items())


def main(dist: Path) -> int:
    """Check every wheel and the sdist in ``dist``: one pure-Python wheel (the fallback), any number of
    binary (abi3) wheels with the compiled core, and exactly one sdist, all of the same version."""
    problems: list[str] = []
    version = version_of_source()
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    pure = [w for w in wheels if w.name.endswith("-py3-none-any.whl")]
    if len(sdists) != 1 or len(pure) != 1:
        print(f"Expected one sdist and one pure-Python wheel in {dist}, found {len(sdists)} and {len(pure)}.")
        return 1
    sdist = sdists[0]
    if f"-{version}.tar.gz" not in sdist.name:
        problems.append(f"{sdist.name} does not carry version {version} from src/evalsuite/version.py")
    with tarfile.open(sdist) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        sdist_names = [m.name.split("/", 1)[1] for m in members]
        sdist_pkg = {n[len("src/") :] for n in sdist_names if n.startswith("src/evalsuite/")}
        sdist_blobs = {m.name: tf.extractfile(m).read() for m in members}
    for need in (
        "rust/Cargo.toml",
        "rust/evalsuite-core/src/lib.rs",
        "rust/evalsuite-py/src/lib.rs",
        "LICENSE-APACHE",
    ):
        if need not in sdist_names:
            problems.append(f"sdist is missing {need} (needed to build from source)")
    for name in sdist_names:
        if FORBIDDEN.search(name) or name.startswith("rust/target/"):
            problems.append(f"sdist: file must not be shipped: {name}")
    for name, data in sdist_blobs.items():
        if SECRET.search(data):
            problems.append(f"sdist: possible credential in {name}")
    meta: dict[str, str] = {}
    count = 0
    for wheel in wheels:
        count, meta = check_wheel(wheel, version, sdist_pkg, problems)
    if problems:
        print("Distribution check failed:\n  " + "\n  ".join(problems))
        return 1
    print(
        f"Distribution check passed: {len(wheels)} wheels ({len(wheels) - 1} binary, 1 pure-Python; "
        f"{count} package files each), {sdist.name}; Requires-Python {meta['Requires-Python']}, "
        f"license {meta['License-Expression']}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist")))
