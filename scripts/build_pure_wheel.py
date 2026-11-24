"""Build the pure-Python wheel (py3-none-any): the same package without the compiled core.

    python scripts/build_pure_wheel.py dist/

pip installs it only where no binary wheel fits the platform; EvalSuite then runs its Python implementations,
which give the same results (tests/test_native.py). Uses hatchling's build hook directly, with the
[tool.hatch.*] configuration in pyproject.toml (the project's build backend is maturin).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(out: str) -> int:
    from hatchling.build import build_wheel

    Path(out).mkdir(parents=True, exist_ok=True)
    name = build_wheel(out)
    if not name.endswith("-py3-none-any.whl"):
        print(f"expected a pure wheel, got {name}")
        return 1
    print(f"built {out.rstrip('/')}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "dist"))
