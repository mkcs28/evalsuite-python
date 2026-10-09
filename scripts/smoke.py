"""Smoke test for an installed evalsuite-python (run it outside the repository, in a clean environment).

    python scripts/smoke.py [expected_version]

Checks the documented import, that importing does not pull in optional heavy dependencies, and a few
known answers from every module. Prints a fingerprint so wheel and sdist installs can be compared.
"""

from __future__ import annotations

import hashlib
import json
import sys

import evalsuite as es

expected = sys.argv[1] if len(sys.argv) > 1 else None
if expected and es.__version__ != expected:
    sys.exit(f"installed version {es.__version__} != expected {expected}")
if "evalsuite" in __file__ and "site-packages" not in es.__file__ and "dist-packages" not in es.__file__:
    print(f"warning: evalsuite imported from {es.__file__}, not from an installed package")

heavy = [m for m in ("matplotlib", "sklearn", "PIL", "pycocotools", "statsmodels") if m in sys.modules]
if heavy:
    sys.exit(f"import evalsuite pulled in optional dependencies: {heavy}")

checks = {
    "accuracy": float(es.evaluate([0, 1, 1, 0], [0, 1, 0, 0])["accuracy"]),
    "rmse": float(es.rmse([1.0, 3.0], [0.0, 0.0])),
    "sensitivity": float(es.sensitivity([1, 1, 1, 0], [1, 1, 0, 0])),
    "holm": [float(v) for v in es.adjust_pvalues([0.01, 0.04], method="holm")],
    "dice": float(es.dice([[1, 1], [0, 0]], [[1, 0], [0, 0]], average=None).value[1]),
    "box_iou": float(es.box_iou([[0, 0, 10, 10]], [[5, 5, 15, 15]])[0, 0]),
    "map": float(
        es.mean_average_precision(
            [{"boxes": [[0, 0, 10, 10]], "labels": [1]}],
            [{"boxes": [[0, 0, 10, 10]], "labels": [1], "scores": [0.9]}],
        )
    ),
    "n_metrics": len(es.list_metrics()),
}
expected_values = {
    "accuracy": 0.75,
    "rmse": 5**0.5,
    "sensitivity": 2 / 3,
    "holm": [0.02, 0.04],
    "dice": 2 / 3,
    "box_iou": 25 / 175,
    "map": 1.0,
}
for key, want in expected_values.items():
    got = checks[key]
    ok = all(abs(a - b) < 1e-12 for a, b in zip(got, want)) if isinstance(want, list) else abs(got - want) < 1e-12
    if not ok:
        sys.exit(f"{key}: got {got}, expected {want}")

fingerprint = hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest()[:16]
print(f"evalsuite {es.__version__} OK: {checks['n_metrics']} metrics, fingerprint {fingerprint}")
