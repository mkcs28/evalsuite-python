from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def assert_close(actual: object, expected: object, tol: float = 1e-10) -> None:
    np.testing.assert_allclose(
        np.asarray(getattr(actual, "value", actual), dtype=float),
        np.asarray(expected, dtype=float),
        rtol=tol,
        atol=tol,
    )
