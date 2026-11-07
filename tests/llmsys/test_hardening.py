"""Regression tests for adversarial inputs (parser exhaustion, regex backtracking) and lazy imports."""

import subprocess
import sys
import time

import pytest

import evalsuite as es
from evalsuite.core.exceptions import InputValidationError

DEEP = "x=" + "-" * 100_000 + "1"


def test_syntax_validity_survives_parser_stack_overflow():
    assert float(es.syntax_validity_rate([DEEP, "x = 1"])) == 0.5


@pytest.mark.parametrize("src", [DEEP, "x=" + "-" * 900 + "1"])
def test_code_complexity_rejects_too_deep_code_cleanly(src):
    with pytest.raises(InputValidationError, match=r"too deeply nested|not valid Python"):
        es.code_complexity([src])


def test_codebleu_deep_input_does_not_crash():
    r = es.codebleu([DEEP], [DEEP])
    assert 0.0 <= float(r) <= 1.0


def test_codebleu_large_program_is_fast():
    src = "\n".join(f"def f{i}(a, b):\n    return [x * b for x in range(a + {i})]\n" for i in range(2000))
    start = time.perf_counter()
    es.codebleu([src], [src])
    assert time.perf_counter() - start < 5


@pytest.mark.parametrize("text", ["1-" * 30_000, "a" * 30_000 + "@" + "b." * 10_000, "a." * 30_000, "a@" * 30_000])
def test_detect_pii_is_linear_on_adversarial_text(text):
    start = time.perf_counter()
    es.detect_pii(text)
    assert time.perf_counter() - start < 1


def test_detect_pii_still_finds_emails():
    found = es.detect_pii("write to a.b@x.org or c@d.co.uk")
    assert found["email"] == ["a.b@x.org", "c@d.co.uk"]


def test_import_does_not_load_scipy():
    code = "import sys, evalsuite; print('scipy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip() == "False"
