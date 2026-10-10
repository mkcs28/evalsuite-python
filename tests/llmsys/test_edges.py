"""Input validation and rarely used branches of the v0.5.0 metrics."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import evalsuite as es
from evalsuite.llmsys import _common, code


@pytest.mark.parametrize(
    "call",
    [
        lambda: _common.seq(None, "x"),
        lambda: _common.seq("abc", "x"),
        lambda: _common.seq({"a": 1}, "x"),
        lambda: _common.seq([], "x"),
        lambda: _common.bools([2], "x"),
        lambda: _common.floats(["a"], "x"),
        lambda: _common.floats([[1.0, 2.0]], "x"),
        lambda: _common.floats([np.nan], "x"),
        lambda: _common.floats([-1.0], "x", lo=0),
        lambda: _common.floats([2.0], "x", hi=1),
        lambda: _common.by_group(np.ones(2), ["a"]),
        lambda: _common.lists_of(["ab"], "x"),
        lambda: _common.lists_of([3], "x"),
    ],
)
def test_common_validation(call) -> None:
    with pytest.raises(es.InputValidationError):
        call()


def test_common_helpers() -> None:
    assert _common.bools(np.array([1, 0]), "x").tolist() == [True, False]
    assert np.isnan(_common.wilson(0, 0)[0])
    assert _common.summary(np.array([2.0]))["std"] == 0.0


def test_code_edge_branches() -> None:
    src = "async def f(a):\n    async for x in a:\n        pass\n    return [y async for y in a]\n"
    assert es.code_complexity([src]).params["per_program"][0]["functions"] == {"f": 3}
    assert code._strip("x = '''unterminated") == "x = '''unterminated"
    assert code._bp(5, 0) == 0.0
    flow_src = (
        "import os\n"
        "def g(p, *, k):\n    v = p\n    with open(v) as fh:\n        w = fh.read()\n"
        "    try:\n        z = w\n    except OSError as err:\n        print(err)\n    finally:\n        q = z\n"
        "    while v:\n        v -= 1\n    else:\n        pass\n    return q, k\n"
    )
    import ast

    flows = code._dataflow(ast.parse(flow_src))
    assert ("err", "comesFrom", ("err",)) in flows and ("z", "comesFrom", ("z",)) in flows
    two_refs = es.codebleu([["x = 1", "def ("]], ["x = 1"])
    assert two_refs.params["syntax_match"] == 1.0


def test_unit_vector_and_luhn_branches() -> None:
    with pytest.raises(es.InputValidationError):
        es.weat_effect_size([[np.nan, 1.0]], [[1.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]])
    for bad in ([[np.nan, 1.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]):
        with pytest.raises(es.InputValidationError):
            es.bitext_mining_accuracy(bad, [[1.0, 0.0], [0.0, 1.0]])
    assert es.detect_pii("pay with 5555 5555 5555 4444 today")["credit_card"] == ["5555 5555 5555 4444"]
    with pytest.raises(es.InputValidationError):
        es.compression_ratio([1], ["a"])


def test_spice_wordnet_path_with_a_stub(monkeypatch) -> None:
    class Syn:
        def __init__(self, name: str) -> None:
            self._n = name

        def name(self) -> str:
            return self._n

    groups = {"car": ["car.n.01"], "automobile": ["car.n.01"], "dog": ["dog.n.01"]}
    wordnet = types.SimpleNamespace(synsets=lambda w: [Syn(n) for n in groups.get(w, [])])
    corpus = types.ModuleType("nltk.corpus")
    corpus.wordnet = wordnet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nltk.corpus", corpus)
    assert float(es.spice([[("automobile",)]], [[("car",)]], synonyms="wordnet")) == 1.0
    assert float(es.spice([[("dog",)]], [[("car",)]], synonyms="wordnet")) == 0.0
