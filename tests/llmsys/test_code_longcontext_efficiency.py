"""v0.5.0 code, long-context, efficiency and SPICE metrics, with radon and the codebleu package as references."""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pytest

import evalsuite as es

PKG = os.path.dirname(es.__file__)
SOURCES = sorted(glob.glob(os.path.join(PKG, "**", "*.py"), recursive=True))

PROGRAMS = [
    "def add(a, b):\n    return a + b\n",
    "def f(xs):\n    total = 0\n    for x in xs:\n        if x > 0 and x < 10:\n            total += x\n        "
    "elif x == 0:\n            continue\n    return total\n",
    "class A:\n    '''doc'''\n    def m(self, y):\n        try:\n            return [i for i in y if i]\n        "
    "except ValueError:\n            return None\n        finally:\n            pass\n\n\ndef g(z):\n    # "
    "comment\n    assert z\n    return z if z else -z  # tail\n",
    "x = 1; y = 2\nif x: y = 3\nwhile y:\n    y -= 1\nelse:\n    pass\n",
]
if sys.version_info >= (3, 10):  # match statements
    PROGRAMS.append(
        'def h(v):\n    match v:\n        case 1:\n            return "one"\n        case [a, b]:\n            return a\n        case _:\n            return None\n'
    )


def test_complexity_and_mi_match_radon() -> None:
    cc = pytest.importorskip("radon.complexity")
    rm = pytest.importorskip("radon.metrics")
    texts = []
    for f in SOURCES[:40]:
        with open(f, encoding="utf-8") as fh:
            texts.append(fh.read())
    for src in PROGRAMS + texts:
        r = es.code_complexity([src])
        theirs = {}
        for b in cc.cc_visit(src):
            if b.letter == "F":
                theirs[b.name] = b.complexity
            elif b.letter == "C":
                for m in b.methods:
                    theirs[f"{b.name}.{m.name}"] = m.complexity
        ours = r.params["per_program"][0]["functions"]
        assert {k: v for k, v in ours.items() if k in theirs} == theirs
        assert r.params["maintainability_index"] == pytest.approx(rm.mi_visit(src, True), abs=1e-9)


def test_complexity_hand_values() -> None:
    r = es.code_complexity(PROGRAMS[:2])
    assert r.params["per_program"][1]["functions"] == {"f": 5}  # for, if, and, elif + 1
    assert float(r) == pytest.approx(3.0) and r.params["max_complexity"] == 5
    assert float(es.code_complexity(["x = 1\n"])) == 1.0
    with pytest.raises(es.InputValidationError):
        es.code_complexity(["def ("])
    with pytest.raises(es.InputValidationError):
        es.code_complexity([1])


CB_REFS = [
    "def add(a, b):\n    return a + b\n",
    "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    ["def mx(a, b):\n    if a > b:\n        return a\n    return b\n", "def mx(a, b):\n    return max(a, b)\n"],
]
CB_PREDS = [
    "def add(x, y):\n    return x + y\n",
    "def total(items):\n    s = 0\n    for i in items:\n        s = s + i\n    return s\n",
    "def mx(a, b):\n    return a if a > b else b\n",
]


def test_codebleu_ngram_components_match_reference() -> None:
    cb = pytest.importorskip("codebleu")
    try:
        ref = cb.calc_codebleu([r if isinstance(r, list) else [r] for r in CB_REFS], CB_PREDS, lang="python")
    except Exception as exc:  # pragma: no cover - tree-sitter version mismatch in the reference package
        pytest.skip(f"codebleu reference unavailable: {exc}")
    ours = es.codebleu(CB_REFS, CB_PREDS)
    assert ours.params["ngram_match"] == pytest.approx(ref["ngram_match_score"], abs=1e-12)
    assert ours.params["weighted_ngram_match"] == pytest.approx(ref["weighted_ngram_match_score"], abs=1e-12)
    assert 0 <= ours.params["syntax_match"] <= 1 and 0 <= ours.params["dataflow_match"] <= 1


def test_codebleu_properties() -> None:
    single = [r if isinstance(r, str) else r[0] for r in CB_REFS]
    same = es.codebleu(single, single)
    assert float(same) == pytest.approx(1.0)
    renamed = es.codebleu(["def f(a):\n    b = a\n    return b\n"], ["def g(x):\n    y = x\n    return y\n"])
    assert renamed.params["syntax_match"] == 1.0 and renamed.params["dataflow_match"] == 1.0
    assert renamed.params["ngram_match"] < 0.5
    junk = es.codebleu(["def f(a):\n    return a\n"], ["zzz qqq"])
    assert junk.params["ngram_match"] == 0.0 and junk.params["syntax_match"] < 1
    noflow = es.codebleu(["pass"], ["pass"])
    q = noflow.params
    assert q["dataflow_match"] == 0.0  # no data flow anywhere: the term counts as 1, as in the reference
    assert float(noflow) == pytest.approx(
        0.25 * (q["ngram_match"] + q["weighted_ngram_match"] + q["syntax_match"] + 1)
    )
    w = es.codebleu(CB_REFS, CB_PREDS, weights=(1, 0, 0, 0))
    assert float(w) == pytest.approx(w.params["ngram_match"])
    assert es.codebleu(["x"], ["def (:"]).params["syntax_match"] == 0.0
    for bad in (
        lambda: es.codebleu(CB_REFS, CB_PREDS, weights=(1, 1, 0, 0)),
        lambda: es.codebleu(CB_REFS, CB_PREDS[:2]),
        lambda: es.codebleu([[]], ["x"]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    flows = es.llmsys.code._normalize_flow(
        es.llmsys.code._dataflow(
            __import__("ast").parse(
                "a = 1\nb = a + c\nfor i in b:\n    a += i\ntry:\n    q = a\nexcept E as e:\n    print(e)\n"
            )
        )
    )
    assert ("var_0", "computedFrom", ()) in flows and len(flows) >= 6


def test_code_test_metrics() -> None:
    u = es.unit_test_pass_rate([[True, True], [True, False, False], [False]])
    assert float(u) == pytest.approx((1 + 1 / 3 + 0) / 3) and u.params["strict_pass_rate"] == pytest.approx(1 / 3)
    with pytest.raises(es.InputValidationError):
        es.unit_test_pass_rate([[]])
    s = es.syntax_validity_rate(["x = 1", "def (:", "print('\\d')"])
    assert float(s) == pytest.approx(2 / 3)
    assert float(es.syntax_validity_rate(["int main(){}"], language="c", checker=lambda c: c.endswith("}"))) == 1.0
    with pytest.raises(es.InputValidationError):
        es.syntax_validity_rate(["x"], language="c")
    v = es.static_analysis_violation_rate([2, 0, 3], [100, 50, 50])
    assert float(v) == 25.0 and v.params["programs_with_violations"] == pytest.approx(2 / 3)
    with pytest.raises(es.InputValidationError):
        es.static_analysis_violation_rate([1], [0])
    x = es.execution_success_rate(["passed", "timeout", "wrong_answer", "passed"])
    assert float(x) == 0.5 and x.params["by_failure"] == {"timeout": 0.25, "wrong_answer": 0.25}
    cov = es.coverage_rate([50, 10, 0], [100, 10, 0])
    assert float(cov) == pytest.approx(60 / 110) and cov.params["mean_per_program"] == pytest.approx(
        (0.5 + 1 + 1) / 3
    )
    for bad in (lambda: es.coverage_rate([2], [1]), lambda: es.coverage_rate([0], [0])):
        with pytest.raises(es.InputValidationError):
            bad()
    assert float(es.patch_acceptance_rate([True, False])) == 0.5
    rep = es.patch_acceptance_rate(fails_before=[True, True, False], passes_after=[True, False, True])
    assert float(rep) == pytest.approx(1 / 3) and rep.params["mode"] == "reproduction"
    with pytest.raises(es.InputValidationError):
        es.patch_acceptance_rate()


def test_swe_bench_security_runtime() -> None:
    r = es.resolved_rate([[True, True], [True], [False]], [[True], [False], []], applied=[True, True, True])
    assert float(r) == pytest.approx(1 / 3) and r.params["f2p_only_rate"] == pytest.approx(2 / 3)
    assert float(es.resolved_rate([[True]], [[]], applied=[False])) == 0.0
    with pytest.raises(es.InputValidationError):
        es.resolved_rate([[]], [[]])
    s = es.security_vulnerability_rate([["low"], [], ["high", "medium"]], min_severity="medium")
    assert float(s) == pytest.approx(1 / 3) and s.params["findings_per_program"]["low"] == pytest.approx(1 / 3)
    for bad in (
        lambda: es.security_vulnerability_rate([["x"]]),
        lambda: es.security_vulnerability_rate([[]], min_severity="x"),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    rt = es.runtime_efficiency(
        [2.0, 1.0, 6.0], [1.0, 1.0, 2.0], memory=[10, 20, 30], reference_memory=[10, 10, 10]
    )
    assert (
        float(rt) == pytest.approx(2.0)
        and rt.params["nmu"] == 2.0
        and rt.params["share_slower_than_2x"] == pytest.approx(1 / 3)
    )
    for bad in (
        lambda: es.runtime_efficiency([1], [0]),
        lambda: es.runtime_efficiency([1], [1], memory=[1], reference_memory=[0]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_long_context_metrics() -> None:
    r = es.retrieval_accuracy_by_length([1, 1, 1, 0, 1, 0], [4000, 4000, 8000, 8000, 16000, 16000], threshold=0.75)
    assert float(r) == pytest.approx(4 / 6) and r.params["effective_length"] == 4000
    n = es.needle_in_haystack([1, 0, 1, 1], [1000, 1000, 2000, 2000], [0.0, 0.5, 0.0, 0.5])
    assert n.params["grid"] == [[1.0, 0.0], [1.0, 1.0]] and n.params["worst_cell"] == 0.0
    p = es.position_accuracy([1, 1, 0, 0, 1], [0.0, 0.1, 0.5, 0.55, 1.0], n_bins=2)
    assert p.params["by_position"] == {"0-0.5": 1.0, "0.5-1": pytest.approx(1 / 3)} and float(p) == pytest.approx(
        2 / 3
    )
    with pytest.raises(es.InputValidationError):
        es.position_accuracy([1], [0.5], n_bins=1)
    lm = es.lost_in_the_middle([1, 1, 0, 0, 1, 1], [0.0, 0.1, 0.4, 0.6, 0.9, 1.0])
    assert float(lm) == 1.0
    for bad in (lambda: es.lost_in_the_middle([1], [0.5]), lambda: es.lost_in_the_middle([1], [0.5], edge=0.6)):
        with pytest.raises(es.InputValidationError):
            bad()
    u = es.context_utilization([["a", "b"], ["c"], []], [["a", "b", "x"], ["c"], []])
    assert float(u) == 0.75 and u.params["mean_per_example"] == pytest.approx((2 / 3 + 1) / 2)
    with pytest.raises(es.InputValidationError):
        es.context_utilization([[]], [[]])
    sc = es.summary_coverage(
        [[True, False, True], [False]], weights=[[3, 2, 1], [1]], claims_supported=[[True, True], [False]]
    )
    assert float(sc) == pytest.approx((4 / 6 + 0) / 2)
    assert sc.params["factual_consistency"] == pytest.approx(2 / 3) and sc.params["fully_consistent_rate"] == 0.5
    with pytest.raises(es.InputValidationError):
        es.summary_coverage([[True]], weights=[[1, 2]])
    cr = es.compression_ratio(["a b c d", "a b c d e f"], ["a b", "a b"])
    assert float(cr) == 2.5 and cr.params["median"] == 2.5
    with pytest.raises(es.InputValidationError):
        es.compression_ratio(["a"], [""])
    cc = es.citation_coverage([[1, 0, 2], [0]])
    assert float(cc) == 0.5 and cc.params["mean_per_output"] == pytest.approx((2 / 3 + 0) / 2)
    with pytest.raises(es.InputValidationError):
        es.citation_coverage([[]])
    xd = es.cross_document_consistency([["A", "a"], {"o1": "x", "o2": "y"}])
    assert float(xd) == 0.5
    with pytest.raises(es.InputValidationError):
        es.cross_document_consistency([["only"]])


def test_efficiency_metrics() -> None:
    start = [0.0, 1.0, 2.0]
    first = [0.2, 1.5, 2.1]
    end = [1.2, 3.5, 2.1]
    out = [11, 21, 1]
    t = es.time_to_first_token(start, first)
    assert float(t) == pytest.approx(np.mean([0.2, 0.5, 0.1])) and t.params["p50"] == pytest.approx(0.2)
    tp = es.time_per_output_token(first, end, out)
    assert float(tp) == pytest.approx(np.mean([1.0 / 10, 2.0 / 20])) and tp.params["n_used"] == 2
    with pytest.raises(es.InputValidationError):
        es.time_per_output_token([0], [1], [1])
    with pytest.raises(es.InputValidationError):
        es.time_to_first_token([1.0], [0.5])
    lat = es.latency_percentiles([1, 2, 3, 4, 100])
    assert float(lat) == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95)) and lat.params["p50"] == 3
    with pytest.raises(es.InputValidationError):
        es.latency_percentiles([1], percentile=101)
    th = es.throughput(out, start, end)
    assert float(th) == pytest.approx(33 / 3.5)
    with pytest.raises(es.InputValidationError):
        es.throughput([1], [1.0], [1.0])
    tu = es.token_usage([100, 300], [10, 30])
    assert float(tu) == 220 and tu.params["total_output"] == 40
    c = es.inference_cost(
        [1000, 3000], [100, 300], input_price=3.0, output_price=15.0, cached_tokens=[0, 1000], cached_price=0.3
    )
    expected = np.array([1000 * 3 + 100 * 15, 3000 * 3 + 300 * 15 - 1000 * 2.7]) / 1e6
    assert float(c) == pytest.approx(expected.mean()) and c.params["cost_per_1k_tokens"] == pytest.approx(
        expected.sum() / 4400 * 1000
    )
    for bad in (
        lambda: es.inference_cost([1], [1], input_price=-1, output_price=1),
        lambda: es.inference_cost([1], [1], input_price=1, output_price=1, cached_tokens=[1]),
        lambda: es.inference_cost([1], [1], input_price=1, output_price=1, cached_tokens=[2], cached_price=0.1),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    ru = es.resource_utilization({"gpu_util": [50, 90, 70], "memory_gb": [10, 12, 11]})
    assert (
        float(ru) == 12 and ru.params["primary"] == "memory_gb" and ru.params["series"]["gpu_util"]["mean"] == 70
    )
    with pytest.raises(es.InputValidationError):
        es.resource_utilization({})
    e = es.energy_per_request([100, 200, 300], interval_s=1.0, n_requests=2)
    assert float(e) == pytest.approx((150 + 250) / 3600 / 2)
    assert float(es.energy_per_request(total_joules=7200, n_requests=1)) == 2.0
    assert float(es.energy_per_request([360], interval_s=10, n_requests=1)) == pytest.approx(1.0)
    for bad in (
        lambda: es.energy_per_request(n_requests=0),
        lambda: es.energy_per_request(n_requests=1),
        lambda: es.energy_per_request(total_joules=-1, n_requests=1),
    ):
        with pytest.raises(es.InputValidationError):
            bad()
    assert float(es.requests_per_second([0.0, 1.0, 2.0, 4.0])) == 0.75
    assert float(es.requests_per_second([1.0, 2.0], start_time=0.0)) == 1.0
    for bad in (lambda: es.requests_per_second([1.0]), lambda: es.requests_per_second([1.0, 1.0])):
        with pytest.raises(es.InputValidationError):
            bad()
    a = es.availability([True, 200, 503, "ok", False], slo=0.9)
    assert (
        float(a) == 0.6
        and a.params["error_rate"] == pytest.approx(0.4)
        and a.params["error_budget_remaining"] == pytest.approx(-3.0)
    )
    with pytest.raises(es.InputValidationError):
        es.availability([True], slo=1)


def test_spice() -> None:
    refs = [[("girl",), ("girl", "young"), ("girl", "ride", "horse"), ("horse",)], [("dog",)]]
    cands = [[("girl",), ("horse",), ("girl", "ride", "horse"), ("hat",)], [("cat",)]]
    r = es.spice(refs, cands)
    # image 1: 3 matches, P = 3/4, R = 3/4 -> F = 0.75; image 2: 0
    assert float(r) == pytest.approx(0.375)
    assert r.params["by_type"]["relation"] == 1.0 and r.params["by_type"]["attribute"] == 0.0
    per = es.spice(refs, cands, average=None).value
    assert per.tolist() == pytest.approx([0.75, 0.0])
    syn = {"cat": {"feline"}, "kitty": {"feline"}}
    assert float(es.spice([[("cat",)]], [[("kitty",)]], synonyms=lambda w: syn.get(w, set()))) == 1.0
    parsed = es.spice(["a dog runs"], ["a dog"], parser=lambda s: [(w,) for w in s.split() if len(w) > 1])
    assert parsed.params["precision"] == 1.0 and parsed.params["recall"] == 0.5
    assert float(es.spice([["dog"]], [["dog"]])) == 1.0
    assert float(es.spice([[]], [[]])) == 0.0
    for bad in (
        lambda: es.spice(refs, cands[:1]),
        lambda: es.spice(refs, cands, average="micro"),
        lambda: es.spice(refs, cands, synonyms=3),
        lambda: es.spice(["text"], [[("a",)]]),
        lambda: es.spice([[("a", "b", "c", "d")]], [[("a",)]]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_spice_wordnet_synonyms() -> None:
    wn = pytest.importorskip("nltk.corpus").wordnet
    try:
        wn.synsets("dog")
    except LookupError:
        pytest.skip("WordNet data not downloaded")
    assert float(es.spice([[("automobile",)]], [[("car",)]], synonyms="wordnet")) == 1.0


def test_every_new_metric_is_registered_and_documented() -> None:
    new = [
        m
        for m in es.list_metrics()
        if m.split(".")[0]
        in {"safety", "robustness", "uncertainty", "agents", "multilingual", "code", "long_context", "efficiency"}
    ]
    assert len(new) == 77
    for m in [*new, "text.spice"]:
        info = es.metric_info(m)
        assert info.definition and info.formula and info.references, m
    assert math.isfinite(len(es.llmsys.__all__))
