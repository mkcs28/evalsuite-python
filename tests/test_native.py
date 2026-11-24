"""The compiled core (evalsuite._core) must give exactly the Python reference's results.

Every kernel is compared with the pure-Python implementation it replaces by switching the core off in the
modules that use it. Tests that need the core skip when the package runs without it (the pure-Python wheel).
"""

import contextlib
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import evalsuite as es
from evalsuite import _native
from evalsuite.stats import _resolve as resolve_mod
from evalsuite.text import generation as generation_mod
from evalsuite.vision import detection as detection_mod

needs_core = pytest.mark.skipif(not _native.HAVE_NATIVE, reason="compiled core not built")


@contextlib.contextmanager
def pure_python():
    saved = (generation_mod.core, detection_mod.core, resolve_mod.core, resolve_mod.MetricCall._setup_compression)
    generation_mod.core = detection_mod.core = resolve_mod.core = None
    resolve_mod.MetricCall._setup_compression = lambda self: None
    try:
        yield
    finally:
        generation_mod.core, detection_mod.core, resolve_mod.core, resolve_mod.MetricCall._setup_compression = (
            saved
        )


def both(fn):
    """(result with the core, result in pure Python); exceptions are compared by type."""

    def run():
        try:
            return fn()
        except Exception as exc:
            return ("error", type(exc).__name__)

    a = run()
    with pure_python():
        b = run()
    return a, b


def assert_same(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        assert a == b
        return
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert x.shape == y.shape
    assert np.array_equal(x, y, equal_nan=True), (x, y)


def test_backend_and_version():
    assert _native.backend() in ("rust", "python")
    cargo = Path(__file__).resolve().parents[1] / "rust" / "Cargo.toml"
    if cargo.exists():
        version = re.search(r'^version = "(.+)"', cargo.read_text(encoding="utf-8"), re.M).group(1)
        assert version == es.__version__, (
            "rust/Cargo.toml and src/evalsuite/version.py must carry the same version"
        )
    if _native.HAVE_NATIVE:
        assert _native.core.__version__ == es.__version__


def test_pure_python_switch():
    env = dict(os.environ, EVALSUITE_PURE_PYTHON="1")
    out = subprocess.run(
        [sys.executable, "-c", "from evalsuite import _native; print(_native.backend())"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == "python"


V = [
    "the",
    "a",
    "of",
    "cat",
    "dog",
    "sat",
    "on",
    "mat",
    "and",
    "big",
    "red",
    ",",
    ".",
    "!",
    "?",
    "blue",
    "Ünïcödé",
    "naïve",
    "ΣΑΣ",
    "résumé",
    "42",
    "x-y",
]


def _corpus(seed, k, nref):
    rnd = random.Random(seed)  # noqa: S311 - test data
    refs = [[" ".join(rnd.choices(V, k=rnd.randint(0, 25))) for _ in range(nref)] for _ in range(k)]
    preds = []
    for rs in refs:
        w = rs[0].split()
        rnd.shuffle(w)
        preds.append(" ".join(w[: rnd.randint(0, len(w))] + rnd.choices(V, k=rnd.randint(0, 5))))
    return refs, preds


TEXT_CASES = {
    "bleu": lambda r, p: float(es.bleu(r, p)),
    "bleu_lowercase_intl": lambda r, p: float(es.bleu(r, p, lowercase=True, tokenize="intl")),
    "sentence_bleu": lambda r, p: es.sentence_bleu(r, p, average=None).value,
    "self_bleu": lambda r, p: float(es.self_bleu(p)),
    "chrf": lambda r, p: float(es.chrf(r, p)),
    "chrf++": lambda r, p: float(es.chrf(r, p, word_order=2, lowercase=True)),
    "chrf_whitespace_eps": lambda r, p: float(es.chrf(r, p, whitespace=True, eps_smoothing=True, beta=1)),
    "ter": lambda r, p: float(es.ter(r, p)),
    "ter_case": lambda r, p: float(es.ter(r, p, case_sensitive=True)),
    "rouge_1": lambda r, p: es.rouge_1(r, p, average=None).value,
    "rouge_2": lambda r, p: es.rouge_2(r, p, average=None, measure="recall").value,
    "rouge_l": lambda r, p: es.rouge_l(r, p, average=None).value,
    "rouge_lsum": lambda r, p: (
        es.rouge_lsum(
            [[x.replace(" , ", "\n") for x in rs] for rs in r], [x.replace(" . ", "\n") for x in p], average=None
        ).value
    ),
    "meteor": lambda r, p: es.meteor(r, p, average=None).value,
    "meteor_no_stem": lambda r, p: (
        es.meteor(r, p, stemmer=None, average=None, alpha=0.8, beta=2.5, gamma=0.3).value
    ),
}


@needs_core
@pytest.mark.parametrize("name", sorted(TEXT_CASES))
@pytest.mark.parametrize("seed", range(12))
def test_text_kernels_match_python(name, seed):
    if name.startswith("meteor"):
        pytest.importorskip("nltk")
    rnd = random.Random(seed)  # noqa: S311 - test data
    refs, preds = _corpus(seed, rnd.randint(2, 12), rnd.randint(1, 3))
    assert_same(*both(lambda: TEXT_CASES[name](refs, preds)))


@needs_core
def test_chrf_falls_back_for_text_the_core_cannot_take():
    refs, preds = [["a \ud800 b"]], ["a b"]  # a lone surrogate cannot cross into Rust
    assert_same(*both(lambda: float(es.chrf(refs, preds))))


def _scene(rng):
    yt, yp = [], []
    for _ in range(int(rng.integers(1, 20))):
        g, d = int(rng.integers(0, 7)), int(rng.integers(0, 9))
        gb = rng.uniform(0, 300, (g, 2))
        gb = np.c_[gb, gb + rng.uniform(4, 150, (g, 2))]
        m = min(g, d)
        extra = rng.uniform(0, 300, (d - m, 2))
        db = np.r_[gb[:m] + rng.normal(0, 8, (m, 4)), np.c_[extra, extra + rng.uniform(4, 150, (d - m, 2))]]
        db = np.c_[np.minimum(db[:, :2], db[:, 2:]), np.maximum(db[:, :2], db[:, 2:])] if d else np.zeros((0, 4))
        yt.append({"boxes": gb, "labels": rng.integers(0, 3, g), "iscrowd": (rng.random(g) < 0.12).astype(int)})
        yp.append({"boxes": db, "labels": rng.integers(0, 3, d), "scores": rng.random(d)})
    return yt, yp


COCO_KEYS = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_1", "mar_10", "mar_100"]


@needs_core
@pytest.mark.parametrize("seed", range(15))
def test_coco_matching_matches_python(seed):
    yt, yp = _scene(np.random.default_rng(seed))
    assert_same(*both(lambda: [float(es.detection_report(yt, yp)[k]) for k in COCO_KEYS]))
    assert_same(*both(lambda: float(es.mean_average_precision(yt, yp, iou_threshold=0.5))))


CONFUSION = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1", "fbeta", "jaccard", "mcc", "npv", "precision"]


@pytest.mark.parametrize("metric", [*CONFUSION, "recall", "specificity"])
@pytest.mark.parametrize("kind", ["binary", "multiclass", "strings", "bool"])
def test_resample_compression_is_exact(metric, kind):
    """Counting-based evaluation equals calling the metric on the resampled observations (no core needed)."""
    rng = np.random.default_rng(len(metric) + len(kind))
    k = 2 if kind in ("binary", "bool") else 3
    y = rng.integers(0, k, 120)
    p = np.where(rng.random(120) < 0.7, y, rng.integers(0, k, 120))
    if kind == "strings":
        y, p = np.array(list("abc"))[y], np.array(list("abc"))[p]
    if kind == "bool":
        y, p = y.astype(bool), p.astype(bool)
    kw = {"beta": 2.0} if metric == "fbeta" else {}
    if k > 2 and metric in ("npv", "specificity", "precision", "recall", "f1", "fbeta", "jaccard"):
        kw["average"] = "macro"
    call = resolve_mod.MetricCall(getattr(es, metric), y, p, kwargs=kw)
    assert call.compressible
    for idx in [rng.integers(0, 120, 120) for _ in range(20)] + [np.zeros(120, dtype=np.int64)]:
        direct, counted = call(idx, compress=False), call(idx)
        assert counted == direct or (np.isnan(direct) and np.isnan(counted))


@pytest.mark.parametrize("method", ["bca", "percentile", "basic"])
@pytest.mark.parametrize("metric", ["accuracy", "f1", "mcc"])
def test_bootstrap_intervals_unchanged(method, metric):
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, 400)
    p = np.where(rng.random(400) < 0.7, y, rng.integers(0, 3, 400))
    kw = {"average": "macro"} if metric == "f1" else {}
    a, b = both(lambda: es.bootstrap_ci(metric, y, p, n_resamples=200, random_state=1, method=method, **kw))
    assert (a.estimate, a.low, a.high) == (b.estimate, b.low, b.high)


def test_paired_bootstrap_and_compare_unchanged():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    pa = np.where(rng.random(300) < 0.8, y, 1 - y)
    pb = np.where(rng.random(300) < 0.75, y, 1 - y)
    a, b = both(lambda: es.paired_bootstrap_test(es.accuracy, y, pa, pb, n_resamples=200, random_state=0))
    assert (a.statistic, a.p_value) == (b.statistic, b.p_value)
    a, b = both(lambda: es.compare(y, {"a": pa, "b": pb}, metrics=["accuracy"], n_resamples=200, random_state=0))
    assert str(a) == str(b)


def test_not_compressible_cases():
    y = np.array([0, 1, 1, 0])
    assert not resolve_mod.MetricCall(es.accuracy, y, y, sample_weight=[1, 2, 1, 1]).compressible  # user weights
    assert not resolve_mod.MetricCall(es.accuracy, y, y.astype(float)).compressible  # mixed dtypes
    assert not resolve_mod.MetricCall(
        es.roc_auc, y, None, y_prob=[0.1, 0.8, 0.6, 0.3]
    ).compressible  # probabilities
    many = np.arange(200)
    assert not resolve_mod.MetricCall(es.accuracy, many, many).compressible  # too many classes


@needs_core
def test_core_rejects_bad_arguments():
    core = _native.core
    codes, cv = _native.array_view(np.array([0, 1, 1], dtype=np.uint32), np.uint32)
    idx, iv = _native.array_view(np.array([[0, 9]], dtype=np.int64), np.int64)
    with pytest.raises(ValueError, match="out of range"):
        core.code_counts(cv, 2, iv, 1)
    with pytest.raises(ValueError, match="aligned"):
        core.code_counts((1, 3), 2, iv, 1)
    with pytest.raises(ValueError, match="sizes"):
        core.coco_match((0, 0), (0, 0), (0, 0), (0, 0), 1, 1, 1, [0.5])
    with pytest.raises(ValueError, match="same length"):
        core.ter_corpus([[1]], [])
    assert core.chrf_f([0.0, 0.0, 0.0], 1, 2.0, False) == 0.0
    del codes, idx
