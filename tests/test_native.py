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
from evalsuite.classification import metrics as clf_mod
from evalsuite.llmsys import _common as llm_common_mod
from evalsuite.llmsys import code as code_mod
from evalsuite.llmsys import safety as safety_mod
from evalsuite.stats import _resolve as resolve_mod
from evalsuite.stats import intervals as intervals_mod
from evalsuite.text import generation as generation_mod
from evalsuite.text import judge as judge_mod
from evalsuite.text import qa as qa_mod
from evalsuite.text import reasoning as reasoning_mod
from evalsuite.vision import detection as detection_mod
from evalsuite.vision import segmentation as segmentation_mod

needs_core = pytest.mark.skipif(not _native.HAVE_NATIVE, reason="compiled core not built")


@contextlib.contextmanager
def pure_python():
    mods = (
        generation_mod,
        detection_mod,
        resolve_mod,
        code_mod,
        clf_mod,
        llm_common_mod,
        safety_mod,
        reasoning_mod,
        judge_mod,
        intervals_mod,
        qa_mod,
        segmentation_mod,
    )
    saved = [m.core for m in mods], resolve_mod.MetricCall._setup_compression
    for m in mods:
        m.core = None
    resolve_mod.MetricCall._setup_compression = lambda self: None
    try:
        yield
    finally:
        for m, c in zip(mods, saved[0]):
            m.core = c
        resolve_mod.MetricCall._setup_compression = saved[1]


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
    "cider": lambda r, p: float(es.cider(r, p)),
    "cider_per_example_o2": lambda r, p: es.cider(r, p, max_order=2, sigma=3.0, average=None).value,
    "distinct_1": lambda r, p: float(es.distinct_n(p)),
    "distinct_3_lower": lambda r, p: float(es.distinct_n(p, n=3, lowercase=True)),
    "codebleu_bleu_terms": lambda r, p: [
        es.codebleu([x[0] for x in r], p).params[k] for k in ("ngram_match", "weighted_ngram_match")
    ],
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


@needs_core
@pytest.mark.parametrize("seed", range(6))
def test_coco_full_evaluation_with_ties_and_areas(seed):
    rng = np.random.default_rng(100 + seed)
    yt, yp = _scene(rng)
    for t, p in zip(yt, yp):
        p["scores"] = np.round(p["scores"], 1)  # many tied scores: the stable sort order must match
        t["area"] = rng.uniform(10, 20000, len(t["labels"]))
    assert_same(*both(lambda: [float(es.detection_report(yt, yp)[k]) for k in COCO_KEYS]))


@needs_core
def test_results_do_not_depend_on_thread_count():
    code = (
        "import evalsuite as es, numpy as np, random;"
        "r=random.Random(1);W='a b c d e f g'.split();"
        "s=lambda:' '.join(r.choices(W,k=r.randint(3,30)));"
        "refs=[[s(),s()] for _ in range(300)];preds=[s() for _ in range(300)];"
        "print(repr([float(es.bleu(refs,preds)),float(es.chrf(refs,preds)),float(es.ter(refs,preds)),"
        "float(es.cider(refs,preds)),float(es.meteor(refs,preds,stemmer=None)),"
        "float(es.bootstrap_ci('accuracy',np.arange(500)%3,np.arange(500)%2,random_state=0,resampling='native').low),"
        "float(es.bootstrap_ci('roc_auc',np.arange(400)%2,y_prob=(np.arange(400)*7%13)/13,random_state=0).low),"
        "es.weat_effect_size(*[np.sin(np.arange(m*8).reshape(m,8)+m) for m in (20,21,5,6)],"
        "random_state=1,resampling='native').params['p_value'],"
        "float(es.hausdorff_distance([np.arange(900).reshape(30,30)%7>2],[np.arange(900).reshape(30,30)%5>1]))]))"
    )
    out = []
    for threads in ("1", "4"):
        env = {**os.environ, "EVALSUITE_NUM_THREADS": threads}
        out.append(
            subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)  # noqa: S603 - fixed command
        )
    assert out[0].stdout == out[1].stdout


@needs_core
@pytest.mark.parametrize("stratify", [False, True])
def test_native_resampling(stratify):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    p = np.where(rng.random(400) < 0.7, y, rng.integers(0, 3, 400))
    kw = {"random_state": 5, "stratify": stratify, "resampling": "native"}
    a = es.bootstrap_ci("f1", y, p, average="macro", **kw)
    b = es.bootstrap_ci("f1", y, p, average="macro", **kw)
    assert (a.low, a.high) == (b.low, b.high) and a.params["resampling"] == "native"
    ref = es.bootstrap_ci("f1", y, p, average="macro", random_state=5, stratify=stratify)
    assert abs(a.low - ref.low) < 0.03 and abs(a.high - ref.high) < 0.03
    from evalsuite.stats.intervals import resample_indices

    idx = resample_indices(400, 200, np.random.default_rng(1), y if stratify else None, "native")
    assert idx.shape == (200, 400) and idx.min() >= 0 and idx.max() < 400
    if stratify:
        assert all((np.bincount(y[row], minlength=3) == np.bincount(y, minlength=3)).all() for row in idx)
    assert es.paired_bootstrap_test("accuracy", y, p, y, resampling="native", random_state=0).p_value < 0.01


def test_resampling_argument_is_checked():
    with pytest.raises(es.InputValidationError):
        es.bootstrap_ci("accuracy", [0, 1] * 20, [0, 1] * 20, resampling="fast")


def _probs(rng, n, k, ties):
    p = rng.dirichlet(np.ones(k), n)
    return np.round(p, 2) if ties else p


@needs_core
@pytest.mark.parametrize("seed", range(8))
def test_roc_auc_and_average_precision_match_python(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 3000))
    ties = bool(seed % 2)
    w = rng.random(n) * 2 if seed % 3 == 0 else None
    yb = rng.integers(0, 2, n)
    pb = np.round(rng.random(n), 2) if ties else rng.random(n)
    y3 = rng.integers(0, 4, n)
    p3 = _probs(rng, n, 4, ties)
    ym = rng.integers(0, 2, (n, 3))
    pm = np.round(rng.random((n, 3)), 2) if ties else rng.random((n, 3))
    cases = [
        lambda: es.roc_auc(yb, pb, sample_weight=w).value,
        lambda: es.average_precision(yb, pb, sample_weight=w).value,
        lambda: es.roc_auc(y3, p3, average=None, sample_weight=w).value,
        lambda: es.roc_auc(y3, p3, average="weighted").value,
        lambda: es.roc_auc(y3, p3, multi_class="ovo").value,
        lambda: es.average_precision(y3, p3, average="macro", sample_weight=w).value,
        lambda: es.roc_auc(ym, pm, average="micro").value,
        lambda: es.average_precision(ym, pm, average=None).value,
        lambda: [
            float(v["value"]) for v in es.evaluate(y3, p3.argmax(1), y_prob=p3).to_dict()["metrics"].values()
        ],
        lambda: es.roc_auc(np.zeros(n, int), pb).value,  # undefined: same error type
    ]
    for case in cases:
        assert_same(*both(case))


def _answers(rnd, groups, size):
    pool = ["Paris", "paris ", "PARIS", "London", "Rome", "rome", "Berlin", "Madrid", ""]
    return [[rnd.choice(pool) for _ in range(rnd.randint(2, size))] for _ in range(groups)]


@needs_core
@pytest.mark.parametrize("seed", range(6))
def test_llm_consistency_metrics_match_python(seed):
    rnd = random.Random(seed)  # noqa: S311 - test data
    rows = _answers(rnd, rnd.randint(1, 60), 9)
    langs = [{f"l{j}": a for j, a in enumerate(r)} for r in rows]
    refs = [rnd.choice(r) for r in rows]
    cases = [
        lambda: es.paraphrase_consistency(rows).to_dict(),
        lambda: es.response_stability(rows).to_dict(),
        lambda: es.cross_document_consistency(rows).to_dict(),
        lambda: es.cross_lingual_consistency(langs).to_dict(),
        lambda: es.majority_vote_accuracy(refs, rows, average=None).value.tolist(),
        lambda: es.majority_vote_accuracy(refs, rows, normalize=False).value,
        lambda: es.paraphrase_consistency([[1, 1.0, True], [2, "2"]]).to_dict(),  # non-strings: Python path
    ]
    for case in cases:
        a, b = both(case)
        assert a == b


@needs_core
@pytest.mark.parametrize("seed", range(4))
def test_exposure_matches_python(seed):
    rng = np.random.default_rng(seed)
    k = int(rng.integers(1, 50))
    canary = np.round(rng.normal(size=k), 1)
    cands = [np.round(rng.normal(size=int(rng.integers(1, 40))), 1).tolist() for _ in range(k)]
    for kw in ({}, {"space_size": 1e6}):
        a, b = both(lambda kw=kw: es.exposure(canary, cands, **kw).to_dict())
        assert a == b
    assert both(lambda: es.exposure([0.5], [[]]))[0] == both(lambda: es.exposure([0.5], [[]]))[1]
    assert both(lambda: es.exposure([0.5], [[float("nan")]]))[0][0] == "error"


@needs_core
@pytest.mark.parametrize("seed", range(4))
def test_ragged_number_metrics_match_python(seed):
    rng = np.random.default_rng(seed)
    rows = [np.round(rng.random(int(rng.integers(0, 30))), 2).tolist() for _ in range(int(rng.integers(1, 80)))]
    rows[0] = rows[0] or [0.5]
    counts = [[int(x) for x in rng.integers(0, 3, int(rng.integers(0, 6)))] for _ in range(len(rows))]
    counts[0] = counts[0] or [1]
    tox = [r or [0.0] for r in rows]
    cases = [
        lambda: es.groundedness(rows).to_dict(),
        lambda: es.groundedness(rows, average="micro").to_dict(),
        lambda: repr(es.groundedness(rows, average=None).value.tolist()),
        lambda: es.groundedness([[0.5, 1.5]]).to_dict(),  # out of range: same error
        lambda: es.citation_coverage(counts).to_dict(),
        lambda: es.toxicity_score(tox, attributes={"hate": tox}).to_dict(),
        lambda: es.toxicity_score([0.2, [0.9, 0.1], 0.4]).to_dict(),
        lambda: es.toxicity_score([[1.2]]).to_dict(),
    ]
    for case in cases:
        a, b = both(case)
        assert a == b


@needs_core
@pytest.mark.parametrize("seed", range(4))
def test_elo_and_bradley_terry_match_python(seed):
    rnd = random.Random(seed)  # noqa: S311 - test data
    models = [f"m{i}" for i in range(rnd.randint(2, 9))]
    comps = []
    for _ in range(rnd.randint(1, 400)):
        a, b = rnd.sample(models, 2)
        comps.append((a, b, rnd.choice(["win", "loss", "tie"])))
    for case in (
        lambda: es.elo_ratings(comps).value.tolist(),
        lambda: es.elo_ratings(comps, k=32, base=2.0, scale=100.0).value.tolist(),
        lambda: es.bradley_terry(comps).value.tolist(),
        lambda: es.bradley_terry(comps, scale="elo").value.tolist(),
    ):
        a, b = both(case)
        assert a == b


@needs_core
@pytest.mark.parametrize("seed", range(6))
def test_bootstrap_roc_auc_and_ap_match_python(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 1500))
    y = rng.integers(0, 2, n)
    if seed % 3 == 0:
        y = np.where(y == 1, "pos", "neg")
    p = np.round(rng.random(n), 2)
    w = rng.random(n) * 2 if seed % 2 else None
    kw = {"pos_label": "pos"} if seed % 3 == 0 else {}
    for metric in ("roc_auc", "average_precision"):
        for method in ("bca", "percentile"):

            def run(metric=metric, method=method):
                ci = es.bootstrap_ci(
                    metric, y, y_prob=p, sample_weight=w, random_state=seed, n_resamples=200, method=method, **kw
                )
                return (ci.estimate, ci.low, ci.high, ci.params["failed_resamples"])

            a, b = both(run)
            assert repr(a) == repr(b)


@needs_core
@pytest.mark.parametrize("seed", range(4))
def test_delong_matches_python(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 3000))
    y = rng.integers(0, 2, n)
    a = np.round(rng.random(n), 2)
    b = np.round(np.clip(a + rng.normal(0, 0.1, n), 0, 1), 2)
    x, z = both(lambda: es.delong_test(y, a, b))
    assert (x.statistic, x.p_value, x.estimate, x.ci.low, x.params) == (
        z.statistic,
        z.p_value,
        z.estimate,
        z.ci.low,
        z.params,
    )


@needs_core
@pytest.mark.parametrize("seed", range(4))
def test_qa_metrics_match_python(seed):
    rnd = random.Random(seed)  # noqa: S311 - test data
    pool = [
        "The",
        "a",
        "an",
        "the",
        "cat,",
        "sat!",
        "on",
        "mat.",
        "Paris",
        "PARIS",
        "café",
        "ß",
        "\t",
        "\n",
        "\x1c",
        "x_y",
        "1.5",
        "A-B",
    ]
    preds = [" ".join(rnd.choices(pool, k=rnd.randint(0, 8))) for _ in range(rnd.randint(1, 60))]
    refs = [[" ".join(rnd.choices(pool, k=rnd.randint(0, 8))) for _ in range(rnd.randint(1, 3))] for _ in preds]
    for case in (
        lambda: es.exact_match(refs, preds, average=None).value.tolist(),
        lambda: es.token_f1(refs, preds, average=None).value.tolist(),
        lambda: es.token_f1(refs, preds, normalize=False).value,
        lambda: es.exact_match(refs, preds, normalize=str.lower).value,
    ):
        a, b = both(case)
        assert repr(a) == repr(b)
    assert [qa_mod.normalize_answer(t) for t in preds] == qa_mod.normalize_answers(preds)


def _blob(rng, shape):
    m = np.zeros(shape, bool)
    grid = np.ogrid[tuple(slice(0, s) for s in shape)]
    for _ in range(int(rng.integers(1, 4))):
        c = [int(rng.integers(0, s)) for s in shape]
        rad = int(rng.integers(1, max(2, min(shape) // 3)))
        m |= sum((g - ci) ** 2 for g, ci in zip(grid, c)) <= rad * rad
    return m.astype(int)


@needs_core
@pytest.mark.parametrize("seed", range(6))
def test_surface_distances_match_python(seed):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(4, 70)), int(rng.integers(4, 70))) if seed % 3 else (int(rng.integers(4, 20)),) * 3
    a = [_blob(rng, shape), rng.integers(0, 3, shape)]
    b = [_blob(rng, shape), rng.integers(0, 3, shape)]
    sp = None if seed % 2 else tuple(np.round(rng.uniform(0.3, 2, len(shape)), 3))
    for case in (
        lambda: es.hausdorff_distance(a, b, spacing=sp, average=None).value.tolist(),
        lambda: es.hausdorff_distance(a, b, spacing=sp, percentile=95).value,
        lambda: es.average_surface_distance(a, b, spacing=sp, average=None).value.tolist(),
    ):
        x, z = both(case)
        assert repr(x) == repr(z)


@needs_core
@pytest.mark.parametrize("seed", range(4))
def test_calibration_matches_python(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 4000))
    y = rng.integers(0, 2, n)
    p = np.round(rng.random(n), 2)
    w = rng.random(n) if seed % 2 else None
    for kw in ({}, {"strategy": "quantile"}, {"n_bins": 7}):
        x, z = both(lambda kw=kw: es.expected_calibration_error(y, p, sample_weight=w, **kw).value)
        assert x == z
        x, z = both(lambda kw=kw: [c.tolist() for c in es.calibration_curve(y, p, sample_weight=w, **kw)])
        assert x == z


@needs_core
def test_weat_exact_matches_python_and_native_is_seeded():
    rng = np.random.default_rng(0)
    X, Y, A, B = (np.round(rng.normal(size=(m, 12)), 1) for m in (6, 5, 4, 4))
    x, z = both(lambda: es.weat_effect_size(X, Y, A, B).to_dict())
    assert x == z
    X, Y = (rng.normal(size=(20, 12)) for _ in range(2))
    a = es.weat_effect_size(X, Y, A, B, random_state=3, resampling="native").params["p_value"]
    b = es.weat_effect_size(X, Y, A, B, random_state=3, resampling="native").params["p_value"]
    c = es.weat_effect_size(X, Y, A, B, random_state=3).params["p_value"]
    assert a == b and abs(a - c) < 0.05
