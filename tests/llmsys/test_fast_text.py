"""Fast paths in text and agreement metrics must equal the reference implementations."""

import random

import numpy as np
import pytest

import evalsuite as es

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
    "blue",
    "green",
    "runs",
    "fast",
    "slow",
    "house",
    "tree",
]


def _corpus(seed, k=12, nref=2):
    rnd = random.Random(seed)  # noqa: S311 - test data
    refs = [[" ".join(rnd.choices(V, k=rnd.randint(0, 25))) for _ in range(nref)] for _ in range(k)]
    preds = []
    for rs in refs:
        w = rs[0].split()
        rnd.shuffle(w)
        preds.append(" ".join(w[: rnd.randint(0, len(w))] + rnd.choices(V, k=rnd.randint(0, 4))))
    return refs, preds


@pytest.mark.parametrize("seed", range(6))
def test_bleu_ter_chrf_match_sacrebleu(seed):
    sacrebleu = pytest.importorskip("sacrebleu")
    refs, preds = _corpus(seed)
    cols = [list(x) for x in zip(*refs)]
    assert float(es.bleu(refs, preds)) == pytest.approx(sacrebleu.corpus_bleu(preds, cols).score, abs=1e-9)
    assert float(es.ter(refs, preds)) == pytest.approx(sacrebleu.corpus_ter(preds, cols).score, abs=1e-9)
    assert float(es.chrf(refs, preds, word_order=2)) == pytest.approx(
        sacrebleu.corpus_chrf(preds, cols, word_order=2).score, abs=1e-9
    )


@pytest.mark.parametrize("seed", range(4))
def test_rouge_matches_rouge_score(seed):
    rs = pytest.importorskip("rouge_score.rouge_scorer")
    refs, preds = _corpus(seed, nref=1)
    refs1 = [r[0] for r in refs]
    sc = rs.RougeScorer(["rouge1", "rouge2", "rougeL"])
    scores = [sc.score(r, p) for r, p in zip(refs1, preds)]
    for key, fn in (("rouge1", es.rouge_1), ("rouge2", es.rouge_2), ("rougeL", es.rouge_l)):
        assert float(fn(refs1, preds)) == pytest.approx(np.mean([s[key].fmeasure for s in scores]), abs=1e-12)


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval", "ratio"])
def test_krippendorff_matches_package(level):
    kr = pytest.importorskip("krippendorff")
    r = np.random.default_rng(0).integers(1, 6, (4, 200)).astype(float)
    r[np.random.default_rng(1).random(r.shape) < 0.1] = np.nan
    want = kr.alpha(reliability_data=r, level_of_measurement=level)
    assert float(es.krippendorff_alpha(r, level=level)) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize(
    "ratings",
    [
        np.random.default_rng(0).integers(0, 3, (50, 4)),
        [["a", "b"], ["a", "a"], ["b", "c"]],
        [[1, "1"], [2, 2], [1, 2]],
    ],
)
def test_fleiss_kappa_matches_statsmodels(ratings):
    ir = pytest.importorskip("statsmodels.stats.inter_rater")
    a = np.asarray(ratings, dtype=object)
    cats = sorted({str(x) for x in a.ravel()})
    table = np.array([[sum(str(x) == c for x in row) for c in cats] for row in a], dtype=float)
    got = es.fleiss_kappa(ratings)
    assert float(got) == pytest.approx(ir.fleiss_kappa(table), abs=1e-12)
    assert got.params["categories"] == cats


def test_language_id_accuracy_matches_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    langs = np.array(["en", "fr", "de", "hi"])
    t = rng.choice(langs, 500)
    p = np.where(rng.random(500) < 0.8, t, rng.choice(langs, 500))
    for tt, pp in ((t, p), (t.tolist(), p.tolist())):
        r = es.language_id_accuracy(tt, pp)
        assert float(r) == pytest.approx(skm.accuracy_score(t, p))
        assert r.params["macro_f1"] == pytest.approx(skm.f1_score(t, p, average="macro"))


def test_moverscore_solvers_agree(monkeypatch):
    import sys

    rng = np.random.default_rng(3)
    x = [rng.normal(size=(int(rng.integers(3, 9)), 6)) for _ in range(8)]
    y = [rng.normal(size=(int(rng.integers(3, 9)), 6)) for _ in range(8)]
    w1 = [rng.random(len(a)) + 0.1 for a in x]
    w2 = [rng.random(len(a)) + 0.1 for a in y]
    first = (float(es.moverscore(x, y)), float(es.moverscore(x, y, reference_weights=w1, prediction_weights=w2)))
    monkeypatch.setitem(sys.modules, "ot", None)  # force the SciPy assignment / LP paths
    second = (float(es.moverscore(x, y)), float(es.moverscore(x, y, reference_weights=w1, prediction_weights=w2)))
    assert first == pytest.approx(second, abs=1e-12)


@pytest.mark.parametrize("spacing", [None, (0.5, 2.0)])
def test_surface_distances_match_distance_transform(spacing):
    from scipy import ndimage

    from evalsuite.vision.segmentation import _surface

    rng = np.random.default_rng(5)
    a = ndimage.binary_dilation(rng.random((40, 40)) < 0.02, iterations=2)
    b = ndimage.binary_dilation(rng.random((40, 40)) < 0.02, iterations=2)
    sa, sb = _surface(a), _surface(b)
    d_ab = ndimage.distance_transform_edt(~sb, sampling=spacing)[sa]
    d_ba = ndimage.distance_transform_edt(~sa, sampling=spacing)[sb]
    hd = float(es.hausdorff_distance(a.astype(int), b.astype(int), labels=[1], spacing=spacing))
    asd = float(es.average_surface_distance(a.astype(int), b.astype(int), labels=[1], spacing=spacing))
    assert hd == pytest.approx(max(d_ab.max(), d_ba.max()), abs=1e-12)
    assert asd == pytest.approx((d_ab.sum() + d_ba.sum()) / (d_ab.size + d_ba.size), abs=1e-12)
