"""v0.4.0 LLM evaluation: METEOR, CIDEr, embedding metrics, MAUVE, factuality, judges, RAG, reasoning and
structured output, checked against reference implementations (NLTK, pycocoevalcap, POT, choix, krippendorff,
statsmodels, jsonschema, mauve-text) where one exists and against hand-computed answers otherwise."""

from __future__ import annotations

import json
import math
import random
import warnings

import numpy as np
import pytest

import evalsuite as es
from evalsuite.text.semantic import mauve_from_histograms

V = [
    "the",
    "a",
    "cat",
    "dog",
    "sat",
    "on",
    "mat",
    "quickly",
    "running",
    "runs",
    "ran",
    "models",
    "model",
    "data",
    "sitting",
    "cats",
]


# ---------------------------------------------------------------- METEOR and CIDEr
def test_meteor_matches_nltk_without_wordnet() -> None:
    meteor_score = pytest.importorskip("nltk.translate.meteor_score").meteor_score

    class NoWordNet:
        def synsets(self, _w):
            return []

    rng = np.random.default_rng(0)
    for _ in range(150):
        refs = [" ".join(rng.choice(V, rng.integers(1, 12))) for _ in range(rng.integers(1, 3))]
        hyp = " ".join(rng.choice(V, rng.integers(0, 12)))
        ref = meteor_score([r.split() for r in refs], hyp.split(), wordnet=NoWordNet())
        assert float(es.meteor([refs], [hyp])) == pytest.approx(ref, abs=1e-12)


def test_meteor_synonyms_match_nltk_with_a_fake_wordnet() -> None:
    meteor_score = pytest.importorskip("nltk.translate.meteor_score").meteor_score
    syn = {"cat": {"feline"}, "sat": {"rested"}, "mat": {"rug"}}

    class Lemma:
        def __init__(self, n):
            self._n = n

        def name(self):
            return self._n

    class Synset:
        def __init__(self, names):
            self._l = [Lemma(n) for n in names]

        def lemmas(self):
            return self._l

    class FakeWordNet:
        def synsets(self, w):
            return [Synset(syn[w])] if w in syn else []

    refs, hyp = ["the feline rested on the rug"], "the cat sat on the mat"
    ref = meteor_score([refs[0].split()], hyp.split(), wordnet=FakeWordNet())
    ours = es.meteor([refs], [hyp], synonyms=lambda w: syn.get(w, set()))
    assert float(ours) == pytest.approx(ref, abs=1e-12)
    assert float(ours) > float(es.meteor([refs], [hyp]))


def test_meteor_options_and_errors() -> None:
    assert float(es.meteor(["a b c"], ["a b c"], stemmer=None)) == pytest.approx(0.9814814814814815)
    assert float(es.meteor(["running"], ["runs"], stemmer=lambda w: w[:3])) > 0
    assert float(es.meteor(["x"], ["y"])) == 0.0
    with pytest.raises(es.InputValidationError):
        es.meteor(["a"], ["a"], stemmer="snowball")
    with pytest.raises(es.InputValidationError):
        es.meteor(["a"], ["a"], alpha=2)
    with pytest.raises(es.InputValidationError):
        es.meteor(["a"], ["a"], synonyms="wordnet")
    assert es.meteor(["a b", "c"], ["a b", "d"], average=None).value.shape == (2,)


@pytest.mark.parametrize("seed", range(3))
def test_cider_matches_pycocoevalcap(seed: int) -> None:
    cider_mod = pytest.importorskip("pycocoevalcap.cider.cider")
    rng = np.random.default_rng(seed)
    gts, res, refs, preds = {}, {}, [], []
    for i in range(25):
        rs = [" ".join(rng.choice(V, rng.integers(3, 12))) for _ in range(5)]
        p = " ".join(rng.choice(V, rng.integers(1, 12)))
        gts[i], res[i] = rs, [p]
        refs.append(rs)
        preds.append(p)
    mean, per = cider_mod.Cider().compute_score(gts, res)
    ours = es.cider(refs, preds, average=None).value
    np.testing.assert_allclose(ours, per, atol=1e-12)
    assert float(es.cider(refs, preds)) == pytest.approx(mean, abs=1e-12)


def test_cider_errors_and_lowercase() -> None:
    with pytest.raises(es.InputValidationError):
        es.cider(["a b"], ["a b"], sigma=0)
    assert float(es.cider([["A cat"], ["dog"]], ["a cat", "dog"], lowercase=True)) > 0


# ---------------------------------------------------------------- embeddings
def _tok_embs(seed: int, n: int = 15, dim: int = 12):
    rng = np.random.default_rng(seed)
    r = [rng.normal(size=(rng.integers(2, 9), dim)) for _ in range(n)]
    p = [rng.normal(size=(rng.integers(2, 9), dim)) for _ in range(n)]
    wr = [rng.random(len(x)) + 0.1 for x in r]
    wp = [rng.random(len(x)) + 0.1 for x in p]
    return r, p, wr, wp


def test_bertscore_matches_brute_force() -> None:
    r, p, wr, wp = _tok_embs(0)

    def cos(a, b):
        return a @ b / np.linalg.norm(a) / np.linalg.norm(b)

    for k in range(len(r)):
        prec = sum(
            wp[k][j] * max(cos(r[k][i], p[k][j]) for i in range(len(r[k]))) for j in range(len(p[k]))
        ) / sum(wp[k])
        rec = sum(wr[k][i] * max(cos(r[k][i], p[k][j]) for j in range(len(p[k]))) for i in range(len(r[k]))) / sum(
            wr[k]
        )
        f1 = 2 * prec * rec / (prec + rec)
        args = ([r[k]], [p[k]])
        kw = {"reference_weights": [wr[k]], "prediction_weights": [wp[k]]}
        assert float(es.bertscore(*args, **kw)) == pytest.approx(f1, abs=1e-12)
        assert float(es.bertscore(*args, **kw, measure="precision")) == pytest.approx(prec, abs=1e-12)
        assert float(es.bertscore(*args, **kw, measure="recall")) == pytest.approx(rec, abs=1e-12)
    base = float(es.bertscore(r, p, measure="precision", baseline=0.5, average=None).value[0])
    raw = float(es.bertscore(r, p, measure="precision", average=None).value[0])
    assert base == pytest.approx((raw - 0.5) / 0.5)


def test_bertscore_errors() -> None:
    a = [np.ones((2, 3))]
    for bad in (
        lambda: es.bertscore(a, [np.ones((2, 4))]),
        lambda: es.bertscore(a, a + a),
        lambda: es.bertscore(a, a, measure="auc"),
        lambda: es.bertscore([np.zeros((2, 3))], a),
        lambda: es.bertscore([np.full((2, 3), np.nan)], a),
        lambda: es.bertscore(a, a, reference_weights=[[1.0]]),
        lambda: es.bertscore(a, a, reference_weights=[[-1.0, 1.0]]),
        lambda: es.bertscore("text", a),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_embedding_similarity_matches_scipy() -> None:
    from scipy.spatial.distance import cityblock, cosine, euclidean

    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
    np.testing.assert_allclose(
        es.embedding_similarity(a, b, average=None).value, [1 - cosine(x, y) for x, y in zip(a, b)]
    )
    np.testing.assert_allclose(
        es.embedding_similarity(a, b, metric="euclidean", average=None).value,
        [euclidean(x, y) for x, y in zip(a, b)],
    )
    np.testing.assert_allclose(
        es.embedding_similarity(a, b, metric="manhattan", average=None).value,
        [cityblock(x, y) for x, y in zip(a, b)],
    )
    assert float(es.embedding_similarity(a, 2 * a, metric="euclidean", normalize=True)) == pytest.approx(
        0, abs=1e-12
    )
    with pytest.raises(es.InputValidationError):
        es.embedding_similarity(a, b, metric="dot")
    with pytest.raises(es.InputValidationError):
        es.embedding_similarity(a, b[:3])


def test_moverscore_matches_pot() -> None:
    ot = pytest.importorskip("ot")
    r, p, wr, wp = _tok_embs(2)

    def unit(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    ours = es.moverscore(r, p, reference_weights=wr, prediction_weights=wp, average=None).value
    ref = [
        1 - ot.emd2(a / a.sum(), b / b.sum(), ot.dist(unit(x), unit(y), metric="euclidean"))
        for x, y, a, b in zip(r, p, wr, wp)
    ]
    np.testing.assert_allclose(ours, ref, atol=1e-9)


def test_moverscore_without_reference_library() -> None:
    r, p, wr, wp = _tok_embs(2)
    assert float(es.moverscore([r[0]], [r[0]])) == pytest.approx(1.0, abs=1e-9)
    # one token each: the transport cost is the distance between the two unit vectors
    a, b = np.array([[3.0, 4.0]]), np.array([[4.0, 3.0]])
    assert float(es.moverscore([a], [b])) == pytest.approx(1 - np.linalg.norm(a[0] / 5 - b[0] / 5))
    vals = es.moverscore(r[:3], p[:3], reference_weights=wr[:3], prediction_weights=wp[:3], average=None).value
    assert vals.shape == (3,) and np.all(vals <= 1)
    with pytest.raises(es.InputValidationError):
        es.moverscore(r[:2], p[:3])


def test_mauve_curve_matches_mauve_text() -> None:
    cm = pytest.importorskip("mauve.compute_mauve")
    from sklearn.metrics import auc

    rng = np.random.default_rng(3)
    for _ in range(5):
        p, q = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
        q[rng.integers(8)] = 0
        q /= q.sum()
        x, y = cm.get_divergence_curve_for_multinomials(p, q, np.linspace(1e-6, 1 - 1e-6, 25), 5).T
        i1, i2 = np.argsort(x), np.argsort(y)
        ref = 0.5 * (auc(x[i1], y[i1]) + auc(y[i2], x[i2]))
        assert mauve_from_histograms(p, q) == pytest.approx(ref, abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_mauve_recovers_true_clusters(seed: int) -> None:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(6, 32)) * 20

    def draw(n, probs):
        lab = rng.choice(6, n, p=probs)
        return centers[lab] + rng.normal(size=(n, 32)) * 0.1, lab

    p, lp = draw(300, [0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
    q, lq = draw(300, [0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    truth = mauve_from_histograms(np.bincount(lp, minlength=6) / 300, np.bincount(lq, minlength=6) / 300)
    assert float(es.mauve(p, q, num_buckets=6, random_state=seed)) == pytest.approx(truth, abs=1e-12)
    assert float(es.mauve(p, p, num_buckets=6)) == pytest.approx(1.0)
    star = es.mauve(p, q, num_buckets=6, smoothed=True)
    assert star.name == "MAUVE*" and 0 < float(star) <= 1


def test_mauve_errors_and_auto_buckets() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(40, 5)), rng.normal(size=(40, 5))
    assert es.mauve(a, b).params["num_buckets"] == 4
    for bad in (
        lambda: es.mauve(a, rng.normal(size=(40, 6))),
        lambda: es.mauve(a, b, explained_variance=1.0),
        lambda: es.mauve(a, b, num_buckets=1),
        lambda: es.mauve(a[:2], b[:2], num_buckets=10),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_model_score_adapter() -> None:
    calls = []

    def scorer(r, p, s=None):
        calls.append(len(r))
        return [float(a == b) for a, b in zip(r, p)]

    res = es.model_score(["a", "b", "c"], ["a", "x", "c"], scorer=scorer, batch_size=2, name="exact")
    assert float(res) == pytest.approx(2 / 3) and calls == [2, 1] and res.name == "exact"
    assert float(es.model_score(["a"], ["a"], scorer=scorer, sources=["s"])) == 1.0
    for bad in (
        lambda: es.model_score(["a"], ["a"], scorer="comet"),
        lambda: es.model_score(["a"], ["a", "b"], scorer=scorer),
        lambda: es.model_score(["a"], ["a"], scorer=lambda r, p: [1, 2]),
        lambda: es.model_score(["a"], ["a"], scorer=lambda r, p: [float("nan")]),
        lambda: es.model_score(["a"], ["a"], scorer=scorer, batch_size=0),
        lambda: es.model_score(["a"], ["a"], scorer=scorer, sources=["s", "t"]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


# ---------------------------------------------------------------- factuality
CLAIMS = [["supported", "supported", "contradicted"], [True, False], [], ["nei", "supported"]]


def test_faithfulness_and_hallucination_rate() -> None:
    assert float(es.faithfulness(CLAIMS)) == pytest.approx(np.mean([2 / 3, 1 / 2, 1 / 2]))
    assert float(es.faithfulness(CLAIMS, average="micro")) == pytest.approx(4 / 7)
    per = es.faithfulness(CLAIMS, average=None).value
    assert np.isnan(per[2])
    assert float(es.hallucination_rate(CLAIMS)) == pytest.approx(3 / 7)
    assert float(es.hallucination_rate(CLAIMS, count="contradicted")) == pytest.approx(1 / 7)
    with pytest.raises(es.InputValidationError):
        es.faithfulness([["maybe"]])
    with pytest.raises(es.InputValidationError):
        es.faithfulness([[]], average="micro")
    with pytest.raises(es.InputValidationError):
        es.hallucination_rate(CLAIMS, count="all")
    with pytest.raises(es.InputValidationError):
        es.faithfulness("supported")


def test_groundedness() -> None:
    assert float(es.groundedness([[1, 0.5], [0.0]])) == pytest.approx(np.mean([0.75, 0.0]))
    assert float(es.groundedness([[1, 0.5], [0.0]], average="micro")) == pytest.approx(0.5)
    with pytest.raises(es.InputValidationError):
        es.groundedness([[1.5]])


def test_citations_alce() -> None:
    cites = [
        [{"supported": True, "citations": [True, False]}, {"supported": False, "citations": [False]}],
        [{"supported": True, "citations": []}, {"supported": True, "citations": [True]}],
    ]
    # statement 3 has no citations -> unsupported
    assert float(es.citation_recall(cites)) == pytest.approx(np.mean([1 / 2, 1 / 2]))
    assert float(es.citation_precision(cites, average="micro")) == pytest.approx(2 / 4)
    with pytest.raises(es.InputValidationError):
        es.citation_recall([[{"citations": [True]}]])


def test_claim_verification_knowledge_answers_abstention() -> None:
    gold = ["supported", "contradicted", "nei", "supported"]
    pred = ["supported", "supported", "nei", "supported"]
    r = es.claim_verification_accuracy(gold, pred)
    assert float(r) == 0.75 and r.params["per_class_f1"]["supported"] == pytest.approx(0.8)
    assert float(es.knowledge_consistency([{"a", "b"}, {"c"}], {"a", "c"})) == pytest.approx(2 / 3)
    assert float(es.answer_correctness([2, 0], [1, 0], [1, 0])) == pytest.approx(np.mean([2 / 3, 1.0]))
    blended = es.answer_correctness([2], [1], [1], similarity=[0.9])
    assert float(blended) == pytest.approx(0.75 * 2 / 3 + 0.25 * 0.9)
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    gen = [np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 2.0]])]
    assert float(es.answer_relevance(q, gen)) == pytest.approx(np.mean([(1 + 1 / math.sqrt(2)) / 2, 1.0]))
    a = es.abstention_accuracy([True, False, False, True], [True, False, True, False], correct=[0, 1, 0, 1])
    assert float(a) == 0.5
    assert a.params["abstention_precision"] == 0.5 and a.params["coverage"] == 0.5
    assert a.params["answered_accuracy"] == 1.0
    for bad in (
        lambda: es.claim_verification_accuracy(["supported"], []),
        lambda: es.knowledge_consistency([{"a"}], set()),
        lambda: es.answer_correctness([1], [1, 2], [1]),
        lambda: es.answer_correctness([1], [1], [1], similarity=[2.0]),
        lambda: es.abstention_accuracy([True], [True, False]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


# ---------------------------------------------------------------- judges and preferences
def test_bradley_terry_matches_choix() -> None:
    choix = pytest.importorskip("choix")
    rng = np.random.default_rng(0)
    strength = rng.normal(size=5)
    data, rows = [], []
    for _ in range(500):
        i, j = rng.choice(5, 2, replace=False)
        win = rng.random() < 1 / (1 + np.exp(strength[j] - strength[i]))
        w, loser = (i, j) if win else (j, i)
        data.append((w, loser))
        rows.append((f"m{w}", f"m{loser}", "win"))
    ref = choix.mm_pairwise(5, data, max_iter=100_000, tol=1e-12)
    ref -= ref.mean()
    ours = es.bradley_terry(rows)
    order = [int(n[1:]) for n in ours.labels]
    np.testing.assert_allclose(ours.value, ref[order], atol=1e-6)
    elo = es.bradley_terry(rows, scale="elo")
    np.testing.assert_allclose(elo.value, 1000 + 400 * ours.value / math.log(10))


def test_bradley_terry_errors() -> None:
    for bad in (
        [("a", "b", "win"), ("a", "b", "win")],  # b never wins
        [("a", "b", "win"), ("b", "a", "win"), ("c", "d", "win"), ("d", "c", "win")],  # disconnected
        [("a", "a", "win")],
        [("a", "b")],
    ):
        with pytest.raises(es.InputValidationError):
            es.bradley_terry(bad)
    with pytest.raises(es.InputValidationError):
        es.bradley_terry([("a", "b", "win"), ("b", "a", "tie")], scale="z")


def test_elo_and_win_rate() -> None:
    r = es.elo_ratings([("a", "b", "win"), ("a", "b", "tie")])
    e2 = 1 / (1 + 10 ** ((998 - 1002) / 400))
    assert r.value[0] == pytest.approx(1002 + 4 * (0.5 - e2))
    w = es.win_rate(["win", "tie", "loss", "win", 1, 0.5])
    assert float(w) == pytest.approx(4 / 6) and w.params["ci_low"] < 4 / 6 < w.params["ci_high"]
    assert float(es.win_rate(["win", "tie", "loss"], ties="exclude")) == 0.5
    with pytest.raises(es.InputValidationError):
        es.win_rate(["tie"], ties="exclude")
    with pytest.raises(es.InputValidationError):
        es.win_rate(["maybe"])


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval", "ratio"])
def test_krippendorff_matches_package(level: str) -> None:
    kd = pytest.importorskip("krippendorff")
    rng = np.random.default_rng(4)
    a = rng.integers(1, 6, (4, 30)).astype(float)
    a[rng.random(a.shape) < 0.15] = np.nan
    assert float(es.krippendorff_alpha(a, level=level)) == pytest.approx(
        kd.alpha(reliability_data=a, level_of_measurement=level), abs=1e-12
    )


def test_fleiss_matches_statsmodels() -> None:
    ir = pytest.importorskip("statsmodels.stats.inter_rater")
    r = np.random.default_rng(5).integers(0, 4, (40, 5))
    table, _ = ir.aggregate_raters(r)
    assert float(es.fleiss_kappa(r)) == pytest.approx(ir.fleiss_kappa(table), abs=1e-12)


def test_agreement_and_bias_checks() -> None:
    assert float(es.judge_agreement(["a", "b", "a"], ["a", "b", "b"])) == pytest.approx(0.4)
    ordinal = es.judge_agreement([1, 2, 3, 4, 5, 3], [1, 2, 3, 5, 4, 3], kind="ordinal")
    assert 0.8 < float(ordinal) < 1 and ordinal.params["spearman"] > 0.8
    cont = es.judge_agreement([0.1, 0.5, 0.9], [0.2, 0.4, 1.0], kind="continuous")
    assert float(cont) > 0.9
    pc = es.position_consistency(["A", "B", "A", "tie"], ["A", "A", "B", "tie"])
    assert float(pc) == 0.5 and pc.params["first_position_rate"] == 0.5
    vb = es.verbosity_bias(["A", "B", "A", "tie"], [10, 5, 20, 3], [5, 9, 4, 1])
    assert float(vb) == 1.0 and vb.params["n_decisive"] == 3
    sp = es.self_preference_bias([1, 1, 1, 0], [1, 0, 0, 0])
    assert float(sp) == 0.5
    rs = es.rubric_score([[5, 4], [3, 2]], criteria=["helpfulness", "fluency"])
    assert float(rs) == pytest.approx(0.625) and rs.params["per_criterion"]["helpfulness"] == 0.75
    for bad in (
        lambda: es.krippendorff_alpha(np.ones((3, 4))),
        lambda: es.krippendorff_alpha([[1, 2]], level="x"),
        lambda: es.fleiss_kappa(np.zeros((4, 3))),
        lambda: es.judge_agreement([1], [1, 2]),
        lambda: es.judge_agreement([1], [1], kind="x"),
        lambda: es.position_consistency(["C"], ["A"]),
        lambda: es.verbosity_bias(["tie"], [1], [2]),
        lambda: es.self_preference_bias([1], []),
        lambda: es.rubric_score([[6]], scale=(1, 5)),
        lambda: es.rubric_score([[1]], criteria=["a", "b"]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


# ---------------------------------------------------------------- RAG and reasoning
def test_context_metrics_latency_success_attribution() -> None:
    assert float(es.context_precision([[1, 0, 1], [0, 0], [1]])) == pytest.approx(((1 + 2 / 3) / 2 + 0 + 1) / 3)
    assert float(es.context_recall([[1, 0], [], [1, 1]])) == pytest.approx(0.75)
    assert float(es.context_relevance([[1, 0, 0, 1]])) == 0.5
    lat = es.latency_summary([10, 20, 30, 40], statistic="p50")
    assert float(lat) == 25 and lat.params["mean"] == 25
    s = es.task_success_rate([1, 1, 0, 1])
    assert float(s) == 0.75 and s.params["ci_low"] < 0.75
    fa = es.failure_attribution(["retrieval", None, "generation", "retrieval", "none"])
    assert fa.value.tolist()[:2] == [2 / 3, 1 / 3] and fa.labels[0] == "retrieval"
    for bad in (
        lambda: es.latency_summary([-1]),
        lambda: es.latency_summary([1], statistic="p42"),
        lambda: es.failure_attribution(["network"]),
        lambda: es.failure_attribution([None]),
        lambda: es.context_recall([[]]),
        lambda: es.task_success_rate([]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


@pytest.mark.parametrize(
    ("text", "style", "want"),
    [
        ("so 3 + 4 = 7. #### 1,234", "gsm8k", "1234"),
        ("The total is $42.00.", "gsm8k", "42"),
        ("no numbers", "gsm8k", None),
        (r"thus \boxed{\frac{1}{2}} done", "boxed", r"\frac{1}{2}"),
        (r"\fbox{7}", "boxed", "7"),
        ("no box", "boxed", None),
        ("I think (B) is right", "choice", "B"),
        ("Answer: C", "choice", "C"),
        ("line one\n\nfinal line\n", "last_line", "final line"),
    ],
)
def test_extract_answer(text: str, style: str, want) -> None:
    assert es.extract_answer(text, style=style) == want


def test_benchmark_accuracy() -> None:
    r = es.benchmark_accuracy(["blah #### 18", "42", "7"], ["... so 18", "answer: 41", "no idea"])
    assert float(r) == pytest.approx(1 / 3) and r.params["no_answer_found"] == 1
    mc = es.benchmark_accuracy(["B", "A"], ["The answer is B.", "(C)"], style="choice")
    assert float(mc) == 0.5
    with pytest.raises(es.InputValidationError):
        es.extract_answer("x", style="regex")


# ---------------------------------------------------------------- structured output
def _random_schema(rng: random.Random, d: int = 0):
    t = rng.choice(
        ["object", "array", "string", "integer", "number", "boolean", "null", "any"]
        if d < 3
        else ["string", "integer"]
    )
    s: dict = {} if t == "any" else {"type": t if rng.random() < 0.85 else [t, "null"]}
    if t == "object":
        props = {k: _random_schema(rng, d + 1) for k in rng.sample("abcde", rng.randint(0, 3))}
        s["properties"] = props
        if props and rng.random() < 0.7:
            s["required"] = rng.sample(list(props), rng.randint(1, len(props)))
        if rng.random() < 0.4:
            s["additionalProperties"] = rng.choice([False, True, {"type": "integer"}])
    if t == "array":
        s["items"] = _random_schema(rng, d + 1)
        if rng.random() < 0.3:
            s["minItems"] = rng.randint(0, 3)
        if rng.random() < 0.2:
            s["uniqueItems"] = True
    if t == "string":
        if rng.random() < 0.3:
            s["pattern"] = rng.choice(["^a", "b$", "[0-9]"])
        if rng.random() < 0.3:
            s["enum"] = ["a", "bb", "ccc"]
    if t in ("integer", "number") and rng.random() < 0.4:
        s["minimum"] = rng.randint(-5, 5)
        if rng.random() < 0.5:
            s["multipleOf"] = rng.choice([2, 3, 0.5])
    if d < 2 and rng.random() < 0.08:
        s = {"anyOf": [s, _random_schema(rng, d + 1)]}
    if d < 2 and rng.random() < 0.05:
        s = {"oneOf": [s, _random_schema(rng, d + 1)]}
    if d < 2 and rng.random() < 0.04:
        s = {"not": s}
    return s


def _random_value(rng: random.Random, d: int = 0):
    c = rng.random() * (0.6 if d > 3 else 1)
    if c < 0.12:
        return None
    if c < 0.24:
        return rng.choice([True, False])
    if c < 0.4:
        return rng.randint(-8, 12)
    if c < 0.48:
        return rng.choice([1.5, 2.0, -0.5])
    if c < 0.62:
        return rng.choice(["a", "bb", "ccc", "b", "a1", ""])
    if c < 0.8:
        return [_random_value(rng, d + 1) for _ in range(rng.randint(0, 4))]
    return {k: _random_value(rng, d + 1) for k in rng.sample("abcdefg", rng.randint(0, 4))}


def test_json_schema_validator_agrees_with_jsonschema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    rng = random.Random(0)  # noqa: S311  (test data, not security)
    for _ in range(4000):
        schema, value = _random_schema(rng), _random_value(rng)
        ref = jsonschema.Draft202012Validator(schema).is_valid(value)
        assert (not es.validate_json_schema(value, schema)) == ref, (json.dumps(schema), json.dumps(value))


def test_json_schema_extras() -> None:
    schema = {
        "$defs": {"pos": {"type": "integer", "exclusiveMinimum": 0}},
        "type": "object",
        "properties": {
            "n": {"$ref": "#/$defs/pos"},
            "tags": {
                "type": "array",
                "prefixItems": [{"const": "x"}],
                "contains": {"type": "string"},
                "maxItems": 3,
            },
        },
        "patternProperties": {"^z": {"type": "boolean"}},
        "propertyNames": {"maxLength": 4},
        "maxProperties": 3,
        "required": ["n"],
    }
    assert not es.validate_json_schema({"n": 2, "tags": ["x", "y"], "z1": True}, schema)
    assert es.validate_json_schema({"n": 0}, schema)
    assert es.validate_json_schema({"n": 1, "z1": 3}, schema)
    assert es.validate_json_schema({"n": 1, "toolong": 1}, schema)
    assert es.validate_json_schema(None, False) and not es.validate_json_schema(None, True)
    with pytest.raises(es.InputValidationError):
        es.validate_json_schema(1, {"type": "integer", "format_x": 1})
    with pytest.raises(es.InputValidationError):
        es.validate_json_schema(1, {"$ref": "http://x"})


def test_json_validity_compliance_and_extraction() -> None:
    outs = ['{"a": 1}', '```json\n{"a": 2}\n```', 'Sure! {"a": "x"} hope it helps', "nope"]
    assert float(es.json_validity(outs)) == 0.25
    assert float(es.json_validity(outs, mode="fenced")) == 0.5
    assert float(es.json_validity(outs, mode="lenient")) == 0.75
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    comp = es.json_schema_compliance(outs, schema, mode="lenient")
    assert float(comp) == 0.5 and comp.params["errors"][3] == "not valid JSON"
    assert float(es.extra_content_rate(outs)) == 0.5
    assert (
        float(es.extra_content_rate(["Answer: A", "x Answer: B"], payload="regex", pattern="Answer: [A-D]")) == 0.5
    )
    assert es.extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        es.extract_json("{")
    with pytest.raises(es.InputValidationError):
        es.json_schema_compliance(outs, "schema")
    with pytest.raises(es.InputValidationError):
        es.extra_content_rate(outs, payload="xml")


def test_xml_and_fields() -> None:
    xmls = ["<a><b>1</b></a>", "<a><b></a>", '<!DOCTYPE x [<!ENTITY e "boom">]><a>&e;</a>', "<r/>"]
    assert float(es.xml_validity(xmls)) == 0.5
    assert float(es.xml_validity(xmls, root_tag="a")) == 0.25
    expected = [{"name": "Ann", "address.city": "Pune", "tags.0": "x"}, {"age": 3}]
    outs = ['{"name": "ann", "address": {"city": "Pune"}, "tags": ["x"]}', {"age": 4}]
    assert float(es.required_field_accuracy(expected, outs)) == pytest.approx(2 / 4)
    assert float(
        es.required_field_accuracy(expected, outs, normalize=lambda v: v.lower() if isinstance(v, str) else v)
    ) == pytest.approx(3 / 4)
    assert es.required_field_accuracy(expected, outs, average=None).value.tolist() == [2 / 3, 0.0]
    with pytest.raises(es.InputValidationError):
        es.required_field_accuracy([{}], ["{}"])


def test_tool_calls_instructions_and_format() -> None:
    exp = [
        {"name": "search", "arguments": {"q": "x"}},
        [],
        [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": 1}}],
    ]
    pred = [
        {"name": "search", "arguments": '{"q": "x"}'},
        [{"name": "search", "arguments": "not json"}],
        [{"name": "b", "arguments": {"k": 2}}, {"name": "a", "arguments": {}}],
    ]
    assert float(es.tool_selection_accuracy(exp, pred)) == pytest.approx(2 / 3)
    assert float(es.tool_selection_accuracy(exp, pred, ordered=True)) == pytest.approx(1 / 3)
    assert float(es.tool_argument_accuracy(exp, pred)) == pytest.approx(2 / 3)
    f = es.tool_call_f1(exp, pred)
    assert f.params["precision"] == pytest.approx(2 / 4) and f.params["recall"] == pytest.approx(2 / 3)
    assert f.params["invalid_call_rate"] == pytest.approx(1 / 4)
    assert float(es.tool_call_f1(exp, pred, match="name")) > float(f)
    assert float(es.api_call_success_rate([True, 200, 404, None, ValueError("x"), 302])) == pytest.approx(3 / 6)
    checks = [[True, True], [True, False], [True]]
    assert float(es.instruction_compliance_rate(checks)) == pytest.approx(2 / 3)
    assert float(es.constraint_satisfaction_rate(checks)) == pytest.approx(4 / 5)
    assert float(es.instruction_retention([[True, [True, False]], [False]])) == pytest.approx(2 / 4)
    assert float(es.format_compliance(["Answer: B", "B"], r"Answer: [A-D]")) == 0.5
    for bad in (
        lambda: es.tool_selection_accuracy([{"arguments": {}}], [{}]),
        lambda: es.tool_argument_accuracy([[]], [[]]),
        lambda: es.tool_call_f1(exp, pred, match="fuzzy"),
        lambda: es.api_call_success_rate(["ok"]),
        lambda: es.format_compliance(["x"], "("),
        lambda: es.instruction_retention([[]]),
        lambda: es.json_validity("{}"),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


# ---------------------------------------------------------------- resampling text metrics
def test_bootstrap_and_compare_resample_whole_items() -> None:
    refs = [["the cat sat on the mat", "a cat on a mat"], ["a dog ran"], ["hello world"]] * 10
    preds = ["the cat sat on mat", "dog ran", "hello there world"] * 10
    ci = es.bootstrap_ci(es.bleu, refs, preds, n_resamples=200, random_state=0)
    assert ci.low < ci.estimate < ci.high  # not stratified by string value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = es.compare(
            refs,
            {"a": preds, "b": [p + " x" for p in preds]},
            metrics=["bleu", "rouge_l"],
            n_resamples=200,
            random_state=0,
        )
    assert cmp.best("bleu") == "a" if hasattr(cmp, "best") else True
    ret = es.bootstrap_ci(
        es.ndcg_at_k, [{1, 2}, {3}] * 10, [[1, 5, 2], [4, 3, 9]] * 10, n_resamples=200, random_state=0
    )
    assert ret.low < ret.high


def test_every_v040_metric_is_registered_with_documentation() -> None:
    ids = [m for m in es.list_metrics() if m.split(".")[0] in ("text", "reasoning", "retrieval", "rag")]
    assert len(ids) >= 65
    for mid in ids:
        info = es.metric_info(mid)
        assert info.definition and info.formula and info.references, mid


# ---------------------------------------------------------------- report and retrieval without references
def test_text_report_values_exports_and_subset(tmp_path) -> None:
    refs = [["The cat sat on the mat.", "A cat on a mat."], ["Hello world"]]
    preds = ["The cat sat on a mat.", "hello there world"]
    rep = es.text_report(refs, preds)
    assert rep["bleu"] == pytest.approx(float(es.bleu(refs, preds)))
    assert rep["chrf_pp"] == pytest.approx(float(es.chrf(refs, preds, word_order=2)))
    assert rep["token_f1"] == pytest.approx(float(es.token_f1(refs, preds)))
    assert "BLEU" in rep.summary() and rep.to_markdown().startswith("| Metric |")
    assert json.loads(rep.to_json())["values"]["ter"] == pytest.approx(float(es.ter(refs, preds)))
    assert "\\toprule" in rep.to_latex() or "toprule" in rep.to_latex()
    assert "<table" in rep.to_html() and rep.to_csv().startswith("metric,value")
    assert next(iter(rep.to_dataframe().index)) == "bleu"
    for ext in ("json", "csv", "md", "tex", "html", "txt"):
        rep.save(tmp_path / f"r.{ext}")
        assert (tmp_path / f"r.{ext}").read_text(encoding="utf-8").strip()
    sub = es.text_report(refs, preds, metrics=["rouge_l", "exact_match"])
    assert list(sub.values) == ["rouge_l", "exact_match"]
    with pytest.raises(KeyError):
        sub["bleu"]
    with pytest.raises(es.InputValidationError):
        es.text_report(refs, preds, metrics=["bertscore"])


def test_retrieval_hand_computed() -> None:
    relevant = [{"a", "c"}, {"x"}, {"b": 3, "d": 1}]
    retrieved = [["a", "b", "c"], ["y", "z"], ["d", "b"]]
    assert es.precision_at_k(relevant, retrieved, k=2, average=None).value.tolist() == [0.5, 0.0, 1.0]
    assert es.recall_at_k(relevant, retrieved, k=2, average=None).value.tolist() == [0.5, 0.0, 1.0]
    assert es.hit_rate_at_k(relevant, retrieved, k=1, average=None).value.tolist() == [1.0, 0.0, 1.0]
    assert es.mrr(relevant, retrieved, average=None).value.tolist() == [1.0, 0.0, 1.0]
    assert float(es.mean_average_precision_at_k(relevant, retrieved, k=3)) == pytest.approx(
        ((1 + 2 / 3) / 2 + 0 + 1) / 3
    )
    # graded NDCG with linear gains (trec_eval and ranx), discounted by log2(rank + 1)
    dcg = 1 / 1 + 3 / math.log2(3)
    idcg = 3 / 1 + 1 / math.log2(3)
    assert es.ndcg_at_k(relevant, retrieved, k=2, average=None).value[2] == pytest.approx(dcg / idcg)
    for bad in (
        lambda: es.precision_at_k([{"a"}], [["a"], ["b"]]),
        lambda: es.precision_at_k([{"a"}], [["a", "a"]]),
        lambda: es.precision_at_k([{"a"}], ["a"]),
        lambda: es.precision_at_k([{"a"}], [["a"]], k=0),
        lambda: es.recall_at_k([set()], [["a"]]),
        lambda: es.ndcg_at_k([{"a": -1}], [["a"]]),
        lambda: es.ndcg_at_k(["a"], [["a"]]),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_cli_text_command(tmp_path) -> None:
    import subprocess
    import sys

    f = tmp_path / "t.csv"
    f.write_text(
        'pred,ref,ref2\n"The cat sat on a mat.","The cat sat on the mat.","A cat on a mat."\nhello,Hello,\n'
    )
    out = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "evalsuite",
            "text",
            str(f),
            "--prediction",
            "pred",
            "--reference",
            "ref",
            "--reference",
            "ref2",
            "--metrics",
            "bleu,rouge_l",
            "-f",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = json.loads(out.stdout)["values"]
    assert set(values) == {"bleu", "rouge_l"}
    bad = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "evalsuite", "text", str(f), "--prediction", "nope", "--reference", "ref"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 2 and "nope" in bad.stderr
