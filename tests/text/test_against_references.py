"""Text, retrieval and reasoning metrics against their reference implementations on random data.

sacreBLEU (BLEU, sentence BLEU, chrF, chrF++, TER), Google's rouge-score (ROUGE-1/2/L/Lsum) and ranx
(Precision@k, Recall@k, Hit rate@k, MRR, MAP, NDCG). Every number must agree to floating-point rounding.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

import evalsuite as es

VOCAB = [
    "the",
    "a",
    "cat",
    "dog",
    "sat",
    "on",
    "mat",
    "quickly",
    "2.5",
    "3,000",
    "well-known",
    "state-of-the-art",
    "&",
    "&amp;",
    "<b>",
    "(hi)",
    "don't",
    "U.S.",
    "e-mail",
    "1990-2000",
    "Hello",
    "WORLD",
    "naïve",
    "café:",
    ";",
    "!",
    "?",
    '"',
    "'",
    "run",
    "running",
    "runs",
    "ran",
    "model",
    "models",
    "data",
    "--",
    "x",
]


def _sentence(rng: np.random.Generator, lo: int = 0, hi: int = 18) -> str:
    return " ".join(rng.choice(VOCAB, int(rng.integers(lo, hi))))


def _corpus(seed: int, n: int = 40, n_refs: int = 1, lo: int = 0):
    rng = np.random.default_rng(seed)
    preds, refs = [], []
    for _ in range(n):
        base = _sentence(rng, max(lo, 1))
        words = base.split()
        # predictions are noisy copies so that n-gram matches exist at every order
        keep = [w for w in words if rng.random() > 0.25] + list(rng.choice(VOCAB, int(rng.integers(0, 4))))
        preds.append(" ".join(keep))
        refs.append([base] + [_sentence(rng, lo) for _ in range(n_refs - 1)])
    return preds, refs


sacrebleu = pytest.importorskip("sacrebleu")


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_refs", [1, 3])
@pytest.mark.parametrize(
    "opts",
    [
        {},
        {"smooth": "none"},
        {"smooth": "floor", "smooth_value": 0.1},
        {"smooth": "add-k", "smooth_value": 1},
        {"lowercase": True},
        {"tokenize": "none"},
        {"effective_order": True},
        {"max_order": 2},
    ],
)
def test_corpus_bleu_matches_sacrebleu(seed: int, n_refs: int, opts: dict) -> None:
    preds, refs = _corpus(seed, n_refs=n_refs)
    ref_streams = [[r[i] for r in refs] for i in range(n_refs)]
    sb = dict(opts)
    if "smooth" in sb:
        sb["smooth_method"] = sb.pop("smooth")
    if "max_order" in sb:
        sb["max_ngram_order"] = sb.pop("max_order")
    expected = sacrebleu.metrics.BLEU(**sb).corpus_score(preds, ref_streams).score
    assert float(es.bleu(refs, preds, **opts)) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", range(3))
def test_sentence_bleu_matches_sacrebleu(seed: int) -> None:
    preds, refs = _corpus(seed, n_refs=2)
    per = es.sentence_bleu(refs, preds, average=None).value
    for p, rs, ours in zip(preds, refs, per):
        assert ours == pytest.approx(sacrebleu.sentence_bleu(p, rs).score, abs=1e-9)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_refs", [1, 2])
@pytest.mark.parametrize(
    "opts",
    [{}, {"word_order": 2}, {"lowercase": True}, {"whitespace": True}, {"eps_smoothing": True}, {"beta": 1}],
)
def test_chrf_matches_sacrebleu(seed: int, n_refs: int, opts: dict) -> None:
    preds, refs = _corpus(seed, n_refs=n_refs)
    streams = [[r[i] for r in refs] for i in range(n_refs)]
    expected = sacrebleu.metrics.CHRF(**opts).corpus_score(preds, streams).score
    assert float(es.chrf(refs, preds, **opts)) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_refs", [1, 2])
@pytest.mark.parametrize("case_sensitive", [False, True])
def test_ter_matches_sacrebleu(seed: int, n_refs: int, case_sensitive: bool) -> None:
    preds, refs = _corpus(seed, n=25, n_refs=n_refs)
    # add reorderings so that block shifts matter
    rng = np.random.default_rng(seed)
    preds = [" ".join(np.roll(p.split(), int(rng.integers(0, 4)))) if p else p for p in preds]
    streams = [[r[i] for r in refs] for i in range(n_refs)]
    expected = sacrebleu.metrics.TER(case_sensitive=case_sensitive).corpus_score(preds, streams).score
    assert float(es.ter(refs, preds, case_sensitive=case_sensitive)) == pytest.approx(expected, abs=1e-9)


rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("n_refs", [1, 3])
@pytest.mark.parametrize("stem", [False, True])
def test_rouge_matches_rouge_score(seed: int, n_refs: int, stem: bool) -> None:
    preds, refs = _corpus(seed, n_refs=n_refs)
    rng = np.random.default_rng(seed + 100)
    # multi-sentence texts (one sentence per line) for ROUGE-Lsum
    preds = [p.replace(" . ", "\n") + ("\n" + _sentence(rng) if rng.random() < 0.5 else "") for p in preds]
    refs = [[r + ("\n" + _sentence(rng) if rng.random() < 0.5 else "") for r in rs] for rs in refs]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=stem)
    stemmer = None
    if stem:
        from rouge_score.tokenizers import DefaultTokenizer

        stemmer = DefaultTokenizer(use_stemmer=True)._stemmer.stem
    expected = [scorer.score_multi(rs, p) for p, rs in zip(preds, refs)]
    for fn, key in [
        (es.rouge_1, "rouge1"),
        (es.rouge_2, "rouge2"),
        (es.rouge_l, "rougeL"),
        (es.rouge_lsum, "rougeLsum"),
    ]:
        for measure in ("fmeasure", "precision", "recall"):
            ours = fn(refs, preds, measure=measure, stemmer=stemmer, average=None).value
            theirs = [getattr(e[key], measure) for e in expected]
            np.testing.assert_allclose(ours, theirs, atol=1e-12, err_msg=f"{key} {measure}")


def _runs(seed: int, n_queries: int = 30, graded: bool = False):
    rng = np.random.default_rng(seed)
    relevant, retrieved = [], []
    for _ in range(n_queries):
        docs = [f"d{i}" for i in range(40)]
        n_rel = int(rng.integers(1, 8))
        rel_docs = list(rng.choice(docs, n_rel, replace=False))
        relevant.append({d: int(rng.integers(1, 4)) for d in rel_docs} if graded else set(rel_docs))
        retrieved.append(list(rng.permutation(docs)[: int(rng.integers(0, 25))]))
    return relevant, retrieved


def _ranx(relevant, retrieved, metric: str) -> float:
    ranx = pytest.importorskip("ranx")  # not available on every Python version; skip only these tests
    qrels = ranx.Qrels(
        {
            f"q{i}": ({d: int(g) for d, g in r.items()} if isinstance(r, dict) else {d: 1 for d in r})
            for i, r in enumerate(relevant)
        }
    )
    run = ranx.Run(
        {
            f"q{i}": ({d: float(len(r) - j) for j, d in enumerate(r)} or {"__none__": 0.0})
            for i, r in enumerate(retrieved)
        }
    )
    return float(ranx.evaluate(qrels, run, metric))


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("k", [1, 5, 10])
def test_retrieval_matches_ranx(seed: int, k: int) -> None:
    relevant, retrieved = _runs(seed)
    assert float(es.precision_at_k(relevant, retrieved, k=k)) == pytest.approx(
        _ranx(relevant, retrieved, f"precision@{k}")
    )
    assert float(es.recall_at_k(relevant, retrieved, k=k)) == pytest.approx(
        _ranx(relevant, retrieved, f"recall@{k}")
    )
    assert float(es.hit_rate_at_k(relevant, retrieved, k=k)) == pytest.approx(
        _ranx(relevant, retrieved, f"hit_rate@{k}")
    )
    assert float(es.mrr(relevant, retrieved, k=k)) == pytest.approx(_ranx(relevant, retrieved, f"mrr@{k}"))
    assert float(es.mean_average_precision_at_k(relevant, retrieved, k=k)) == pytest.approx(
        _ranx(relevant, retrieved, f"map@{k}")
    )
    assert float(es.mrr(relevant, retrieved)) == pytest.approx(_ranx(relevant, retrieved, "mrr"))
    assert float(es.mean_average_precision_at_k(relevant, retrieved)) == pytest.approx(
        _ranx(relevant, retrieved, "map")
    )


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("k", [3, 10])
def test_ndcg_matches_ranx_with_graded_relevance(seed: int, k: int) -> None:
    relevant, retrieved = _runs(seed, graded=True)
    assert float(es.ndcg_at_k(relevant, retrieved, k=k)) == pytest.approx(_ranx(relevant, retrieved, f"ndcg@{k}"))
    assert float(es.ndcg_at_k(relevant, retrieved, k=k, gains="exponential")) == pytest.approx(
        _ranx(relevant, retrieved, f"ndcg_burges@{k}")
    )


def test_pass_at_k_matches_the_combinatorial_definition() -> None:
    rng = np.random.default_rng(0)
    n = rng.integers(5, 30, 50)
    c = np.array([rng.integers(0, m + 1) for m in n])
    for k in (1, 3, 5):
        expected = np.mean([1 - math.comb(int(a) - int(b), k) / math.comb(int(a), k) for a, b in zip(n, c)])
        assert float(es.pass_at_k(n, c, k=k)) == pytest.approx(expected, rel=1e-12)
    assert float(es.pass_at_k(10, [0, 10, 5], k=1)) == pytest.approx((0 + 1 + 0.5) / 3)


def test_self_bleu_is_mean_leave_one_out_sentence_bleu() -> None:
    preds, _ = _corpus(7, n=6)
    expected = np.mean(
        [sacrebleu.sentence_bleu(p, [q for j, q in enumerate(preds) if j != i]).score for i, p in enumerate(preds)]
    )
    assert float(es.self_bleu(preds)) == pytest.approx(expected, abs=1e-9)


def test_distinct_n_counts() -> None:
    preds = ["a b c a", "a b d"]
    assert float(es.distinct_n(preds, n=1)) == pytest.approx(4 / 7)
    assert float(es.distinct_n(preds, n=2)) == pytest.approx(4 / 5)  # ab bc ca | ab bd
    pairs = list(combinations(range(3), 2))
    assert pairs  # sanity for combinations import
