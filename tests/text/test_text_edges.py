"""Question answering (against the official SQuAD v2 scoring logic), sampling-based reasoning, generation
options and input validation for the text metrics."""

from __future__ import annotations

import collections
import math
import re
import string

import numpy as np
import pytest

import evalsuite as es


# ---------------------------------------------------------------- SQuAD v2 official logic (re-stated)
def _squad_normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _squad_f1(gold: str, pred: str) -> float:
    g, p = _squad_normalize(gold).split(), _squad_normalize(pred).split()
    if not g or not p:
        return float(g == p)
    common = collections.Counter(g) & collections.Counter(p)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


CASES = [
    ("The Eiffel Tower", "eiffel tower!"),
    ("Paris, France", "in Paris"),
    ("an apple a day", "Apple"),
    ("", ""),
    ("1990", "1991"),
    ("New York City", "the city of New York"),
]


def test_exact_match_and_token_f1_match_squad() -> None:
    golds, preds = [g for g, _ in CASES], [p for _, p in CASES]
    em = es.exact_match(golds, preds, average=None).value
    f1 = es.token_f1(golds, preds, average=None).value
    np.testing.assert_allclose(em, [float(_squad_normalize(g) == _squad_normalize(p)) for g, p in CASES])
    np.testing.assert_allclose(f1, [_squad_f1(g, p) for g, p in CASES])
    # several references: best one counts
    assert float(es.exact_match([["Paris", "Paris, France"]], ["paris france"])) == 1.0
    assert float(es.token_f1([["x y", "Paris France"]], ["paris"])) == pytest.approx(2 / 3)


def test_qa_normalisation_options() -> None:
    assert float(es.exact_match(["Paris"], ["paris"], normalize=False)) == 0.0
    assert float(es.exact_match(["Paris"], [" PARIS "], normalize=str.strip)) == 0.0
    assert float(es.exact_match(["Paris"], [" PARIS "], normalize=lambda s: s.strip().lower())) == 1.0
    assert es.exact_match(["a"], ["a"]).params["normalize"] == "squad"
    assert es.token_f1(["a"], ["a"], normalize=False).params["normalize"] == "none"
    assert es.normalize_answer("The  Quick, brown fox!") == "quick brown fox"
    with pytest.raises(es.InputValidationError):
        es.exact_match(["a"], ["a"], normalize="squad")


@pytest.mark.parametrize(
    "bad",
    [
        lambda: es.exact_match("a", ["a"]),
        lambda: es.exact_match(["a"], "a"),
        lambda: es.exact_match([], []),
        lambda: es.exact_match(["a", "b"], ["a"]),
        lambda: es.exact_match([1], ["a"]),
        lambda: es.exact_match(["a"], [3]),
        lambda: es.exact_match([[]], ["a"]),
        lambda: es.exact_match([None], ["a"]),
        lambda: es.token_f1(None, ["a"]),
    ],
)
def test_text_input_validation(bad) -> None:
    with pytest.raises(es.InputValidationError):
        bad()


def test_numpy_inputs_are_accepted() -> None:
    assert float(es.exact_match(np.array(["a", "b"]), np.array(["a", "c"]))) == 0.5


# ---------------------------------------------------------------- pass@k and majority vote
def test_pass_at_k_and_errors() -> None:
    assert float(es.pass_at_k([10, 10], [0, 10], k=1)) == pytest.approx(0.5)
    assert float(es.pass_at_k(5, [1], k=1)) == pytest.approx(0.2)
    assert float(es.pass_at_k(5, [1], k=5)) == pytest.approx(1.0)
    assert float(es.pass_at_k(10, [3], k=2)) == pytest.approx(1 - math.comb(7, 2) / math.comb(10, 2))
    assert es.pass_at_k([4, 4], [1, 2], k=2, average=None).value.shape == (2,)
    for bad in (
        lambda: es.pass_at_k([5, 5], [1], k=1),
        lambda: es.pass_at_k([5.0], [1.0], k=1),
        lambda: es.pass_at_k([5], [6], k=1),
        lambda: es.pass_at_k([2], [1], k=3),
        lambda: es.pass_at_k([5], [1], k=0),
        lambda: es.pass_at_k([5], [1], k=True),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_majority_vote_accuracy() -> None:
    samples = [["42", "41", "42"], ["7", "8"], ["The cat", "cat", "dog"]]
    assert float(es.majority_vote_accuracy(["42", "8", "cat"], samples)) == pytest.approx(2 / 3)
    assert es.majority_vote_accuracy(["42", "8", "cat"], samples, average=None).value.tolist() == [1, 0, 1]
    with pytest.raises(es.InputValidationError):
        es.majority_vote_accuracy(["a"], [[]])
    with pytest.raises(es.InputValidationError):
        es.majority_vote_accuracy(["a"], ["a"])


# ---------------------------------------------------------------- generation options
def test_bleu_options_and_errors() -> None:
    refs, preds = ["The cat sat on the mat."], ["the cat sat on a mat."]
    assert float(es.bleu(refs, preds, lowercase=True)) > float(es.bleu(refs, preds))
    assert float(es.bleu(refs, preds, tokenize="none")) >= 0
    assert float(es.bleu(refs, preds, tokenize=lambda s: s.lower().split())) > 0
    assert float(es.bleu(refs, preds, smooth="floor")) > 0 and float(es.bleu(refs, preds, smooth="add-k")) > 0
    assert float(es.bleu(["a b"], ["c d"])) == 0.0
    assert float(es.bleu(["a b c d e"], ["a b"], effective_order=True)) > 0
    assert es.sentence_bleu(refs * 2, preds * 2, average=None).value.shape == (2,)
    for bad in (
        lambda: es.bleu(refs, preds, smooth="laplace"),
        lambda: es.bleu(refs, preds, tokenize="intl"),
        lambda: es.bleu(refs, preds, max_order=0),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_rouge_measures_and_errors() -> None:
    r = es.rouge_l(["a b c d"], ["a b x"], measure="precision")
    assert float(r) == pytest.approx(2 / 3)
    assert float(es.rouge_l(["a b c d"], ["a b x"], measure="recall")) == pytest.approx(0.5)
    with pytest.raises(es.InputValidationError):
        es.rouge_1(["a"], ["a"], measure="auc")
    with pytest.raises(es.InputValidationError):
        es.rouge_1(["a"], ["a"], stemmer="porter")


def test_likelihood_metrics() -> None:
    lp = [np.log([0.5, 0.25]), np.log([0.5])]
    assert float(es.cross_entropy(lp)) == pytest.approx(-(np.log(0.5) * 2 + np.log(0.25)) / 3)
    assert float(es.cross_entropy(lp, base=2)) == pytest.approx(4 / 3)
    assert float(es.perplexity(lp)) == pytest.approx(2 ** (4 / 3))
    assert float(es.perplexity(lp, average="sequences")) == pytest.approx(np.mean([2**1.5, 2.0]))
    assert float(es.perplexity(np.log([0.5, 0.5]))) == pytest.approx(2.0)
    assert float(es.perplexity([-0.5, -1.0])) == pytest.approx(math.exp(0.75))
    for bad in (
        lambda: es.perplexity([]),
        lambda: es.perplexity([[0.5]]),
        lambda: es.perplexity([[]]),
        lambda: es.perplexity(lp, average="median"),
        lambda: es.cross_entropy(lp, base=1),
        lambda: es.perplexity("abc"),
    ):
        with pytest.raises(es.InputValidationError):
            bad()


def test_diversity_metrics() -> None:
    assert float(es.distinct_n(["a b", "A b"], lowercase=True)) == pytest.approx(2 / 4)
    with pytest.raises(es.InputValidationError):
        es.distinct_n(["a"], n=2)
    with pytest.raises(es.InputValidationError):
        es.self_bleu(["only one"])
