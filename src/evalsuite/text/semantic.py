"""Embedding-based text metrics: BERTScore, embedding similarity and distance, MoverScore, MAUVE, and an
adapter for learned metrics (COMET, BLEURT, BARTScore, AlignScore, ...).

EvalSuite does not download or run language models. These functions take the embeddings or scores your own
model produces, so any encoder can be used and results stay reproducible:

- ``bertscore`` takes contextual token embeddings (one ``(tokens, dim)`` array per text);
- ``embedding_similarity`` takes one sentence embedding per text;
- ``moverscore`` takes token embeddings and optional IDF weights;
- ``mauve`` takes one feature vector per text for the reference and the generated sets;
- ``model_score`` wraps any scorer callable (a COMET or BLEURT model, a judge) so its scores get the same
  intervals, comparisons and reports as every other metric.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import per_item

__all__ = ["bertscore", "embedding_similarity", "mauve", "model_score", "moverscore"]

_C = "text"
_REF_BERTSCORE = (
    "Zhang T, Kishore V, Wu F, Weinberger KQ, Artzi Y. BERTScore: evaluating text generation with BERT. "
    "ICLR. 2020."
)
_REF_MOVER = (
    "Zhao W, Peyrard M, Liu F, Gao Y, Meyer CM, Eger S. MoverScore: text generation evaluating with "
    "contextualized embeddings and earth mover distance. EMNLP. 2019:563-578."
)
_REF_MAUVE = (
    "Pillutla K, Swayamdipta S, Zellers R, Thickstun J, Welleck S, Choi Y, Harchaoui Z. MAUVE: measuring the "
    "gap between neural text and human text using divergence frontiers. NeurIPS. 2021."
)
_REF_MAUVE2 = (
    "Pillutla K, Liu L, Thickstun J, Welleck S, Swayamdipta S, Zellers R, Oh S, Choi Y, Harchaoui Z. MAUVE "
    "scores for generative models: theory and practice. JMLR. 2023;24(356):1-92."
)
_REF_COMET = "Rei R, Stewart C, Farinha AC, Lavie A. COMET: a neural framework for MT evaluation. EMNLP. 2020."
_REF_BLEURT = "Sellam T, Das D, Parikh AP. BLEURT: learning robust metrics for text generation. ACL. 2020."


# ---------------------------------------------------------------- input helpers
def _matrices(x: Any, name: str) -> list[NDArray[np.float64]]:
    if x is None or isinstance(x, (str, bytes)):
        raise InputValidationError(f"{name} must be a list of 2-D token-embedding arrays (tokens × dim).")
    items = list(x) if not (isinstance(x, np.ndarray) and x.ndim == 2) else [x]
    if not items:
        raise InputValidationError(f"{name} is empty.")
    out = []
    dim = None
    for i, m in enumerate(items):
        a = np.asarray(m, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] == 0:
            raise InputValidationError(f"{name}[{i}] must be a non-empty 2-D array (tokens × dim); got {a.shape}.")
        if not np.all(np.isfinite(a)):
            raise InputValidationError(f"{name}[{i}] contains NaN or infinite values.")
        if dim is None:
            dim = a.shape[1]
        elif a.shape[1] != dim:
            raise InputValidationError(f"{name}[{i}] has dimension {a.shape[1]}; expected {dim} like the others.")
        out.append(a)
    return out


def _vectors(x: Any, name: str) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise InputValidationError(f"{name} must be a 2-D array with one embedding per row; got {a.shape}.")
    if not np.all(np.isfinite(a)):
        raise InputValidationError(f"{name} contains NaN or infinite values.")
    return a


def _weights(w: Any, n: int, name: str) -> NDArray[np.float64]:
    if w is None:
        return np.ones(n)
    a = np.asarray(w, dtype=np.float64).ravel()
    if a.shape[0] != n:
        raise InputValidationError(f"{name} must have one weight per token ({n}); got {a.shape[0]}.")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
        raise InputValidationError(f"{name} must be finite, non-negative and not all zero.")
    return a


def _unit(a: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise InputValidationError("Embeddings must not be all-zero vectors (cosine similarity is undefined).")
    return a / norms


def _finish(
    metric: str, name: str, scores: NDArray[np.float64], average: Optional[str], params: dict[str, Any]
) -> MetricResult:
    if average not in ("mean", None):
        raise InputValidationError("average must be 'mean' or None (per example).")
    value: Any = scores if average is None else float(scores.mean())
    return MetricResult(metric, name, value, {**params, "n_examples": int(scores.shape[0])})


# ---------------------------------------------------------------- BERTScore
@register(
    category=_C,
    task="semantic-similarity",
    name="BERTScore",
    definition="Greedy matching of contextual token embeddings by cosine similarity: precision averages, over "
    "prediction tokens, the best similarity to any reference token; recall does the reverse; F1 combines them. "
    "Optional IDF weights and baseline rescaling as in the original implementation.",
    formula="P = Σ_j w_j max_i cos(r_i, p_j) / Σ_j w_j;  R = Σ_i w_i max_j cos(r_i, p_j) / Σ_i w_i;  "
    "F1 = 2PR/(P+R)",
    range="[-1, 1] ([0, 1] in practice)",
    input_requirements=("reference_embeddings", "prediction_embeddings"),
    references=(_REF_BERTSCORE,),
)
@per_item
def bertscore(
    reference_embeddings: Any,
    prediction_embeddings: Any,
    *,
    reference_weights: Any = None,
    prediction_weights: Any = None,
    measure: str = "f1",
    baseline: Optional[float] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """BERTScore from token embeddings you computed (e.g. a layer of ``roberta-large``, without [CLS]/[SEP]).

    ``reference_weights`` / ``prediction_weights``: optional IDF weight per token (one array per example).
    ``measure``: ``"f1"``, ``"precision"`` or ``"recall"``. ``baseline``: the model's baseline value ``b`` to
    rescale scores as ``(s − b) / (1 − b)`` (``rescale_with_baseline`` in ``bert_score``)."""
    refs = _matrices(reference_embeddings, "reference_embeddings")
    preds = _matrices(prediction_embeddings, "prediction_embeddings")
    if len(refs) != len(preds):
        raise InputValidationError(
            f"reference_embeddings and prediction_embeddings must contain the same number of examples. "
            f"Received {len(refs)} and {len(preds)}."
        )
    if refs[0].shape[1] != preds[0].shape[1]:
        raise InputValidationError("Reference and prediction embeddings must have the same dimension.")
    if measure not in ("f1", "precision", "recall"):
        raise InputValidationError("measure must be 'f1', 'precision' or 'recall'.")
    rw = [None] * len(refs) if reference_weights is None else list(reference_weights)
    pw = [None] * len(preds) if prediction_weights is None else list(prediction_weights)
    if len(rw) != len(refs) or len(pw) != len(preds):
        raise InputValidationError("Provide one weight array per example (or none).")
    scores = np.empty(len(refs))
    for k, (r, p) in enumerate(zip(refs, preds)):
        sim = _unit(r) @ _unit(p).T  # (ref tokens, pred tokens)
        wr, wp = (
            _weights(rw[k], r.shape[0], "reference_weights"),
            _weights(pw[k], p.shape[0], "prediction_weights"),
        )
        precision = float((sim.max(axis=0) * wp).sum() / wp.sum())
        recall = float((sim.max(axis=1) * wr).sum() / wr.sum())
        if baseline is not None:
            precision, recall = ((v - baseline) / (1 - baseline) for v in (precision, recall))
        if measure == "precision":
            scores[k] = precision
        elif measure == "recall":
            scores[k] = recall
        else:
            scores[k] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return _finish(
        "bertscore", f"BERTScore {measure}", scores, average, {"measure": measure, "baseline": baseline}
    )


# ---------------------------------------------------------------- sentence embeddings
@register(
    category=_C,
    task="semantic-similarity",
    name="Embedding similarity",
    definition="Similarity or distance between the sentence embedding of each prediction and of its reference: "
    "cosine similarity, or Euclidean / Manhattan distance. Values depend on the embedding model.",
    formula="cos = a·b / (‖a‖‖b‖);  euclidean = ‖a − b‖₂;  manhattan = ‖a − b‖₁",
    range="cosine [-1, 1]; distances [0, ∞)",
    input_requirements=("reference_embeddings", "prediction_embeddings"),
    references=("Reimers N, Gurevych I. Sentence-BERT. EMNLP-IJCNLP. 2019:3982-3992.",),
)
@per_item
def embedding_similarity(
    reference_embeddings: Any,
    prediction_embeddings: Any,
    *,
    metric: str = "cosine",
    normalize: bool = False,
    average: Optional[str] = "mean",
) -> MetricResult:
    """Row-wise similarity of two ``(n, dim)`` arrays of sentence embeddings. ``metric`` is ``"cosine"``
    (higher is more similar), ``"euclidean"`` or ``"manhattan"`` (lower is more similar). ``normalize=True``
    scales embeddings to unit length before a distance."""
    a = _vectors(reference_embeddings, "reference_embeddings")
    b = _vectors(prediction_embeddings, "prediction_embeddings")
    if a.shape != b.shape:
        raise InputValidationError(f"Embedding arrays must have the same shape; got {a.shape} and {b.shape}.")
    if metric == "cosine":
        scores = np.sum(_unit(a) * _unit(b), axis=1)
    elif metric in ("euclidean", "manhattan"):
        if normalize:
            a, b = _unit(a), _unit(b)
        diff = a - b
        scores = np.sqrt(np.sum(diff * diff, axis=1)) if metric == "euclidean" else np.abs(diff).sum(axis=1)
    else:
        raise InputValidationError("metric must be 'cosine', 'euclidean' or 'manhattan'.")
    return _finish("embedding_similarity", f"Embedding {metric}", scores, average, {"metric": metric})


# ---------------------------------------------------------------- MoverScore
def _emd(a: NDArray[np.float64], b: NDArray[np.float64], cost: NDArray[np.float64]) -> float:
    """Exact earth mover's distance between histograms a and b (both summing to 1) by linear programming."""
    from scipy.optimize import linprog

    n, m = cost.shape
    a_eq = np.zeros((n + m, n * m))
    for i in range(n):
        a_eq[i, i * m : (i + 1) * m] = 1
    for j in range(m):
        a_eq[n + j, j::m] = 1
    res = linprog(cost.ravel(), A_eq=a_eq[:-1], b_eq=np.concatenate([a, b])[:-1], bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover - the transport problem is always feasible
        raise InputValidationError(f"Earth mover's distance failed: {res.message}")
    return float(res.fun)


@register(
    category=_C,
    task="semantic-similarity",
    name="MoverScore",
    definition="One minus the earth mover's distance between the IDF-weighted token embeddings of prediction "
    "and reference, with Euclidean transport cost between L2-normalised embeddings (MoverScore v2 style).",
    formula="1 − EMD(w_ref, w_pred; ‖r̂_i − p̂_j‖₂)",
    range="(−∞, 1]",
    input_requirements=("reference_embeddings", "prediction_embeddings"),
    references=(_REF_MOVER,),
)
@per_item
def moverscore(
    reference_embeddings: Any,
    prediction_embeddings: Any,
    *,
    reference_weights: Any = None,
    prediction_weights: Any = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """MoverScore from token embeddings (and optional IDF weights) you computed; uses an exact transport
    solver (SciPy HiGHS), identical to POT's ``ot.emd2``."""
    refs = _matrices(reference_embeddings, "reference_embeddings")
    preds = _matrices(prediction_embeddings, "prediction_embeddings")
    if len(refs) != len(preds):
        raise InputValidationError("reference_embeddings and prediction_embeddings must have the same length.")
    rw = [None] * len(refs) if reference_weights is None else list(reference_weights)
    pw = [None] * len(preds) if prediction_weights is None else list(prediction_weights)
    scores = np.empty(len(refs))
    for k, (r, p) in enumerate(zip(refs, preds)):
        wr = _weights(rw[k], r.shape[0], "reference_weights")
        wp = _weights(pw[k], p.shape[0], "prediction_weights")
        ru, pu = _unit(r), _unit(p)
        cost = np.sqrt(np.maximum(((ru[:, None, :] - pu[None, :, :]) ** 2).sum(-1), 0))
        scores[k] = 1.0 - _emd(wr / wr.sum(), wp / wp.sum(), cost)
    return _finish("moverscore", "MoverScore", scores, average, {"weighted": reference_weights is not None})


# ---------------------------------------------------------------- MAUVE
def _kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    if np.any((p != 0) & (q == 0)):
        return math.inf
    m = (p != 0) & (q != 0)
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def _trapezoid(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def mauve_from_histograms(
    p_hist: NDArray[np.float64],
    q_hist: NDArray[np.float64],
    *,
    discretization: int = 25,
    scaling_factor: float = 5,
) -> float:
    """MAUVE from two quantized distributions (the step after clustering; identical to ``mauve-text``)."""
    weights = np.linspace(1e-6, 1 - 1e-6, discretization)
    curve = [[0.0, math.inf]]
    for w in np.sort(weights):
        r = w * p_hist + (1 - w) * q_hist
        curve.append([_kl(q_hist, r), _kl(p_hist, r)])
    curve.append([math.inf, 0.0])
    xy = np.exp(-scaling_factor * np.asarray(curve))
    x, y = xy[:, 0], xy[:, 1]
    i1, i2 = np.argsort(x), np.argsort(y)
    return 0.5 * (_trapezoid(x[i1], y[i1]) + _trapezoid(y[i2], x[i2]))


def _kmeans(
    x: NDArray[np.float64], k: int, rng: np.random.Generator, n_init: int, max_iter: int
) -> NDArray[np.int64]:
    best_labels, best_inertia = None, math.inf
    for _ in range(n_init):
        centers = [x[rng.integers(len(x))]]  # k-means++ seeding
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
            total = d2.sum()
            centers.append(x[rng.choice(len(x), p=d2 / total)] if total > 0 else x[rng.integers(len(x))])
        c = np.asarray(centers)
        labels = np.zeros(len(x), dtype=np.int64)
        for it in range(max_iter):
            d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
            new = d.argmin(axis=1)
            if it and np.array_equal(new, labels):
                break
            labels = new
            for j in range(k):
                members = x[labels == j]
                if len(members):
                    c[j] = members.mean(axis=0)
        inertia = float(((x - c[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    if best_labels is None:  # pragma: no cover - n_init >= 1 always sets it
        raise InputValidationError("k-means failed to run.")
    return best_labels


@register(
    category=_C,
    task="generation",
    name="MAUVE",
    definition="Gap between the distribution of generated text and of human text: both sets of feature vectors "
    "are quantized together (L2 normalisation, PCA to 90% variance, k-means), and MAUVE is the area under the "
    "divergence frontier of the two histograms. 1 means indistinguishable.",
    formula="area under {(exp(−c·KL(Q‖R_λ)), exp(−c·KL(P‖R_λ))) : R_λ = λP + (1−λ)Q}",
    range="(0, 1]",
    input_requirements=("reference_features", "generated_features"),
    references=(_REF_MAUVE, _REF_MAUVE2),
)
def mauve(
    reference_features: Any,
    generated_features: Any,
    *,
    num_buckets: Any = "auto",
    explained_variance: float = 0.9,
    kmeans_n_init: int = 5,
    kmeans_max_iter: int = 500,
    discretization: int = 25,
    scaling_factor: float = 5.0,
    smoothed: bool = False,
    random_state: Optional[int] = 0,
) -> MetricResult:
    """MAUVE between human (reference) and model (generated) texts from one feature vector per text (e.g. the
    last-token hidden state of GPT-2 large, as in the paper). ``num_buckets="auto"`` uses n/10 clusters.
    ``smoothed=True`` gives MAUVE* (Krichevsky–Trofimov smoothing, Pillutla et al. 2023). Results depend on the
    clustering; set ``random_state`` and report it."""
    p = _vectors(reference_features, "reference_features")
    q = _vectors(generated_features, "generated_features")
    if p.shape[1] != q.shape[1]:
        raise InputValidationError("reference_features and generated_features must have the same dimension.")
    if not 0 < explained_variance < 1:
        raise InputValidationError("explained_variance must be in (0, 1).")
    if num_buckets == "auto":
        k = max(2, round(min(len(p), len(q)) / 10))
    elif isinstance(num_buckets, (int, np.integer)) and num_buckets >= 2:
        k = int(num_buckets)
    else:
        raise InputValidationError("num_buckets must be 'auto' or an integer >= 2.")
    if k > len(p) + len(q):
        raise InputValidationError(f"num_buckets ({k}) exceeds the number of texts ({len(p) + len(q)}).")
    data = _unit(np.vstack([q, p]))
    centred = data - data.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    ratio = s**2 / np.sum(s**2) if np.sum(s**2) > 0 else np.ones_like(s) / len(s)
    dims = int(np.argmax(np.cumsum(ratio) >= explained_variance)) + 1
    reduced = centred @ vt[:dims].T
    labels = _kmeans(reduced, k, np.random.default_rng(random_state), int(kmeans_n_init), int(kmeans_max_iter))
    q_counts = np.bincount(labels[: len(q)], minlength=k).astype(float)
    p_counts = np.bincount(labels[len(q) :], minlength=k).astype(float)
    if smoothed:
        p_counts, q_counts = p_counts + 0.5, q_counts + 0.5
    value = mauve_from_histograms(
        p_counts / p_counts.sum(),
        q_counts / q_counts.sum(),
        discretization=discretization,
        scaling_factor=scaling_factor,
    )
    return MetricResult(
        "mauve",
        "MAUVE*" if smoothed else "MAUVE",
        value,
        {
            "num_buckets": k,
            "pca_dims": dims,
            "random_state": random_state,
            "scaling_factor": scaling_factor,
            "n_reference": len(p),
            "n_generated": len(q),
        },
    )


# ---------------------------------------------------------------- learned-metric adapter
@register(
    category=_C,
    task="learned-metric",
    name="Model-based score (COMET, BLEURT, BARTScore, AlignScore, ...)",
    definition="Scores from any learned evaluation model or judge you supply, per example, so they get the same "
    "confidence intervals, model comparison and reports as every other metric.",
    formula="mean_i scorer(reference_i, prediction_i[, source_i])",
    range="that of the scorer",
    input_requirements=("references", "predictions", "scorer"),
    references=(_REF_COMET, _REF_BLEURT),
)
@per_item
def model_score(
    references: Any,
    predictions: Any,
    *,
    scorer: Callable[..., Any],
    sources: Any = None,
    batch_size: int = 64,
    name: str = "model score",
    higher_is_better: bool = True,
    average: Optional[str] = "mean",
) -> MetricResult:
    """Run ``scorer(references, predictions)`` (or ``scorer(references, predictions, sources)``) in batches
    and collect one score per example. Examples:

    - COMET: ``scorer=lambda r, p, s: model.predict(
      [{"src": a, "mt": b, "ref": c} for a, b, c in zip(s, p, r)]).scores``
    - BLEURT: ``scorer=lambda r, p: bleurt_scorer.score(references=r, candidates=p)``
    """
    if not callable(scorer):
        raise InputValidationError("scorer must be a callable taking (references, predictions[, sources]).")
    refs = list(references)
    preds = list(predictions)
    if len(refs) != len(preds) or not refs:
        raise InputValidationError("references and predictions must be non-empty and the same length.")
    srcs = None if sources is None else list(sources)
    if srcs is not None and len(srcs) != len(refs):
        raise InputValidationError("sources must have one entry per example.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise InputValidationError("batch_size must be a positive integer.")
    out: list[float] = []
    for start in range(0, len(refs), batch_size):
        sl = slice(start, start + batch_size)
        got = scorer(refs[sl], preds[sl]) if srcs is None else scorer(refs[sl], preds[sl], srcs[sl])
        vals = np.asarray(got, dtype=np.float64).ravel()
        if vals.shape[0] != len(refs[sl]):
            raise InputValidationError(
                f"scorer returned {vals.shape[0]} scores for a batch of {len(refs[sl])} examples."
            )
        if not np.all(np.isfinite(vals)):
            raise InputValidationError("scorer returned NaN or infinite scores.")
        out.extend(vals.tolist())
    return _finish("model_score", name, np.array(out), average, {"higher_is_better": higher_is_better})
