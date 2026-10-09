"""Text-generation metrics: BLEU, chrF / chrF++, TER, ROUGE, lexical diversity and language-model likelihood.

``references`` come first (like ``y_true``) and ``predictions`` second. Each example may have one reference
string or a list of reference strings. BLEU, chrF and TER are corpus-level and reported on a 0–100 scale, with
the tokenization, smoothing and multi-reference rules of sacreBLEU (Post 2018), so scores are comparable with
published results; ROUGE follows Google's ``rouge-score`` package (0–1, averaged over examples).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import as_references, as_texts, check_positive_int, per_item, word_ngrams

__all__ = [
    "bleu",
    "chrf",
    "cider",
    "cross_entropy",
    "distinct_n",
    "meteor",
    "perplexity",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "rouge_lsum",
    "self_bleu",
    "sentence_bleu",
    "ter",
]

_C = "text"
_REF_BLEU = (
    "Papineni K, Roukos S, Ward T, Zhu WJ. BLEU: a method for automatic evaluation of machine translation. "
    "ACL. 2002:311-318."
)
_REF_SACRE = "Post M. A call for clarity in reporting BLEU scores. WMT. 2018:186-191."
_REF_SMOOTH = (
    "Chen B, Cherry C. A systematic comparison of smoothing techniques for sentence-level BLEU. WMT. 2014."
)
_REF_CHRF = "Popović M. chrF: character n-gram F-score for automatic MT evaluation. WMT. 2015:392-395."
_REF_CHRFPP = "Popović M. chrF++: words helping character n-grams. WMT. 2017:612-618."
_REF_TER = (
    "Snover M, Dorr B, Schwartz R, Micciulla L, Makhoul J. A study of translation edit rate with targeted human "
    "annotation. AMTA. 2006:223-231."
)
_REF_ROUGE = (
    "Lin CY. ROUGE: a package for automatic evaluation of summaries. Text Summarization Branches Out. 2004."
)
_REF_DISTINCT = (
    "Li J, Galley M, Brockett C, Gao J, Dolan B. A diversity-promoting objective function for neural "
    "conversation models. NAACL. 2016:110-119."
)
_REF_SELFBLEU = "Zhu Y, et al. Texygen: a benchmarking platform for text generation models. SIGIR. 2018:1097-1100."
_REF_PPL = (
    "Jelinek F, Mercer RL, Bahl LR, Baker JK. Perplexity—a measure of the difficulty of speech recognition "
    "tasks. JASA. 1977;62(S1):S63."
)


# ---------------------------------------------------------------- tokenization (sacreBLEU 13a)
_RE_13A = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),  # punctuation (apostrophe excluded)
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),  # period and comma unless preceded by a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),  # ... unless followed by a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),  # dash after a digit
]


def _tok_13a(line: str) -> list[str]:
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    line = f" {line} "
    for pattern, repl in _RE_13A:
        line = pattern.sub(repl, line)
    return line.split()


Tokenize = Literal["13a", "none"]


def _tokenizer(tokenize: Any) -> Callable[[str], list[str]]:
    if tokenize == "13a":
        return _tok_13a
    if tokenize == "none":
        return str.split
    if callable(tokenize):
        return lambda s: list(tokenize(s))
    raise InputValidationError("tokenize must be '13a' (sacreBLEU default), 'none' (whitespace) or a callable.")


# ---------------------------------------------------------------- BLEU
_SMOOTH_DEFAULTS: dict[str, Optional[float]] = {"none": None, "floor": 0.1, "add-k": 1.0, "exp": None}


def _bleu_stats(hyp: list[str], refs: list[list[str]], order: int) -> list[float]:
    """[hyp_len, ref_len, correct_1..N, total_1..N] for one segment (sacreBLEU)."""
    hyp_len = len(hyp)
    ref_lens = [len(r) for r in refs]
    closest = min(ref_lens, key=lambda r: (abs(hyp_len - r), r))
    ref_max: Counter[tuple[str, ...]] = Counter()
    for r in refs:
        for n in range(1, order + 1):
            for ng, c in word_ngrams(r, n).items():
                if c > ref_max[ng]:
                    ref_max[ng] = c
    correct = [0] * order
    total = [0] * order
    for n in range(1, order + 1):
        hyp_ng = word_ngrams(hyp, n)
        total[n - 1] = max(hyp_len - n + 1, 0)
        correct[n - 1] = sum(min(c, ref_max.get(ng, 0)) for ng, c in hyp_ng.items())
    return [hyp_len, closest, *correct, *total]


def _bleu_from_stats(
    stats: Sequence[float], order: int, smooth: str, smooth_value: Optional[float], effective_order: bool
) -> tuple[float, list[float], float]:
    sys_len, ref_len = stats[0], stats[1]
    correct = list(stats[2 : 2 + order])
    total = list(stats[2 + order :])
    if smooth_value is None:
        smooth_value = _SMOOTH_DEFAULTS[smooth]
    bp = 1.0 if sys_len >= ref_len else (math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0)
    precisions = [0.0] * order
    if not any(correct):
        return 0.0, precisions, bp
    smooth_mteval = 1.0
    eff = order
    for n in range(1, order + 1):
        if smooth == "add-k" and n > 1:
            correct[n - 1] += smooth_value  # type: ignore[operator]
            total[n - 1] += smooth_value  # type: ignore[operator]
        if total[n - 1] == 0:
            break
        if effective_order:
            eff = n
        if correct[n - 1] == 0:
            if smooth == "exp":
                smooth_mteval *= 2
                precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
            elif smooth == "floor":
                precisions[n - 1] = 100.0 * smooth_value / total[n - 1]  # type: ignore[operator]
        else:
            precisions[n - 1] = 100.0 * correct[n - 1] / total[n - 1]
    logs = [math.log(p) if p > 0 else -9999999999.0 for p in precisions[:eff]]
    return bp * math.exp(sum(logs) / eff), precisions, bp


def _check_smooth(smooth: str) -> str:
    if smooth not in _SMOOTH_DEFAULTS:
        raise InputValidationError(f"smooth must be one of {sorted(_SMOOTH_DEFAULTS)}; got {smooth!r}.")
    return smooth


def _prep(texts: list[str], lowercase: bool, tok: Callable[[str], list[str]]) -> list[list[str]]:
    return [tok((t.lower() if lowercase else t).rstrip()) for t in texts]


@register(
    category=_C,
    task="generation",
    name="BLEU",
    definition="Corpus-level geometric mean of clipped n-gram precisions (n = 1..4) times a brevity penalty, "
    "computed exactly as sacreBLEU (13a tokenization, exponential smoothing).",
    formula="BLEU = BP · exp(Σ_n (1/N) log p_n), BP = min(1, exp(1 − r/c))",
    range="[0, 100]",
    input_requirements=("references", "predictions"),
    references=(_REF_BLEU, _REF_SACRE, _REF_SMOOTH),
)
@per_item
def bleu(
    references: Any,
    predictions: Any,
    *,
    max_order: int = 4,
    smooth: str = "exp",
    smooth_value: Optional[float] = None,
    lowercase: bool = False,
    tokenize: Any = "13a",
    effective_order: bool = False,
) -> MetricResult:
    """Corpus BLEU (0–100), identical to ``sacrebleu.corpus_bleu`` with the same options.

    Statistics are summed over the corpus before the precisions are taken, which is what BLEU is defined on;
    averaging sentence BLEU is a different (and noisier) quantity: use :func:`sentence_bleu` for that.
    """
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    order = check_positive_int(max_order, "max_order")
    smooth = _check_smooth(smooth)
    tok = _tokenizer(tokenize)
    hyp_tok = _prep(preds, lowercase, tok)
    stats = np.zeros(2 + 2 * order)
    for h, rs in zip(hyp_tok, refs):
        stats += np.array(_bleu_stats(h, _prep(rs, lowercase, tok), order), dtype=float)
    score, precisions, bp = _bleu_from_stats(stats.tolist(), order, smooth, smooth_value, effective_order)
    return MetricResult(
        "bleu",
        "BLEU",
        score,
        {
            "max_order": order,
            "smooth": smooth,
            "lowercase": lowercase,
            "tokenize": tokenize if isinstance(tokenize, str) else "custom",
            "precisions": [round(p, 6) for p in precisions],
            "brevity_penalty": bp,
            "hyp_len": int(stats[0]),
            "ref_len": int(stats[1]),
            "n_examples": len(preds),
        },
    )


@register(
    category=_C,
    task="generation",
    name="Sentence BLEU",
    definition="BLEU computed for each example separately (exponential smoothing, effective order) and "
    "averaged; use for per-example scores, not for reporting corpus quality.",
    formula="mean_i BLEU(prediction_i, references_i)",
    range="[0, 100]",
    input_requirements=("references", "predictions"),
    references=(_REF_BLEU, _REF_SMOOTH),
)
@per_item
def sentence_bleu(
    references: Any,
    predictions: Any,
    *,
    max_order: int = 4,
    smooth: str = "exp",
    smooth_value: Optional[float] = None,
    lowercase: bool = False,
    tokenize: Any = "13a",
    average: Optional[str] = "mean",
) -> MetricResult:
    """Mean sentence-level BLEU (``sacrebleu.sentence_bleu`` per example); ``average=None`` returns the array."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    order = check_positive_int(max_order, "max_order")
    smooth = _check_smooth(smooth)
    tok = _tokenizer(tokenize)
    scores = np.array(
        [
            _bleu_from_stats(_bleu_stats(h, _prep(rs, lowercase, tok), order), order, smooth, smooth_value, True)[
                0
            ]
            for h, rs in zip(_prep(preds, lowercase, tok), refs)
        ]
    )
    value: Any = scores if average is None else float(scores.mean())
    return MetricResult("sentence_bleu", "Sentence BLEU", value, {"max_order": order, "smooth": smooth})


# ---------------------------------------------------------------- chrF / chrF++
_PUNCTS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _chrf_words(sent: str) -> list[str]:
    out: list[str] = []
    for w in sent.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def _chrf_ngrams(sent: str, char_order: int, word_order: int, whitespace: bool) -> list[Counter[Any]]:
    line = sent if whitespace else "".join(sent.split())
    out: list[Counter[Any]] = [
        Counter(line[i : i + n] for i in range(len(line) - n + 1)) for n in range(1, char_order + 1)
    ]
    if word_order:
        words = _chrf_words(sent)
        out += [
            Counter(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
            for n in range(1, word_order + 1)
        ]
    return out


def _chrf_f(stats: Sequence[float], order: int, beta: float, eps_smoothing: bool) -> float:
    eps = 1e-16
    factor = beta**2
    score = 0.0
    avg_p = avg_r = 0.0
    eff = 0
    for i in range(order):
        n_hyp, n_ref, n_match = stats[3 * i : 3 * i + 3]
        p = n_match / n_hyp if n_hyp > 0 else eps
        r = n_match / n_ref if n_ref > 0 else eps
        denom = factor * p + r
        score += ((1 + factor) * p * r / denom) if denom > 0 else eps
        if n_hyp > 0 and n_ref > 0:
            avg_p += p
            avg_r += r
            eff += 1
    if eps_smoothing:
        return 100 * score / order
    if eff == 0:
        return 0.0
    avg_p /= eff
    avg_r /= eff
    return 100 * (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r) if avg_p + avg_r else 0.0


@register(
    category=_C,
    task="generation",
    name="chrF / chrF++",
    definition="F-beta score (β = 2) over character n-grams (n = 1..6), plus word uni- and bigrams for chrF++ "
    "(word_order=2); robust for morphologically rich languages. Computed exactly as sacreBLEU.",
    formula="chrF_β = (1 + β²) · chrP · chrR / (β² · chrP + chrR)",
    range="[0, 100]",
    input_requirements=("references", "predictions"),
    references=(_REF_CHRF, _REF_CHRFPP, _REF_SACRE),
)
@per_item
def chrf(
    references: Any,
    predictions: Any,
    *,
    char_order: int = 6,
    word_order: int = 0,
    beta: float = 2,
    lowercase: bool = False,
    whitespace: bool = False,
    eps_smoothing: bool = False,
) -> MetricResult:
    """Corpus chrF (``word_order=0``) or chrF++ (``word_order=2``), identical to ``sacrebleu.corpus_chrf``.
    With several references, each example uses the reference that gives it the best F-score."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    char_order = check_positive_int(char_order, "char_order")
    if isinstance(word_order, bool) or not isinstance(word_order, (int, np.integer)) or word_order < 0:
        raise InputValidationError("word_order must be an integer >= 0 (2 gives chrF++).")
    order = char_order + int(word_order)
    total = np.zeros(3 * order)
    for hyp, rs in zip(preds, refs):
        h = _chrf_ngrams(hyp.lower() if lowercase else hyp, char_order, word_order, whitespace)
        best: Optional[list[float]] = None
        best_f = -1.0
        for ref in rs:
            r = _chrf_ngrams(ref.lower() if lowercase else ref, char_order, word_order, whitespace)
            seg: list[float] = []
            for hc, rc in zip(h, r):
                n_hyp = sum(hc.values())
                match = sum(min(c, rc[ng]) for ng, c in hc.items() if ng in rc)
                seg += [n_hyp if rc else 0, sum(rc.values()), match]
            f = _chrf_f(seg, order, beta, eps_smoothing)
            if f > best_f:
                best_f, best = f, seg
        total += np.array(best)
    name = f"chrF{beta:g}" + "+" * int(word_order)
    return MetricResult(
        "chrf",
        name,
        _chrf_f(total.tolist(), order, beta, eps_smoothing),
        {"char_order": char_order, "word_order": int(word_order), "beta": beta, "lowercase": lowercase},
    )


# ---------------------------------------------------------------- TER (Tercom, as in sacreBLEU)
_MAX_SHIFT_SIZE = 10
_MAX_SHIFT_DIST = 50
_BEAM_WIDTH = 25
_MAX_SHIFT_CANDIDATES = 1000
_INF = int(1e16)


def _edit_distance(hyp: list[str], ref: list[str]) -> tuple[int, str]:
    """Beam edit distance (insert/delete/substitute cost 1) and the trace, as Tercom/sacreBLEU."""
    nh, nr = len(hyp), len(ref)
    dist: list[list[tuple[int, str]]] = [[(i, "i") for i in range(nr + 1)]]
    dist += [[(_INF, "x")] * (nr + 1) for _ in range(nh)]
    ratio = nr / nh if hyp else 1
    beam = math.ceil(ratio / 2 + _BEAM_WIDTH) if ratio / 2 > _BEAM_WIDTH else _BEAM_WIDTH
    for i in range(1, nh + 1):
        diag = math.floor(i * ratio)
        lo = max(0, diag - beam)
        hi = nr + 1 if i == nh else min(nr + 1, diag + beam)
        row, prev = dist[i], dist[i - 1]
        for j in range(lo, hi):
            if j == 0:
                row[j] = (prev[j][0] + 1, "d")
                continue
            sub = 0 if hyp[i - 1] == ref[j - 1] else 1
            best = row[j]
            for cost, op in (
                (prev[j - 1][0] + sub, " " if sub == 0 else "s"),
                (prev[j][0] + 1, "d"),
                (row[j - 1][0] + 1, "i"),
            ):
                if best[0] > cost:
                    best = (cost, op)
            row[j] = best
    trace = []
    i, j = nh, nr
    while i > 0 or j > 0:
        op = dist[i][j][1]
        trace.append(op)
        if op in ("s", " "):
            i -= 1
            j -= 1
        elif op == "i":
            j -= 1
        elif op == "d":
            i -= 1
        else:  # pragma: no cover - unreachable for valid tables
            raise RuntimeError("invalid edit trace")
    return dist[nh][nr][0], "".join(reversed(trace))


def _ter_shift(hyp: list[str], ref: list[str], checked: int) -> tuple[int, list[str], int]:
    pre, inv_trace = _edit_distance(hyp, ref)
    trace = inv_trace.translate(str.maketrans("id", "di"))
    align: dict[int, int] = {}
    ref_err: list[int] = []
    hyp_err: list[int] = []
    ph = pr = -1
    for op in trace:
        if op in (" ", "s"):
            ph += 1
            pr += 1
            align[pr] = ph
            e = 0 if op == " " else 1
            hyp_err.append(e)
            ref_err.append(e)
        elif op == "i":
            ph += 1
            hyp_err.append(1)
        else:
            pr += 1
            align[pr] = ph
            ref_err.append(1)
    best: Optional[tuple[int, int, int, int, list[str]]] = None
    for sh, sr, length in _shifted_pairs(hyp, ref):
        if sum(hyp_err[sh : sh + length]) == 0 or sum(ref_err[sr : sr + length]) == 0:
            continue
        if sh <= align[sr] < sh + length:
            continue
        prev_idx = -1
        for off in range(-1, length):
            if sr + off == -1:
                idx = 0
            elif sr + off in align:
                idx = align[sr + off] + 1
            else:
                break
            if idx == prev_idx:
                continue
            prev_idx = idx
            shifted = _perform_shift(hyp, sh, length, idx)
            cand = (pre - _edit_distance(shifted, ref)[0], length, -sh, -idx, shifted)
            checked += 1
            if best is None or cand > best:
                best = cand
        if checked >= _MAX_SHIFT_CANDIDATES:
            break
    if best is None:
        return 0, hyp, checked
    return best[0], best[4], checked


def _shifted_pairs(hyp: list[str], ref: list[str]) -> Any:
    """(start_h, start_r, length) of matching word runs, in Tercom's order."""
    nh, nr = len(hyp), len(ref)
    for sh in range(nh):
        for sr in range(nr):
            if abs(sr - sh) > _MAX_SHIFT_DIST:
                continue
            length = 0
            while hyp[sh + length] == ref[sr + length] and length < _MAX_SHIFT_SIZE:
                length += 1
                yield sh, sr, length
                if nh == sh + length or nr == sr + length:
                    break


def _perform_shift(words: list[str], start: int, length: int, target: int) -> list[str]:
    if target < start:
        return words[:target] + words[start : start + length] + words[target:start] + words[start + length :]
    if target > start + length:
        return words[:start] + words[start + length : target] + words[start : start + length] + words[target:]
    return (
        words[:start]
        + words[start + length : length + target]
        + words[start : start + length]
        + words[length + target :]
    )


def _ter_edits(hyp: list[str], ref: list[str]) -> int:
    if not ref:
        return len(hyp)
    shifts = 0
    words = hyp
    checked = 0
    while True:
        delta, new_words, checked = _ter_shift(words, ref, checked)
        if checked >= _MAX_SHIFT_CANDIDATES or delta <= 0:
            break
        shifts += 1
        words = new_words
    return shifts + _edit_distance(words, ref)[0]


@register(
    category=_C,
    task="generation",
    name="TER",
    definition="Translation edit rate: minimum number of insertions, deletions, substitutions and block shifts "
    "to turn the prediction into the closest reference, divided by the average reference length (Tercom, as in "
    "sacreBLEU). Lower is better.",
    formula="TER = Σ edits / Σ average reference length × 100",
    range="[0, ∞)",
    input_requirements=("references", "predictions"),
    references=(_REF_TER, _REF_SACRE),
    higher_is_better=False,
)
@per_item
def ter(references: Any, predictions: Any, *, case_sensitive: bool = False) -> MetricResult:
    """Corpus TER (0–100+, lower is better), identical to ``sacrebleu.corpus_ter`` with default settings."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))

    def tok(s: str) -> list[str]:
        s = s.rstrip()
        return (s if case_sensitive else s.lower()).split()

    edits = 0.0
    ref_len = 0.0
    for hyp, rs in zip(preds, refs):
        h = tok(hyp)
        rws = [tok(r) for r in rs]
        edits += min(_ter_edits(h, r) for r in rws)
        ref_len += sum(len(r) for r in rws) / len(rws)
    value = 100 * edits / ref_len if ref_len > 0 else (100.0 if edits > 0 else 0.0)
    return MetricResult(
        "ter", "TER", value, {"case_sensitive": case_sensitive, "edits": edits, "ref_len": ref_len}
    )


# ---------------------------------------------------------------- ROUGE (rouge-score)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VALID = re.compile(r"^[a-z0-9]+$")


def _rouge_tok(text: str, stemmer: Optional[Callable[[str], str]]) -> list[str]:
    tokens = re.split(r"\s+", _NON_ALNUM.sub(" ", text.lower()))
    if stemmer is not None:
        tokens = [stemmer(t) if len(t) > 3 else t for t in tokens]
    return [t for t in tokens if _VALID.match(t)]


def _f(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def _rouge_n(t: list[str], p: list[str], n: int) -> tuple[float, float, float]:
    tn, pn = word_ngrams(t, n), word_ngrams(p, n)
    inter = sum(min(c, pn[g]) for g, c in tn.items())
    prec = inter / max(sum(pn.values()), 1)
    rec = inter / max(sum(tn.values()), 1)
    return prec, rec, _f(prec, rec)


def _lcs_table(a: list[str], b: list[str]) -> list[list[int]]:
    t = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        row, prev = t[i], t[i - 1]
        for j in range(1, len(b) + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else max(prev[j], row[j - 1])
    return t


def _rouge_l(t: list[str], p: list[str]) -> tuple[float, float, float]:
    if not t or not p:
        return 0.0, 0.0, 0.0
    lcs = _lcs_table(t, p)[-1][-1]
    prec, rec = lcs / len(p), lcs / len(t)
    return prec, rec, _f(prec, rec)


def _lcs_indices(ref: list[str], can: list[str]) -> list[int]:
    t = _lcs_table(ref, can)
    i, j = len(ref), len(can)
    out: list[int] = []
    while i > 0 and j > 0:
        if ref[i - 1] == can[j - 1]:
            out.append(i - 1)
            i -= 1
            j -= 1
        elif t[i][j - 1] > t[i - 1][j]:
            j -= 1
        else:
            i -= 1
    return out


def _rouge_lsum(ts: list[list[str]], ps: list[list[str]]) -> tuple[float, float, float]:
    if not ts or not ps:
        return 0.0, 0.0, 0.0
    m, n = sum(map(len, ts)), sum(map(len, ps))
    if not n or not m:
        return 0.0, 0.0, 0.0
    cr: Counter[str] = Counter()
    cc: Counter[str] = Counter()
    for s in ts:
        cr.update(s)
    for s in ps:
        cc.update(s)
    hits = 0
    for r in ts:
        union = sorted(set().union(*[_lcs_indices(r, c) for c in ps]))
        for tok in (r[i] for i in union):
            if cc[tok] > 0 and cr[tok] > 0:
                hits += 1
                cc[tok] -= 1
                cr[tok] -= 1
    prec, rec = hits / n, hits / m
    return prec, rec, _f(prec, rec)


_MEASURES = {"precision": 0, "recall": 1, "fmeasure": 2}


def _rouge(
    variant: str,
    references: Any,
    predictions: Any,
    measure: str,
    stemmer: Optional[Callable[[str], str]],
    average: Optional[str],
) -> MetricResult:
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    if measure not in _MEASURES:
        raise InputValidationError("measure must be 'fmeasure', 'precision' or 'recall'.")
    if stemmer is not None and not callable(stemmer):
        raise InputValidationError("stemmer must be a callable str -> str, e.g. nltk PorterStemmer().stem.")
    k = _MEASURES[measure]

    def sents(text: str) -> list[list[str]]:
        return [_rouge_tok(s, stemmer) for s in text.split("\n") if len(s)]

    scores = []
    for pred, rs in zip(preds, refs):
        triples = []
        for ref in rs:
            if variant == "Lsum":
                triples.append(_rouge_lsum(sents(ref), sents(pred)))
            else:
                t, p = _rouge_tok(ref, stemmer), _rouge_tok(pred, stemmer)
                triples.append(_rouge_l(t, p) if variant == "L" else _rouge_n(t, p, int(variant)))
        best = max(range(len(triples)), key=lambda i: (triples[i][2], -i))  # highest F, first on ties
        scores.append(triples[best][k])
    arr = np.array(scores)
    value: Any = arr if average is None else float(arr.mean())
    metric = {"1": "rouge_1", "2": "rouge_2", "L": "rouge_l", "Lsum": "rouge_lsum"}[variant]
    return MetricResult(metric, f"ROUGE-{variant}", value, {"measure": measure, "stemmer": stemmer is not None})


def _rouge_doc(variant: str, what: str) -> dict[str, Any]:
    return {
        "category": _C,
        "task": "generation",
        "name": f"ROUGE-{variant}",
        "definition": f"{what} between prediction and reference after lowercasing and splitting on "
        "non-alphanumerics (as Google's rouge-score); F-measure averaged over examples, best reference per "
        "example.",
        "range": "[0, 1]",
        "input_requirements": ("references", "predictions"),
        "references": (_REF_ROUGE,),
    }


@register(formula="F1 of clipped unigram overlap", **_rouge_doc("1", "Unigram overlap"))
@per_item
def rouge_1(
    references: Any,
    predictions: Any,
    *,
    measure: str = "fmeasure",
    stemmer: Optional[Callable[[str], str]] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """ROUGE-1 (identical to ``rouge_score`` ``rouge1``; pass ``stemmer=PorterStemmer().stem`` for stemming)."""
    return _rouge("1", references, predictions, measure, stemmer, average)


@register(formula="F1 of clipped bigram overlap", **_rouge_doc("2", "Bigram overlap"))
@per_item
def rouge_2(
    references: Any,
    predictions: Any,
    *,
    measure: str = "fmeasure",
    stemmer: Optional[Callable[[str], str]] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """ROUGE-2 (identical to ``rouge_score`` ``rouge2``)."""
    return _rouge("2", references, predictions, measure, stemmer, average)


@register(formula="P = LCS/|pred|, R = LCS/|ref|, F1", **_rouge_doc("L", "Longest common subsequence"))
@per_item
def rouge_l(
    references: Any,
    predictions: Any,
    *,
    measure: str = "fmeasure",
    stemmer: Optional[Callable[[str], str]] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """ROUGE-L, sentence-level LCS (identical to ``rouge_score`` ``rougeL``)."""
    return _rouge("L", references, predictions, measure, stemmer, average)


@register(
    formula="union-LCS over newline-separated sentences",
    **_rouge_doc("Lsum", "Summary-level union LCS over sentences (one sentence per line)"),
)
@per_item
def rouge_lsum(
    references: Any,
    predictions: Any,
    *,
    measure: str = "fmeasure",
    stemmer: Optional[Callable[[str], str]] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    """ROUGE-Lsum (identical to ``rouge_score`` ``rougeLsum``); put one sentence per line."""
    return _rouge("Lsum", references, predictions, measure, stemmer, average)


# ---------------------------------------------------------------- diversity
@register(
    category=_C,
    task="generation",
    name="Distinct-n",
    definition="Number of distinct n-grams divided by the total number of n-grams across all predictions "
    "(whitespace tokens); a simple lexical-diversity indicator.",
    formula="|unique n-grams| / |n-grams|",
    range="[0, 1]",
    input_requirements=("predictions",),
    references=(_REF_DISTINCT,),
)
@per_item
def distinct_n(predictions: Any, *, n: int = 1, lowercase: bool = False) -> MetricResult:
    """Distinct-n over the whole set of predictions (``n=1`` Distinct-1, ``n=2`` Distinct-2)."""
    preds = as_texts(predictions, "predictions")
    n = check_positive_int(n, "n")
    counts: Counter[tuple[str, ...]] = Counter()
    for p in preds:
        counts.update(word_ngrams((p.lower() if lowercase else p).split(), n))
    total = sum(counts.values())
    if total == 0:
        raise InputValidationError(f"The predictions contain no {n}-grams.")
    return MetricResult("distinct_n", f"Distinct-{n}", len(counts) / total, {"n": n, "total_ngrams": total})


@register(
    category=_C,
    task="generation",
    name="Self-BLEU",
    definition="Average sentence BLEU of each prediction against all other predictions as references; higher "
    "means the outputs are more alike (less diverse).",
    formula="mean_i BLEU(p_i, {p_j : j ≠ i})",
    range="[0, 100]",
    input_requirements=("predictions",),
    references=(_REF_SELFBLEU,),
    higher_is_better=False,
)
@per_item
def self_bleu(predictions: Any, *, max_order: int = 4, tokenize: Any = "13a") -> MetricResult:
    """Self-BLEU with sacreBLEU sentence settings (exponential smoothing, effective order)."""
    preds = as_texts(predictions, "predictions")
    if len(preds) < 2:
        raise InputValidationError("Self-BLEU needs at least two predictions.")
    order = check_positive_int(max_order, "max_order")
    tok = _tokenizer(tokenize)
    toks = [tok(p.rstrip()) for p in preds]
    scores = [
        _bleu_from_stats(_bleu_stats(toks[i], toks[:i] + toks[i + 1 :], order), order, "exp", None, True)[0]
        for i in range(len(toks))
    ]
    return MetricResult("self_bleu", "Self-BLEU", float(np.mean(scores)), {"max_order": order})


# ---------------------------------------------------------------- language-model likelihood
def _logprob_arrays(token_logprobs: Any) -> list[np.ndarray]:
    if isinstance(token_logprobs, np.ndarray) and token_logprobs.ndim == 1:
        token_logprobs = [token_logprobs]
    seqs = list(token_logprobs) if not isinstance(token_logprobs, (str, bytes)) else None
    if not seqs:
        raise InputValidationError("token_logprobs must be a non-empty list of per-token log-probabilities.")
    if all(isinstance(v, (int, float, np.floating)) for v in seqs):
        seqs = [seqs]
    out = []
    for i, s in enumerate(seqs):
        a = np.asarray(s, dtype=np.float64).ravel()
        if a.size == 0:
            raise InputValidationError(f"token_logprobs[{i}] is empty.")
        if not np.all(np.isfinite(a)) or np.any(a > 1e-9):
            raise InputValidationError(
                f"token_logprobs[{i}] must contain finite natural-log probabilities (<= 0)."
            )
        out.append(a)
    return out


@register(
    category=_C,
    task="language-modeling",
    name="Cross-entropy (negative log-likelihood)",
    definition="Average negative log-probability the model assigns to the observed tokens, pooled over all "
    "tokens of all sequences.",
    formula="H = −(1/T) Σ_t log p(x_t | x_<t)",
    range="[0, ∞)",
    input_requirements=("token_logprobs",),
    references=(_REF_PPL,),
    higher_is_better=False,
)
@per_item
def cross_entropy(token_logprobs: Any, *, base: float = math.e) -> MetricResult:
    """Token-pooled cross-entropy from natural-log token probabilities, in nats (``base=2`` for bits)."""
    seqs = _logprob_arrays(token_logprobs)
    total = sum(float(s.sum()) for s in seqs)
    n_tokens = sum(s.size for s in seqs)
    if base <= 0 or base == 1:
        raise InputValidationError("base must be positive and not 1.")
    value = -total / n_tokens / math.log(base)
    return MetricResult("cross_entropy", "Cross-entropy", value, {"base": base, "n_tokens": n_tokens})


@register(
    category=_C,
    task="language-modeling",
    name="Perplexity",
    definition="Exponentiated token-average negative log-likelihood, pooled over all tokens. Only comparable "
    "between models that share a tokenizer.",
    formula="PPL = exp(−(1/T) Σ_t log p(x_t | x_<t))",
    range="[1, ∞)",
    input_requirements=("token_logprobs",),
    references=(_REF_PPL,),
    higher_is_better=False,
)
@per_item
def perplexity(token_logprobs: Any, *, average: str = "tokens") -> MetricResult:
    """Perplexity from per-token natural-log probabilities (one array per sequence).

    ``average="tokens"`` pools all tokens (the usual corpus perplexity); ``"sequences"`` averages the
    per-sequence perplexities instead."""
    seqs = _logprob_arrays(token_logprobs)
    if average == "tokens":
        value = math.exp(-sum(float(s.sum()) for s in seqs) / sum(s.size for s in seqs))
    elif average == "sequences":
        value = float(np.mean([math.exp(-float(s.mean())) for s in seqs]))
    else:
        raise InputValidationError("average must be 'tokens' or 'sequences'.")
    return MetricResult("perplexity", "Perplexity", value, {"average": average, "n_sequences": len(seqs)})


# ---------------------------------------------------------------- METEOR
_REF_METEOR = (
    "Lavie A, Agarwal A. METEOR: an automatic metric for MT evaluation with high levels of correlation with "
    "human judgments. WMT. 2007:228-231."
)
_REF_BANERJEE = (
    "Banerjee S, Lavie A. METEOR: an automatic metric for MT evaluation with improved correlation with human "
    "judgments. ACL Workshop on Evaluation Measures. 2005:65-72."
)


def _meteor_match(
    hyp: list[tuple[int, str]], ref: list[tuple[int, str]]
) -> tuple[list[tuple[int, int]], Any, Any]:
    """NLTK's matcher: each hypothesis word (from the end) takes the latest unused identical reference word."""
    positions: dict[str, list[int]] = {}
    for j, (_, w) in enumerate(ref):
        positions.setdefault(w, []).append(j)
    pairs, used_h, used_r = [], set(), set()
    for i in range(len(hyp) - 1, -1, -1):
        pos = positions.get(hyp[i][1])
        if pos:
            j = pos.pop()
            used_h.add(i)
            used_r.add(j)
            pairs.append((hyp[i][0], ref[j][0]))
    return (
        pairs,
        [p for i, p in enumerate(hyp) if i not in used_h],
        [p for j, p in enumerate(ref) if j not in used_r],
    )


def _meteor_synonym_match(
    hyp: list[tuple[int, str]], ref: list[tuple[int, str]], synonyms: Callable[[str], Any]
) -> list[tuple[int, int]]:
    positions: dict[str, list[int]] = {}
    for j, (_, w) in enumerate(ref):
        positions.setdefault(w, []).append(j)
    pairs = []
    for i in range(len(hyp) - 1, -1, -1):
        cands = set(synonyms(hyp[i][1]) or ()) | {hyp[i][1]}
        best_j, best_w = -1, None
        for s in cands:
            pos = positions.get(s)
            if pos and pos[-1] > best_j:
                best_j, best_w = pos[-1], s
        if best_w is not None:
            positions[best_w].pop()
            pairs.append((hyp[i][0], ref[best_j][0]))
    return pairs


def _meteor_single(
    hyp: list[str],
    ref: list[str],
    stemmer: Optional[Callable[[str], str]],
    synonyms: Optional[Callable[[str], Any]],
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    h = list(enumerate(w.lower() for w in hyp))
    r = list(enumerate(w.lower() for w in ref))
    n_h, n_r = len(h), len(r)
    matches, h, r = _meteor_match(h, r)
    if stemmer is not None:
        # As NLTK, the unmatched words go on to the synonym stage in their stemmed form.
        stem_pairs, h, r = _meteor_match([(i, stemmer(w)) for i, w in h], [(j, stemmer(w)) for j, w in r])
        matches += stem_pairs
    if synonyms is not None:
        matches += _meteor_synonym_match(h, r, synonyms)
    matches.sort(key=lambda m: m[0])
    m = len(matches)
    if m == 0 or n_h == 0 or n_r == 0:
        return 0.0
    precision, recall = m / n_h, m / n_r
    fmean = precision * recall / (alpha * precision + (1 - alpha) * recall)
    chunks = 1
    for a, b in zip(matches, matches[1:]):
        if not (b[0] == a[0] + 1 and b[1] == a[1] + 1):
            chunks += 1
    return float((1 - gamma * (chunks / m) ** beta) * fmean)


@register(
    category=_C,
    task="generation",
    name="METEOR",
    definition="Unigram alignment between prediction and reference by exact, stemmed and (optionally) synonym "
    "matches; harmonic mean weighted towards recall, penalised for fragmented alignments. Best reference per "
    "example, averaged over examples (NLTK's meteor_score).",
    formula="(1 − γ (chunks/m)^β) · P·R / (α P + (1 − α) R)",
    range="[0, 1]",
    input_requirements=("references", "predictions"),
    references=(_REF_BANERJEE, _REF_METEOR),
)
@per_item
def meteor(
    references: Any,
    predictions: Any,
    *,
    stemmer: Any = "porter",
    synonyms: Optional[Callable[[str], Any]] = None,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
    tokenize: Any = "none",
    average: Optional[str] = "mean",
) -> MetricResult:
    """METEOR as in NLTK ``meteor_score`` (lowercased, whitespace tokens by default).

    ``stemmer="porter"`` uses NLTK's Porter stemmer when NLTK is installed (``None`` disables stemming, or pass
    any callable str -> str). WordNet synonyms need a corpus download, so they are off unless you pass
    ``synonyms``: a callable returning the synonyms of a word, e.g.
    ``lambda w: {l.name() for s in wordnet.synsets(w) for l in s.lemmas() if "_" not in l.name()}``."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    tok = _tokenizer(tokenize)
    stem_fn: Optional[Callable[[str], str]]
    if stemmer == "porter":
        try:
            from nltk.stem.porter import PorterStemmer
        except ImportError as exc:
            from ..core.exceptions import OptionalDependencyError

            raise OptionalDependencyError(
                "nltk", "llm", "METEOR's default Porter stemmer (or pass stemmer=None)"
            ) from exc
        stem_fn = PorterStemmer().stem
    elif stemmer is None or callable(stemmer):
        stem_fn = stemmer
    else:
        raise InputValidationError("stemmer must be 'porter', None or a callable str -> str.")
    if synonyms is not None and not callable(synonyms):
        raise InputValidationError("synonyms must be a callable returning the synonyms of a word.")
    for name, v in (("alpha", alpha), ("gamma", gamma)):
        if not 0 <= v <= 1:
            raise InputValidationError(f"{name} must be in [0, 1].")
    scores = [
        max(_meteor_single(tok(p), tok(r), stem_fn, synonyms, alpha, beta, gamma) for r in rs)
        for p, rs in zip(preds, refs)
    ]
    arr = np.array(scores)
    value: Any = arr if average is None else float(arr.mean())
    return MetricResult(
        "meteor",
        "METEOR",
        value,
        {
            "stemmer": stem_fn is not None,
            "synonyms": synonyms is not None,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
    )


# ---------------------------------------------------------------- CIDEr-D
_REF_CIDER = (
    "Vedantam R, Zitnick CL, Parikh D. CIDEr: consensus-based image description evaluation. CVPR. 2015:4566-4575."
)


def _cider_counts(words: list[str], n: int) -> Counter[tuple[str, ...]]:
    c: Counter[tuple[str, ...]] = Counter()
    for k in range(1, n + 1):
        c.update(word_ngrams(words, k))
    return c


@register(
    category=_C,
    task="captioning",
    name="CIDEr-D",
    definition="Consensus with several references: cosine similarity of TF-IDF weighted n-gram vectors "
    "(n = 1..4), clipped to the reference counts and damped by a Gaussian length penalty; IDF comes from the "
    "references of the whole evaluated set (the coco-caption implementation).",
    formula="10 · mean_n mean_refs [Σ min(g_h, g_r)·g_r / (‖g_h‖‖g_r‖)] · exp(−Δ²/2σ²)",
    range="[0, 10]",
    input_requirements=("references", "predictions"),
    references=(_REF_CIDER,),
)
@per_item
def cider(
    references: Any,
    predictions: Any,
    *,
    max_order: int = 4,
    sigma: float = 6.0,
    tokenize: Any = "none",
    lowercase: bool = False,
    average: Optional[str] = "mean",
) -> MetricResult:
    """CIDEr-D identical to ``pycocoevalcap`` (on whitespace tokens; the COCO pipeline first applies its PTB
    tokenizer and lowercasing, which you can do beforehand). Needs the whole evaluation set, because the
    document frequencies come from all references; scores of one image depend on the others."""
    preds = as_texts(predictions, "predictions")
    refs = as_references(references, len(preds))
    n = check_positive_int(max_order, "max_order")
    if sigma <= 0:
        raise InputValidationError("sigma must be positive.")
    tok = _tokenizer(tokenize)

    def words(s: str) -> list[str]:
        return tok(s.lower() if lowercase else s)

    ctest = [_cider_counts(words(p), n) for p in preds]
    crefs = [[_cider_counts(words(r), n) for r in rs] for rs in refs]
    df: Counter[tuple[str, ...]] = Counter()
    for rs in crefs:
        df.update({g for r in rs for g in r})
    ref_len = math.log(float(len(crefs)))

    def vec(cnts: Counter[tuple[str, ...]]) -> tuple[list[dict[tuple[str, ...], float]], list[float], int]:
        v: list[dict[tuple[str, ...], float]] = [{} for _ in range(n)]
        norm = [0.0] * n
        length = 0
        for g, tf in cnts.items():
            k = len(g) - 1
            w = float(tf) * (ref_len - math.log(max(1.0, df.get(g, 0))))
            v[k][g] = w
            norm[k] += w * w
            if k == 1:  # pycocoevalcap counts bigrams as the "length" (kept for identical scores)
                length += tf
        return v, [math.sqrt(x) for x in norm], length

    scores = []
    for t, rs in zip(ctest, crefs):
        vh, nh, lh = vec(t)
        total = np.zeros(n)
        for r in rs:
            vr, nr, lr = vec(r)
            damp = math.exp(-((lh - lr) ** 2) / (2 * sigma**2))
            for k in range(n):
                val = sum(min(w, vr[k].get(g, 0.0)) * vr[k].get(g, 0.0) for g, w in vh[k].items())
                if nh[k] != 0 and nr[k] != 0:
                    val /= nh[k] * nr[k]
                total[k] += val * damp
        scores.append(float(np.mean(total)) / len(rs) * 10.0)
    arr = np.array(scores)
    value: Any = arr if average is None else float(arr.mean())
    return MetricResult("cider", "CIDEr-D", value, {"max_order": n, "sigma": sigma})
