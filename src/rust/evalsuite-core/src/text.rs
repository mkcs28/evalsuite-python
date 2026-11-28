//! BLEU sufficient statistics, chrF / chrF++ statistics, ROUGE-N overlaps and longest common subsequences.

use crate::hash::{ngram_counts, FxMap};

/// `[hyp_len, closest_ref_len, correct_1..N, total_1..N]` for one segment, as sacreBLEU (`_bleu_stats`).
/// The closest reference length is the one nearest the hypothesis length, the shorter one on ties.
pub fn bleu_stats(hyp: &[u32], refs: &[&[u32]], order: usize) -> Vec<i64> {
    let hyp_len = hyp.len() as i64;
    let mut closest = i64::MAX;
    let mut best_key = (i64::MAX, i64::MAX);
    for r in refs {
        let rl = r.len() as i64;
        let key = ((hyp_len - rl).abs(), rl);
        if key < best_key {
            best_key = key;
            closest = rl;
        }
    }
    let mut out = vec![0i64; 2 + 2 * order];
    out[0] = hyp_len;
    out[1] = if refs.is_empty() { 0 } else { closest };
    // Short segments (the usual case): sorted n-gram runs merged against each reference, which avoids hash
    // maps and their allocations; long ones: hash counting. Counts are integers, so both give the same result.
    let short = hyp.len() <= 256 && refs.iter().all(|r| r.len() <= 256);
    for n in 1..=order {
        let correct = if short {
            clipped_sorted(hyp, refs, n)
        } else {
            clipped_hashed(hyp, refs, n)
        };
        out[1 + n] = correct;
        out[1 + order + n] = (hyp_len - n as i64 + 1).max(0);
    }
    out
}

/// Runs of equal n-grams in sorted order: (n-gram, count).
fn sorted_runs(seq: &[u32], n: usize) -> Vec<(&[u32], u32)> {
    if n == 0 || seq.len() < n {
        return Vec::new();
    }
    let mut w: Vec<&[u32]> = seq.windows(n).collect();
    w.sort_unstable();
    let mut runs: Vec<(&[u32], u32)> = Vec::with_capacity(w.len());
    for g in w {
        match runs.last_mut() {
            Some((last, c)) if *last == g => *c += 1,
            _ => runs.push((g, 1)),
        }
    }
    runs
}

fn clipped_sorted(hyp: &[u32], refs: &[&[u32]], n: usize) -> i64 {
    // n-grams packed into one integer when the ids fit (16 bits each, n ≤ 4): sorting integers is much
    // cheaper than sorting slices; otherwise slices.
    let fits = n <= 4
        && hyp
            .iter()
            .chain(refs.iter().flat_map(|r| r.iter()))
            .all(|&t| t < 1 << 16);
    if fits {
        let pack = |s: &[u32]| -> Vec<u64> {
            let mut v: Vec<u64> = if s.len() < n {
                Vec::new()
            } else {
                s.windows(n)
                    .map(|w| w.iter().fold(0u64, |k, &t| (k << 16) | u64::from(t)))
                    .collect()
            };
            v.sort_unstable();
            v
        };
        return clipped_merge(&runs_of(&pack(hyp)), refs.iter().map(|r| runs_of(&pack(r))));
    }
    let h = sorted_runs(hyp, n);
    clipped_merge(&h, refs.iter().map(|r| sorted_runs(r, n)))
}

fn runs_of(sorted: &[u64]) -> Vec<(u64, u32)> {
    let mut runs: Vec<(u64, u32)> = Vec::with_capacity(sorted.len());
    for &g in sorted {
        match runs.last_mut() {
            Some((last, c)) if *last == g => *c += 1,
            _ => runs.push((g, 1)),
        }
    }
    runs
}

fn clipped_merge<K: Ord>(h: &[(K, u32)], refs: impl Iterator<Item = Vec<(K, u32)>>) -> i64 {
    if h.is_empty() {
        return 0;
    }
    let mut max_ref = vec![0u32; h.len()];
    for rr in refs {
        let (mut i, mut j) = (0, 0);
        while i < h.len() && j < rr.len() {
            match h[i].0.cmp(&rr[j].0) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    max_ref[i] = max_ref[i].max(rr[j].1);
                    i += 1;
                    j += 1;
                }
            }
        }
    }
    h.iter()
        .zip(&max_ref)
        .map(|((_, c), m)| i64::from((*c).min(*m)))
        .sum()
}

fn clipped_hashed(hyp: &[u32], refs: &[&[u32]], n: usize) -> i64 {
    // Max count of each n-gram over the references, then matches clipped by it.
    let mut ref_max: FxMap<&[u32], u32> = FxMap::default();
    for r in refs {
        for (g, c) in ngram_counts(r, n) {
            let e = ref_max.entry(g).or_insert(0);
            if c > *e {
                *e = c;
            }
        }
    }
    let mut correct = 0i64;
    for (g, c) in ngram_counts(hyp, n) {
        if let Some(&m) = ref_max.get(g) {
            correct += i64::from(c.min(m));
        }
    }
    correct
}

/// Clipped n-gram overlap between a reference `t` and a prediction `p` (ROUGE-N numerator).
pub fn ngram_overlap(t: &[u32], p: &[u32], n: usize) -> i64 {
    let pn = ngram_counts(p, n);
    let mut inter = 0i64;
    for (g, c) in ngram_counts(t, n) {
        if let Some(&pc) = pn.get(g) {
            inter += i64::from(c.min(pc));
        }
    }
    inter
}

/// Length of the longest common subsequence (two rolling rows).
pub fn lcs_length(a: &[u32], b: &[u32]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    let mut row = vec![0usize; b.len() + 1];
    for &ai in a {
        row[0] = 0;
        let mut left = 0usize;
        for (j, &bj) in b.iter().enumerate() {
            left = if ai == bj {
                prev[j] + 1
            } else if prev[j + 1] > left {
                prev[j + 1]
            } else {
                left
            };
            row[j + 1] = left;
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[b.len()]
}

/// Indices into `reference` of one longest common subsequence with `candidate`, in the order rouge-score's
/// `_lcs_ind` backtracks them (from the end, preferring to move along the candidate on ties).
pub fn lcs_indices(reference: &[u32], candidate: &[u32]) -> Vec<usize> {
    let (m, n) = (reference.len(), candidate.len());
    let w = n + 1;
    let mut t = vec![0u32; (m + 1) * w];
    for i in 1..=m {
        let ai = reference[i - 1];
        for j in 1..=n {
            t[i * w + j] = if ai == candidate[j - 1] {
                t[(i - 1) * w + j - 1] + 1
            } else {
                t[(i - 1) * w + j].max(t[i * w + j - 1])
            };
        }
    }
    let (mut i, mut j) = (m, n);
    let mut out = Vec::new();
    while i > 0 && j > 0 {
        if reference[i - 1] == candidate[j - 1] {
            out.push(i - 1);
            i -= 1;
            j -= 1;
        } else if t[i * w + j - 1] > t[(i - 1) * w + j] {
            j -= 1;
        } else {
            i -= 1;
        }
    }
    out
}

/// ROUGE-Lsum hit count (rouge-score's summary-level LCS): union LCS per reference sentence, each token counted
/// at most as often as it occurs in both texts. Returns (hits, reference tokens, prediction tokens).
pub fn rouge_lsum_hits(ref_sents: &[&[u32]], pred_sents: &[&[u32]]) -> (i64, i64, i64) {
    let m: usize = ref_sents.iter().map(|s| s.len()).sum();
    let n: usize = pred_sents.iter().map(|s| s.len()).sum();
    if ref_sents.is_empty() || pred_sents.is_empty() || m == 0 || n == 0 {
        return (0, m as i64, n as i64);
    }
    let mut cr: FxMap<u32, i64> = FxMap::default();
    let mut cc: FxMap<u32, i64> = FxMap::default();
    for s in ref_sents {
        for &t in *s {
            *cr.entry(t).or_insert(0) += 1;
        }
    }
    for s in pred_sents {
        for &t in *s {
            *cc.entry(t).or_insert(0) += 1;
        }
    }
    let mut hits = 0i64;
    for r in ref_sents {
        let mut union: Vec<usize> = pred_sents.iter().flat_map(|c| lcs_indices(r, c)).collect();
        union.sort_unstable();
        union.dedup();
        for i in union {
            let tok = r[i];
            let a = cc.get(&tok).copied().unwrap_or(0);
            let b = cr.get(&tok).copied().unwrap_or(0);
            if a > 0 && b > 0 {
                hits += 1;
                *cc.get_mut(&tok).unwrap() -= 1;
                *cr.get_mut(&tok).unwrap() -= 1;
            }
        }
    }
    (hits, m as i64, n as i64)
}

/// sacreBLEU's chrF F-score from accumulated statistics `[n_hyp, n_ref, n_match] * order` (`_chrf_f`).
pub fn chrf_f(stats: &[f64], order: usize, beta: f64, eps_smoothing: bool) -> f64 {
    let eps = 1e-16;
    let factor = beta * beta;
    let mut score = 0.0;
    let (mut avg_p, mut avg_r) = (0.0f64, 0.0f64);
    let mut eff = 0u32;
    for i in 0..order {
        let (n_hyp, n_ref, n_match) = (stats[3 * i], stats[3 * i + 1], stats[3 * i + 2]);
        let p = if n_hyp > 0.0 { n_match / n_hyp } else { eps };
        let r = if n_ref > 0.0 { n_match / n_ref } else { eps };
        let denom = factor * p + r;
        score += if denom > 0.0 {
            (1.0 + factor) * p * r / denom
        } else {
            eps
        };
        if n_hyp > 0.0 && n_ref > 0.0 {
            avg_p += p;
            avg_r += r;
            eff += 1;
        }
    }
    if eps_smoothing {
        return 100.0 * score / order as f64;
    }
    if eff == 0 {
        return 0.0;
    }
    avg_p /= f64::from(eff);
    avg_r /= f64::from(eff);
    if avg_p + avg_r != 0.0 {
        100.0 * (1.0 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)
    } else {
        0.0
    }
}

/// One text prepared for chrF: its characters (code points, whitespace already removed or kept by the caller)
/// and its chrF++ words (ids).
pub struct ChrfText<'a> {
    pub chars: &'a [u32],
    pub words: &'a [u32],
}

fn chrf_counts<'a>(
    t: &ChrfText<'a>,
    char_order: usize,
    word_order: usize,
) -> Vec<FxMap<&'a [u32], u32>> {
    let mut out = Vec::with_capacity(char_order + word_order);
    for n in 1..=char_order {
        out.push(ngram_counts(t.chars, n));
    }
    for n in 1..=word_order {
        out.push(ngram_counts(t.words, n));
    }
    out
}

/// chrF statistics of one example against its best reference (highest F-score, first on ties), as sacreBLEU.
pub fn chrf_segment(
    hyp: &ChrfText<'_>,
    refs: &[ChrfText<'_>],
    char_order: usize,
    word_order: usize,
    beta: f64,
    eps_smoothing: bool,
) -> Option<Vec<f64>> {
    let order = char_order + word_order;
    let h = chrf_counts(hyp, char_order, word_order);
    let h_tot: Vec<f64> = h
        .iter()
        .map(|m| m.values().map(|&c| f64::from(c)).sum())
        .collect();
    let mut best: Option<Vec<f64>> = None;
    let mut best_f = -1.0;
    for r in refs {
        let rc = chrf_counts(r, char_order, word_order);
        let mut seg = Vec::with_capacity(3 * order);
        for (k, (hm, rm)) in h.iter().zip(rc.iter()).enumerate() {
            let (small, big) = if hm.len() <= rm.len() {
                (hm, rm)
            } else {
                (rm, hm)
            };
            let mut matched = 0u64;
            for (g, &c) in small {
                if let Some(&d) = big.get(g) {
                    matched += u64::from(c.min(d));
                }
            }
            let r_tot: f64 = rm.values().map(|&c| f64::from(c)).sum();
            seg.push(if rm.is_empty() { 0.0 } else { h_tot[k] });
            seg.push(r_tot);
            seg.push(matched as f64);
        }
        let f = chrf_f(&seg, order, beta, eps_smoothing);
        if f > best_f {
            best_f = f;
            best = Some(seg);
        }
    }
    best
}

/// Python's built-in `sum()` over floats, exactly: Python >= 3.12 uses Neumaier's compensated summation
/// (CPython `builtin_sum_impl`), earlier versions add sequentially. `compensated` selects which.
pub struct PySum {
    total: f64,
    c: f64,
    compensated: bool,
}

impl PySum {
    pub fn new(compensated: bool) -> Self {
        PySum {
            total: 0.0,
            c: 0.0,
            compensated,
        }
    }

    #[inline]
    pub fn add(&mut self, x: f64) {
        if self.compensated {
            let t = self.total + x;
            if self.total.abs() >= x.abs() {
                self.c += (self.total - t) + x;
            } else {
                self.c += (x - t) + self.total;
            }
            self.total = t;
        } else {
            self.total += x;
        }
    }

    pub fn value(&self) -> f64 {
        // As CPython: the compensation is applied only when non-zero and finite.
        if self.compensated && self.c != 0.0 && self.c.is_finite() {
            self.total + self.c
        } else {
            self.total
        }
    }
}

/// Counts of every n-gram (orders `1..=n`) of `seq`, in first-occurrence order: all unigrams by first
/// occurrence, then all bigrams, and so on, as a Python `Counter` updated order by order iterates them.
fn ordered_counts(seq: &[u32], n: usize) -> Vec<(&[u32], u32)> {
    let mut out: Vec<(&[u32], u32)> = Vec::new();
    for k in 1..=n {
        if seq.len() < k {
            break;
        }
        let mut index: FxMap<&[u32], usize> = FxMap::default();
        for w in seq.windows(k) {
            match index.get(w) {
                Some(&i) => out[i].1 += 1,
                None => {
                    index.insert(w, out.len());
                    out.push((w, 1));
                }
            }
        }
    }
    out
}

struct CiderVec<'a> {
    /// Per order: (n-gram, weight) in first-occurrence order, plus a lookup.
    weights: Vec<Vec<(&'a [u32], f64)>>,
    lookup: Vec<FxMap<&'a [u32], f64>>,
    norms: Vec<f64>,
    length: i64,
}

fn cider_vec<'a>(
    counts: &[(&'a [u32], u32)],
    n: usize,
    df: &FxMap<&[u32], u32>,
    ref_len: f64,
) -> CiderVec<'a> {
    let mut weights: Vec<Vec<(&'a [u32], f64)>> = vec![Vec::new(); n];
    let mut lookup: Vec<FxMap<&'a [u32], f64>> = vec![FxMap::default(); n];
    let mut sq = vec![0.0f64; n];
    let mut length = 0i64;
    for &(g, tf) in counts {
        let k = g.len() - 1;
        let d = f64::from(df.get(g).copied().unwrap_or(0));
        let w = f64::from(tf) * (ref_len - d.max(1.0).ln());
        weights[k].push((g, w));
        lookup[k].insert(g, w);
        sq[k] += w * w;
        if k == 1 {
            // pycocoevalcap uses the bigram count as the sentence "length" (kept for identical scores)
            length += i64::from(tf);
        }
    }
    CiderVec {
        weights,
        lookup,
        norms: sq.into_iter().map(f64::sqrt).collect(),
        length,
    }
}

#[allow(clippy::needless_range_loop)] // `k` indexes several per-order arrays at once
/// CIDEr-D per prediction, before the final mean: for each prediction, the `n` per-order totals summed over its
/// references (pycocoevalcap / EvalSuite's `cider`). The caller averages them and scales by 10 / len(refs).
pub fn cider_totals(
    hyps: &[&[u32]],
    refs: &[Vec<&[u32]>],
    n: usize,
    sigma: f64,
    compensated: bool,
) -> Vec<Vec<f64>> {
    use rayon::prelude::*;

    let ctest: Vec<Vec<(&[u32], u32)>> = hyps.iter().map(|h| ordered_counts(h, n)).collect();
    type Counts<'c> = Vec<(&'c [u32], u32)>;
    let crefs: Vec<Vec<Counts<'_>>> = refs
        .iter()
        .map(|rs| rs.iter().map(|r| ordered_counts(r, n)).collect())
        .collect();
    // Document frequency: in how many images' reference sets each n-gram occurs.
    let mut df: FxMap<&[u32], u32> = FxMap::default();
    for rs in &crefs {
        let mut seen: FxMap<&[u32], ()> = FxMap::default();
        for r in rs {
            for &(g, _) in r {
                seen.insert(g, ());
            }
        }
        for g in seen.into_keys() {
            *df.entry(g).or_insert(0) += 1;
        }
    }
    let ref_len = (crefs.len() as f64).ln();
    let two_s2 = 2.0 * (sigma * sigma);
    ctest
        .par_iter()
        .zip(crefs.par_iter())
        .with_min_len(16)
        .map(|(t, rs)| {
            let vh = cider_vec(t, n, &df, ref_len);
            let mut total = vec![0.0f64; n];
            for r in rs {
                let vr = cider_vec(r, n, &df, ref_len);
                let dl = (vh.length - vr.length) as f64;
                let damp = (-(dl * dl) / two_s2).exp();
                for k in 0..n {
                    let mut acc = PySum::new(compensated);
                    for &(g, w) in &vh.weights[k] {
                        let rv = vr.lookup[k].get(g).copied().unwrap_or(0.0);
                        acc.add(w.min(rv) * rv);
                    }
                    let mut val = acc.value();
                    if vh.norms[k] != 0.0 && vr.norms[k] != 0.0 {
                        val /= vh.norms[k] * vr.norms[k];
                    }
                    total[k] += val * damp;
                }
            }
            total
        })
        .collect()
}

/// CodeBLEU's BLEU statistics for one example: per order `n = 1..=4`, the (numerator, denominator) terms in the
/// order the reference implementation adds them (one term per order; with keyword weighting, one per reference),
/// plus the hypothesis length and the closest reference length.
pub fn codebleu_example(
    hyp: &[u32],
    refs: &[&[u32]],
    keyword: Option<&[bool]>,
    compensated: bool,
) -> (Vec<Vec<(f64, f64)>>, i64, i64) {
    let mut terms: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 4];
    for n in 1..=4usize {
        let hc = ngram_counts(hyp, n);
        match keyword {
            None => {
                let mut maxc: FxMap<&[u32], u32> = FxMap::default();
                for r in refs {
                    for (g, c) in ngram_counts(r, n) {
                        let e = maxc.entry(g).or_insert(0);
                        *e = (*e).max(c);
                    }
                }
                let num: u64 = hc
                    .iter()
                    .map(|(g, &c)| u64::from(c.min(maxc.get(g).copied().unwrap_or(0))))
                    .sum();
                let total: u64 = hc.values().map(|&c| u64::from(c)).sum();
                terms[n - 1].push((num as f64, total.max(1) as f64));
            }
            Some(kw) => {
                for r in refs {
                    // Reference n-grams in first-occurrence order (the order the reference implementation sums them).
                    let rc: Vec<(&[u32], u32)> = ordered_counts(r, n)
                        .into_iter()
                        .filter(|(g, _)| g.len() == n)
                        .collect();
                    if n == 1 {
                        let wt = |g: &[u32]| if kw[g[0] as usize] { 1.0 } else { 0.2 };
                        let mut num = PySum::new(compensated);
                        let mut den = PySum::new(compensated);
                        for &(g, c) in &rc {
                            let clipped = c.min(hc.get(g).copied().unwrap_or(0));
                            num.add(f64::from(clipped) * wt(g));
                            den.add(f64::from(c) * wt(g));
                        }
                        let d = den.value();
                        terms[0].push((num.value(), if d < 1.0 { 1.0 } else { d }));
                    } else {
                        let num: u64 = rc
                            .iter()
                            .map(|&(g, c)| u64::from(c.min(hc.get(g).copied().unwrap_or(0))))
                            .sum();
                        let den: u64 = rc.iter().map(|&(_, c)| u64::from(c)).sum();
                        terms[n - 1].push((num as f64, den.max(1) as f64));
                    }
                }
            }
        }
    }
    let hl = hyp.len() as i64;
    let closest = refs
        .iter()
        .map(|r| r.len() as i64)
        .min_by_key(|&rl| ((rl - hl).abs(), rl))
        .unwrap_or(0);
    (terms, hl, closest)
}

/// (distinct n-grams, total n-grams) over all sequences.
pub fn distinct_ngrams(seqs: &[&[u32]], n: usize) -> (u64, u64) {
    let mut seen: FxMap<&[u32], ()> = FxMap::default();
    let mut total = 0u64;
    for s in seqs {
        if n == 0 || s.len() < n {
            continue;
        }
        for w in s.windows(n) {
            seen.insert(w, ());
            total += 1;
        }
    }
    (seen.len() as u64, total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bleu_counts() {
        let h = [1, 2, 3, 4];
        let r1 = [1, 2, 3, 5];
        let s = bleu_stats(&h, &[&r1], 2);
        assert_eq!(s, vec![4, 4, 3, 2, 4, 3]);
    }

    #[test]
    fn lcs() {
        assert_eq!(lcs_length(&[1, 2, 3, 4], &[2, 4, 3]), 2);
        assert_eq!(lcs_length(&[], &[1]), 0);
    }
}
