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
    for n in 1..=order {
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
        out[1 + n] = correct;
        out[1 + order + n] = (hyp_len - n as i64 + 1).max(0);
    }
    out
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
