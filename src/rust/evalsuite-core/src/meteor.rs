//! METEOR for one hypothesis/reference pair with exact and stem matching, as NLTK's `meteor_score`
//! (and EvalSuite's Python reference `_meteor_single`). Words and stems are integer ids.

use crate::hash::FxMap;

/// NLTK's matcher: each hypothesis word, from the end, takes the latest unused identical reference word.
/// `hyp` and `reference` are (original position, key) pairs; returns matched (hyp position, ref position)
/// pairs and the unmatched entries of each side, in order.
#[allow(clippy::type_complexity)]
fn match_words(
    hyp: &[(usize, u32)],
    reference: &[(usize, u32)],
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let mut positions: FxMap<u32, Vec<usize>> = FxMap::default();
    for (j, &(_, w)) in reference.iter().enumerate() {
        positions.entry(w).or_default().push(j);
    }
    let mut pairs = Vec::new();
    let mut used_h = vec![false; hyp.len()];
    let mut used_r = vec![false; reference.len()];
    for i in (0..hyp.len()).rev() {
        if let Some(pos) = positions.get_mut(&hyp[i].1) {
            if let Some(j) = pos.pop() {
                used_h[i] = true;
                used_r[j] = true;
                pairs.push((hyp[i].0, reference[j].0));
            }
        }
    }
    let rest_h = (0..hyp.len()).filter(|&i| !used_h[i]).collect();
    let rest_r = (0..reference.len()).filter(|&j| !used_r[j]).collect();
    (pairs, rest_h, rest_r)
}

/// METEOR score of one pair. `hyp_words`/`ref_words` are lower-cased word ids; `hyp_stems`/`ref_stems` the ids of
/// their stems (same length), or `None` to skip the stem stage.
pub fn meteor_single(
    hyp_words: &[u32],
    ref_words: &[u32],
    stems: Option<(&[u32], &[u32])>,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> f64 {
    let (n_h, n_r) = (hyp_words.len(), ref_words.len());
    let h: Vec<(usize, u32)> = hyp_words.iter().copied().enumerate().collect();
    let r: Vec<(usize, u32)> = ref_words.iter().copied().enumerate().collect();
    let (mut matches, rest_h, rest_r) = match_words(&h, &r);
    if let Some((hs, rs)) = stems {
        // As NLTK, the words left unmatched go on to the stem stage, compared by their stems.
        let h2: Vec<(usize, u32)> = rest_h.iter().map(|&k| (h[k].0, hs[h[k].0])).collect();
        let r2: Vec<(usize, u32)> = rest_r.iter().map(|&k| (r[k].0, rs[r[k].0])).collect();
        let (stem_pairs, _, _) = match_words(&h2, &r2);
        matches.extend(stem_pairs);
    }
    matches.sort_by_key(|m| m.0); // hypothesis positions are unique, so this order is total
    let m = matches.len();
    if m == 0 || n_h == 0 || n_r == 0 {
        return 0.0;
    }
    let precision = m as f64 / n_h as f64;
    let recall = m as f64 / n_r as f64;
    let fmean = precision * recall / (alpha * precision + (1.0 - alpha) * recall);
    let mut chunks = 1usize;
    for w in matches.windows(2) {
        let (a, b) = (w[0], w[1]);
        if !(b.0 == a.0 + 1 && b.1 == a.1 + 1) {
            chunks += 1;
        }
    }
    (1.0 - gamma * (chunks as f64 / m as f64).powf(beta)) * fmean
}
