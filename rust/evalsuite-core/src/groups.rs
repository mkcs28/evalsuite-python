//! Per-item agreement among a group of answers (encoded as integer ids) and per-row rank counts, for the
//! LLM-systems consistency metrics (paraphrase, cross-document, cross-lingual consistency, response stability,
//! majority vote) and memorisation exposure.

use crate::hash::FxMap;

/// Agreement within one group of answer ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupStats {
    /// Every answer equals the first.
    pub unanimous: bool,
    /// Equal unordered pairs, Σ c·(c−1)/2 over the answer counts c.
    pub equal_pairs: u64,
    /// All unordered pairs, m·(m−1)/2.
    pub pairs: u64,
    /// Count of the most common answer.
    pub modal_count: u64,
    /// The most common answer, the first seen among ties (as `Counter.most_common(1)`).
    pub mode: u32,
}

pub fn group_stats(ids: &[u32]) -> GroupStats {
    let m = ids.len() as u64;
    let mut counts: FxMap<u32, u64> = FxMap::default();
    let mut first_seen: Vec<u32> = Vec::new();
    for &x in ids {
        let c = counts.entry(x).or_insert(0);
        if *c == 0 {
            first_seen.push(x);
        }
        *c += 1;
    }
    let mut mode = ids.first().copied().unwrap_or(0);
    let mut modal_count = 0;
    let mut equal_pairs = 0;
    for &x in &first_seen {
        let c = counts[&x];
        equal_pairs += c * (c - 1) / 2;
        if c > modal_count {
            modal_count = c;
            mode = x;
        }
    }
    GroupStats {
        unanimous: first_seen.len() <= 1,
        equal_pairs,
        pairs: m * m.saturating_sub(1) / 2,
        modal_count,
        mode,
    }
}

/// For each row, how many of its values are strictly below its threshold. `None` if a value or threshold is
/// NaN or infinite.
pub fn count_below(rows: &[Vec<f64>], thresholds: &[f64]) -> Option<Vec<u64>> {
    if rows
        .iter()
        .flatten()
        .chain(thresholds)
        .any(|v| !v.is_finite())
    {
        return None;
    }
    Some(
        rows.iter()
            .zip(thresholds)
            .map(|(r, &t)| r.iter().filter(|&&v| v < t).count() as u64)
            .collect(),
    )
}

/// (NumPy `sum`, min, max, number of values > 0) of one row. An empty row gives (0, +inf, -inf, 0).
pub fn row_stats(row: &[f64]) -> (f64, f64, f64, u64) {
    let (mut lo, mut hi, mut pos) = (f64::INFINITY, f64::NEG_INFINITY, 0u64);
    for &v in row {
        lo = lo.min(v);
        hi = hi.max(v);
        pos += u64::from(v > 0.0);
    }
    (crate::curve::np_sum(row), lo, hi, pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats() {
        let s = group_stats(&[3, 1, 1, 3, 2]);
        assert_eq!(
            (s.unanimous, s.equal_pairs, s.pairs, s.modal_count, s.mode),
            (false, 2, 10, 2, 3)
        );
        assert!(group_stats(&[4, 4]).unanimous);
        assert_eq!(
            count_below(&[vec![1.0, 2.0, 3.0], vec![0.5]], &[2.5, 1.0]),
            Some(vec![2, 1])
        );
        assert_eq!(row_stats(&[0.5, -1.0, 2.0]), (1.5, -1.0, 2.0, 2));
    }
}
