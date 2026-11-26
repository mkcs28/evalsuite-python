//! Survival analysis: Harrell's concordance index in O(n log n) with a Fenwick tree.

/// `(Σ w_i (concordant_i + ties_i / 2), Σ w_i comparable_i, comparable pairs)` over events `i`, where `j` is
/// comparable to event `i` when `t_j > t_i`, or `t_j == t_i` with `j` censored; risks within `tol` are ties.
/// Subjects are processed in decreasing time (stable), censored subjects at a time before its events, exactly
/// as EvalSuite's Python implementation, so the sums are accumulated in the same order and bit-identical.
pub fn concordance(t: &[f64], event: &[u8], risk: &[f64], w: &[f64], tol: f64) -> (f64, f64, u64) {
    let n = t.len();
    let mut order_r: Vec<f64> = risk.to_vec();
    order_r.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let below = |x: f64| order_r.partition_point(|&v| v < x);
    let at_most = |x: f64| order_r.partition_point(|&v| v <= x);
    let slot: Vec<usize> = risk.iter().map(|&r| below(r)).collect();
    let mut order_t: Vec<usize> = (0..n).collect();
    order_t.sort_by(|&a, &b| {
        (-t[a])
            .partial_cmp(&-t[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    }); // stable, as NumPy's
    let mut tree = vec![0u64; n + 1];
    let add = |tree: &mut Vec<u64>, pos: usize| {
        let mut i = pos + 1;
        while i <= n {
            tree[i] += 1;
            i += i & i.wrapping_neg();
        }
    };
    let prefix = |tree: &Vec<u64>, pos: usize| {
        let mut i = pos;
        let mut s = 0u64;
        while i > 0 {
            s += tree[i];
            i -= i & i.wrapping_neg();
        }
        s
    };
    let (mut num, mut den, mut pairs, mut inserted) = (0.0f64, 0.0f64, 0u64, 0u64);
    let mut i = 0;
    while i < n {
        let ti = t[order_t[i]];
        let mut j = i;
        while j < n && t[order_t[j]] == ti {
            j += 1;
        }
        let group = &order_t[i..j];
        for &k in group.iter().filter(|&&k| event[k] == 0) {
            add(&mut tree, slot[k]);
            inserted += 1;
        }
        for &k in group.iter().filter(|&&k| event[k] != 0) {
            let less = prefix(&tree, below(risk[k] - tol));
            let ties = prefix(&tree, at_most(risk[k] + tol)) - less;
            if inserted > 0 {
                num += w[k] * (less as f64 + 0.5 * ties as f64);
                den += w[k] * inserted as f64;
                pairs += inserted;
            }
        }
        for &k in group.iter().filter(|&&k| event[k] != 0) {
            add(&mut tree, slot[k]);
            inserted += 1;
        }
        i = j;
    }
    (num, den, pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_example() {
        // times 1..4, all events, risk decreasing with time: every comparable pair concordant
        let (num, den, pairs) = concordance(
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1, 1, 1],
            &[4.0, 3.0, 2.0, 1.0],
            &[1.0; 4],
            1e-8,
        );
        assert_eq!((num, den, pairs), (6.0, 6.0, 6));
    }
}
