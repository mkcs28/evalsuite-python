//! Permutation tests on a vector of per-item scores: for WEAT, the statistic of a split of the items into a
//! group of size `k` and the rest is `2·Σ_group s − Σ s`. Exact enumeration (bit-identical to EvalSuite's
//! Python path, which sums each group with NumPy) and seeded random splits (opt-in; its own random stream).

use crate::curve::np_sum;
use crate::resample::Xoshiro256pp;

/// How many of the C(n, k) splits, in lexicographic order, have statistic `>= threshold`.
pub fn exact_count(s: &[f64], k: usize, total: f64, threshold: f64) -> u64 {
    let n = s.len();
    if k > n {
        return 0;
    }
    let mut c: Vec<usize> = (0..k).collect();
    let mut buf = vec![0.0f64; k];
    let mut count = 0u64;
    loop {
        for (b, &i) in buf.iter_mut().zip(&c) {
            *b = s[i];
        }
        if 2.0 * np_sum(&buf) - total >= threshold {
            count += 1;
        }
        // Next combination (lexicographic), as itertools.combinations.
        let mut i = k;
        loop {
            if i == 0 {
                return count;
            }
            i -= 1;
            if c[i] != i + n - k {
                break;
            }
            if i == 0 && c[0] == n - k {
                return count;
            }
        }
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// How many of `trials` random size-`k` groups (partial Fisher–Yates, generator seeded from (seed, trial))
/// have statistic `>= threshold`. Trials in parallel; the count depends only on the seed.
pub fn random_count(
    s: &[f64],
    k: usize,
    total: f64,
    threshold: f64,
    trials: usize,
    seed: u64,
) -> u64 {
    use rayon::prelude::*;

    let n = s.len();
    (0..trials)
        .into_par_iter()
        .map_init(
            || ((0..n).collect::<Vec<usize>>(), vec![0.0f64; k]),
            |(perm, buf), t| {
                // Every trial starts from the identity, so its draw depends only on (seed, t).
                for (i, x) in perm.iter_mut().enumerate() {
                    *x = i;
                }
                let mut g = Xoshiro256pp::new(seed, t as u64);
                for i in 0..k {
                    let j = i + g.below((n - i) as u64) as usize;
                    perm.swap(i, j);
                    buf[i] = s[perm[i]];
                }
                u64::from(2.0 * np_sum(buf) - total >= threshold)
            },
        )
        .sum()
}

#[cfg(test)]
mod tests {
    #[test]
    fn exact_enumerates_all() {
        // C(4, 2) = 6 splits; statistic = 2·(pair sum) − 10 for s = 1, 2, 3, 4.
        let s = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(super::exact_count(&s, 2, 10.0, f64::NEG_INFINITY), 6);
        assert_eq!(super::exact_count(&s, 2, 10.0, 0.0), 4); // pair sums 5, 5, 6, 7 reach 5
        assert_eq!(
            super::random_count(&s, 2, 10.0, f64::NEG_INFINITY, 50, 3),
            50
        );
    }
}
