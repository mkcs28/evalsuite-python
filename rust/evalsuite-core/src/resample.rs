//! Counting outcome codes over bootstrap resamples without materialising the resampled arrays.

/// For each of the `rows` resamples (`indices` is `rows × n`, row-major), the number of times each code
/// `0..k` occurs among `codes[indices[row, ..]]`. Returns `rows × k` counts, row-major.
/// Indices must be in `0..codes.len()` (checked; an out-of-range index returns `None`).
pub fn code_counts(codes: &[u32], k: usize, indices: &[i64], rows: usize) -> Option<Vec<u64>> {
    use rayon::prelude::*;

    if rows == 0 {
        return Some(Vec::new());
    }
    let n = indices.len() / rows;
    if n * rows != indices.len() || codes.iter().any(|&c| c as usize >= k) {
        return None;
    }
    let len = codes.len() as i64;
    if indices.iter().any(|&i| i < 0 || i >= len) {
        return None;
    }
    let mut out = vec![0u64; rows * k];
    // One resample per task; each writes only its own row of counts.
    out.par_chunks_mut(k.max(1))
        .zip(indices.par_chunks(n.max(1)))
        .for_each(|(counts, idx)| {
            for &i in idx {
                counts[codes[i as usize] as usize] += 1;
            }
        });
    Some(out)
}

/// SplitMix64 step: seeds the per-resample generators from (seed, row) so every row's stream is fixed by the
/// seed alone, whatever the number of threads.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ (Blackman & Vigna, 2019).
pub struct Xoshiro256pp([u64; 4]);

impl Xoshiro256pp {
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut mix = stream.wrapping_mul(0xD605_BBB5_8C8A_BBDB);
        let mut st = seed ^ splitmix64(&mut mix);
        Xoshiro256pp([
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
            splitmix64(&mut st),
        ])
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..bound` (`bound > 0`), unbiased (Lemire's multiply-and-reject).
    #[inline]
    pub fn below(&mut self, bound: u64) -> u64 {
        let mut m = u128::from(self.next_u64()) * u128::from(bound);
        if (m as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (m as u64) < threshold {
                m = u128::from(self.next_u64()) * u128::from(bound);
            }
        }
        (m >> 64) as u64
    }
}

/// Bootstrap indices drawn in the core: `rows × n` (row-major) indices with replacement. `groups` lists the
/// members of each stratum (concatenated, `offsets` has one more entry than there are strata); the columns of a
/// row are filled stratum by stratum, each with draws from its own members, as EvalSuite's NumPy path lays them out.
/// Without strata every index is drawn from `0..n`. Row `r` uses its own generator seeded from `(seed, r)`, so the
/// output depends only on the seed, never on the thread count.
pub fn bootstrap_indices(
    n: usize,
    rows: usize,
    seed: u64,
    strata: Option<(&[i64], &[usize])>,
) -> Vec<i64> {
    let mut out = vec![0i64; rows * n];
    fill_bootstrap_indices(&mut out, n, seed, strata);
    out
}

/// As [`bootstrap_indices`], writing into `out` (`rows × n`, row-major; `out.len()` must be a multiple of `n`).
pub fn fill_bootstrap_indices(
    out: &mut [i64],
    n: usize,
    seed: u64,
    strata: Option<(&[i64], &[usize])>,
) {
    use rayon::prelude::*;

    if n == 0 {
        return;
    }
    out.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        let mut g = Xoshiro256pp::new(seed, r as u64);
        match strata {
            None => {
                for x in row.iter_mut() {
                    *x = g.below(n as u64) as i64;
                }
            }
            Some((members, offsets)) => {
                let mut pos = 0;
                for w in offsets.windows(2) {
                    let m = &members[w[0]..w[1]];
                    for x in row[pos..pos + m.len()].iter_mut() {
                        *x = m[g.below(m.len() as u64) as usize];
                    }
                    pos += m.len();
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn bootstrap_indices_are_reproducible_and_in_range() {
        let a = super::bootstrap_indices(50, 20, 7, None);
        assert_eq!(a, super::bootstrap_indices(50, 20, 7, None));
        assert_ne!(a, super::bootstrap_indices(50, 20, 8, None));
        assert!(a.iter().all(|&i| (0..50).contains(&i)));
        let members = [1i64, 3, 0, 2, 4];
        let s = super::bootstrap_indices(5, 10, 1, Some((&members, &[0, 2, 5])));
        for row in s.chunks(5) {
            assert!(row[..2].iter().all(|i| [1, 3].contains(i)));
            assert!(row[2..].iter().all(|i| [0, 2, 4].contains(i)));
        }
    }

    #[test]
    fn counts_per_row() {
        let codes = [0u32, 1, 1, 2];
        let idx = [0i64, 1, 1, 3, 2, 2, 2, 2];
        assert_eq!(
            super::code_counts(&codes, 3, &idx, 2).unwrap(),
            vec![1, 2, 1, 0, 4, 0]
        );
        assert!(super::code_counts(&codes, 3, &[9], 1).is_none());
    }
}
