//! Counting outcome codes over bootstrap resamples without materialising the resampled arrays.

/// For each of the `rows` resamples (`indices` is `rows × n`, row-major), the number of times each code
/// `0..k` occurs among `codes[indices[row, ..]]`. Returns `rows × k` counts, row-major.
/// Indices must be in `0..codes.len()` (checked; an out-of-range index returns `None`).
pub fn code_counts(codes: &[u32], k: usize, indices: &[i64], rows: usize) -> Option<Vec<u64>> {
    if rows == 0 {
        return Some(Vec::new());
    }
    let n = indices.len() / rows;
    if n * rows != indices.len() || codes.iter().any(|&c| c as usize >= k) {
        return None;
    }
    let len = codes.len() as i64;
    let mut out = vec![0u64; rows * k];
    for (r, idx) in indices.chunks_exact(n).enumerate() {
        let counts = &mut out[r * k..(r + 1) * k];
        for &i in idx {
            if i < 0 || i >= len {
                return None;
            }
            counts[codes[i as usize] as usize] += 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
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
