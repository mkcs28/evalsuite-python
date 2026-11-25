//! Small numeric helpers shared by the metric modules.

use crate::error::{invalid, Result};

/// NumPy's `np.sum` / `add.reduce` over a 1-D float64 array (pairwise summation).
pub fn np_sum(a: &[f64]) -> f64 {
    evalsuite_core::curve::np_sum(a)
}

/// Mean as `np.mean` (pairwise sum divided by the count).
pub fn mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// `np.average(values, weights=w)`.
pub fn weighted_mean(values: &[f64], w: &[f64]) -> f64 {
    let prod: Vec<f64> = values.iter().zip(w).map(|(v, w)| v * w).collect();
    np_sum(&prod) / np_sum(w)
}

/// Population (`ddof = 0`) or sample (`ddof = 1`) variance, as `np.var`.
pub fn var(a: &[f64], ddof: usize) -> f64 {
    let m = mean(a);
    let sq: Vec<f64> = a.iter().map(|x| (x - m) * (x - m)).collect();
    np_sum(&sq) / (a.len() - ddof) as f64
}

/// Checks that `x` is non-empty and finite.
pub fn check_finite(x: &[f64], name: &str) -> Result<()> {
    if x.is_empty() {
        return invalid(format!("{name} is empty. Provide at least one example."));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite values."));
    }
    Ok(())
}

pub fn same_len(a: usize, b: usize, what: &str) -> Result<()> {
    if a != b {
        return invalid(format!("{what} must contain the same number of observations. Received {a} and {b}."));
    }
    Ok(())
}

/// Validated sample weights (non-negative, finite, not all zero), or ones.
pub fn weights(w: Option<&[f64]>, n: usize) -> Result<Vec<f64>> {
    match w {
        None => Ok(vec![1.0; n]),
        Some(w) => {
            if w.len() != n {
                return invalid(format!(
                    "sample_weight must contain one weight per observation. Received {} for {n} observations.",
                    w.len()
                ));
            }
            if w.iter().any(|x| !x.is_finite()) {
                return invalid("sample_weight contains NaN or infinite values.");
            }
            if w.iter().any(|&x| x < 0.0) {
                return invalid("sample_weight must be non-negative.");
            }
            if np_sum(w) == 0.0 {
                return invalid("sample_weight must not sum to zero.");
            }
            Ok(w.to_vec())
        }
    }
}

/// The q-th percentile (0–100) of sorted data with linear interpolation, as `np.percentile`.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    quantile(sorted, q / 100.0)
}

/// `np.linspace(start, stop, num)` (endpoint included).
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

/// The q-th quantile (0–1) of sorted data with linear interpolation, as `np.quantile`.
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    // NumPy's `_lerp`: a + (b - a) * t, switching to b - (b - a) * (1 - t) for t >= 0.5.
    let (a, b) = (sorted[lo], sorted[hi]);
    let d = b - a;
    if frac >= 0.5 {
        b - d * (1.0 - frac)
    } else {
        a + d * frac
    }
}

/// Sorted copy of `x` (NaN-free input).
pub fn sorted(x: &[f64]) -> Vec<f64> {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}
