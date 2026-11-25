//! Shared result shapes of the LLM-system metrics: a proportion with its 95% Wilson interval ([`Rate`]) and
//! a distribution summary ([`Summary`]), as EvalSuite's Python `evalsuite.llmsys._common`.

use crate::error::{invalid, Result};
use crate::util::{np_sum, percentile, sorted};

const Z95: f64 = 1.959963984540054;

/// A proportion `count / n` with its 95% Wilson interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rate {
    pub value: f64,
    pub count: usize,
    pub n: usize,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// 95% Wilson interval of `k` successes in `n` trials (NaN when `n == 0`).
pub fn wilson(k: f64, n: f64) -> (f64, f64) {
    if n <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let p = k / n;
    let z2 = Z95 * Z95;
    let centre = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
    let half = Z95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / (1.0 + z2 / n);
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// The rate of `true` flags.
pub fn rate(flags: &[bool]) -> Rate {
    let n = flags.len();
    let k = flags.iter().filter(|&&f| f).count();
    let (ci_low, ci_high) = wilson(k as f64, n as f64);
    Rate {
        value: k as f64 / n as f64,
        count: k,
        n,
        ci_low,
        ci_high,
    }
}

/// Mean, sample standard deviation (ddof 1; 0 for one value), extremes and percentiles (linear interpolation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Summary {
    /// Summary of a non-empty slice.
    pub fn of(a: &[f64]) -> Summary {
        let s = sorted(a);
        let mean = np_sum(a) / a.len() as f64;
        let std = if a.len() > 1 {
            let c: Vec<f64> = a.iter().map(|x| (x - mean) * (x - mean)).collect();
            (c.iter().sum::<f64>() / (a.len() - 1) as f64).sqrt()
        } else {
            0.0
        };
        Summary {
            mean,
            std,
            min: s[0],
            max: s[s.len() - 1],
            p50: percentile(&s, 50.0),
            p90: percentile(&s, 90.0),
            p95: percentile(&s, 95.0),
            p99: percentile(&s, 99.0),
        }
    }

    /// Any percentile `q` in [0, 100] of `a` (linear interpolation).
    pub fn percentile(a: &[f64], q: f64) -> f64 {
        percentile(&sorted(a), q)
    }
}

/// Finite values, optionally bounded below / above; `name` labels errors.
pub(crate) fn floats(x: &[f64], name: &str, lo: Option<f64>, hi: Option<f64>) -> Result<()> {
    if x.is_empty() {
        return invalid(format!("{name} is empty. Provide at least one example."));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite values."));
    }
    if let Some(lo) = lo {
        if x.iter().any(|&v| v < lo) {
            return invalid(format!("{name} must be >= {lo}."));
        }
    }
    if let Some(hi) = hi {
        if x.iter().any(|&v| v > hi) {
            return invalid(format!("{name} must be <= {hi}."));
        }
    }
    Ok(())
}

pub(crate) fn nonempty<T>(x: &[T], name: &str) -> Result<()> {
    if x.is_empty() {
        return invalid(format!("{name} is empty. Provide at least one example."));
    }
    Ok(())
}

pub(crate) fn same(a: usize, b: usize, na: &str, nb: &str) -> Result<()> {
    if a != b {
        return invalid(format!(
            "{na} and {nb} must have the same length. Received {a} and {b}."
        ));
    }
    Ok(())
}

/// Mean of `values` per group label, sorted by label.
pub fn by_group<S: AsRef<str>>(
    values: &[f64],
    groups: &[S],
    name: &str,
) -> Result<Vec<(String, f64)>> {
    if groups.len() != values.len() {
        return invalid(format!(
            "{name} must have one label per example ({}); got {}.",
            values.len(),
            groups.len()
        ));
    }
    let mut labels: Vec<&str> = groups.iter().map(AsRef::as_ref).collect();
    labels.sort_unstable();
    labels.dedup();
    Ok(labels
        .into_iter()
        .map(|l| {
            let v: Vec<f64> = values
                .iter()
                .zip(groups)
                .filter(|(_, g)| g.as_ref() == l)
                .map(|(v, _)| *v)
                .collect();
            (l.to_owned(), np_sum(&v) / v.len() as f64)
        })
        .collect())
}
