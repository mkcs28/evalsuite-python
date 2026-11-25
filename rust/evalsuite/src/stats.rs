//! Statistics for comparing models, as EvalSuite's Python `evalsuite.stats`: hypothesis tests, effect sizes,
//! confidence intervals, DeLong's test, McNemar's test and bootstrap intervals and tests. SciPy is the
//! reference; results agree to about 1e-12.

use crate::error::{invalid, undefined, Result};
use crate::special::{self, chi2_sf, ln_gamma, norm_cdf, norm_ppf, t_cdf, t_ppf};
use crate::util::{np_sum, quantile, sorted};

/// The alternative hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alternative {
    #[default]
    TwoSided,
    Greater,
    Less,
}

/// A test result: statistic, p-value, effect size estimate and optional confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub estimate: f64,
    pub ci: Option<(f64, f64)>,
}

/// A confidence interval around an estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub estimate: f64,
    pub low: f64,
    pub high: f64,
}

fn check_level(level: f64) -> Result<()> {
    if !(level > 0.0 && level < 1.0) {
        return invalid("level must be between 0 and 1, for example 0.95.");
    }
    Ok(())
}

fn vec_check(x: &[f64], name: &str, min_n: usize) -> Result<()> {
    if x.is_empty() {
        return invalid(format!("{name} is empty. Provide at least one example."));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite values."));
    }
    if x.len() < min_n {
        return undefined(format!(
            "{name} needs at least {min_n} observations; got {}.",
            x.len()
        ));
    }
    Ok(())
}

fn mean(x: &[f64]) -> f64 {
    np_sum(x) / x.len() as f64
}

/// Sample variance with the two-pass formula (as NumPy).
fn var1(x: &[f64]) -> f64 {
    let m = mean(x);
    let d: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    d / (x.len() - 1) as f64
}

fn t_p(t: f64, df: f64, alt: Alternative) -> f64 {
    match alt {
        Alternative::TwoSided => (2.0 * t_cdf(-t.abs(), df)).min(1.0),
        Alternative::Less => t_cdf(t, df),
        Alternative::Greater => t_cdf(-t, df),
    }
}

fn z_p(z: f64, alt: Alternative) -> f64 {
    match alt {
        Alternative::TwoSided => (2.0 * norm_cdf(-z.abs())).min(1.0),
        Alternative::Less => norm_cdf(z),
        Alternative::Greater => norm_cdf(-z),
    }
}

fn mean_diff_ci(diff: f64, se: f64, df: f64, level: f64, alt: Alternative) -> (f64, f64) {
    match alt {
        Alternative::TwoSided => {
            let q = t_ppf(1.0 - (1.0 - level) / 2.0, df);
            (diff - q * se, diff + q * se)
        }
        Alternative::Greater => (diff - t_ppf(level, df) * se, f64::INFINITY),
        Alternative::Less => (f64::NEG_INFINITY, diff + t_ppf(level, df) * se),
    }
}

/// Two-sample t-test (Welch's by default; Student's with `equal_var`). Estimate: mean difference a − b.
pub fn t_test(
    a: &[f64],
    b: &[f64],
    equal_var: bool,
    alt: Alternative,
    level: f64,
) -> Result<TestResult> {
    check_level(level)?;
    vec_check(a, "a", 2)?;
    vec_check(b, "b", 2)?;
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let (m1, m2) = (mean(a), mean(b));
    let (v1, v2) = (var1(a), var1(b));
    if v1 == 0.0 && v2 == 0.0 {
        return undefined("The t-test is undefined: both samples have zero variance.");
    }
    let (se, df) = if equal_var {
        let df = n1 + n2 - 2.0;
        let sp2 = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / df;
        ((sp2 * (1.0 / n1 + 1.0 / n2)).sqrt(), df)
    } else {
        let se = (v1 / n1 + v2 / n2).sqrt();
        let df = (v1 / n1 + v2 / n2).powi(2)
            / ((v1 / n1).powi(2) / (n1 - 1.0) + (v2 / n2).powi(2) / (n2 - 1.0));
        (se, df)
    };
    let diff = m1 - m2;
    let t = diff / se;
    Ok(TestResult {
        statistic: t,
        p_value: t_p(t, df, alt),
        estimate: diff,
        ci: Some(mean_diff_ci(diff, se, df, level, alt)),
    })
}

/// Paired t-test on a − b. Estimate: mean difference.
pub fn paired_t_test(a: &[f64], b: &[f64], alt: Alternative, level: f64) -> Result<TestResult> {
    check_level(level)?;
    vec_check(a, "a", 2)?;
    vec_check(b, "b", 2)?;
    if a.len() != b.len() {
        return invalid(format!(
            "Paired samples must have the same length; got {} and {}.",
            a.len(),
            b.len()
        ));
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = d.len() as f64;
    let diff = mean(&d);
    let sd = (d.iter().map(|v| (v - diff) * (v - diff)).sum::<f64>() / (n - 1.0)).sqrt();
    if sd == 0.0 {
        return undefined("The paired t-test is undefined: every difference is identical.");
    }
    let se = sd / n.sqrt();
    let t = diff / se;
    Ok(TestResult {
        statistic: t,
        p_value: t_p(t, n - 1.0, alt),
        estimate: diff,
        ci: Some(mean_diff_ci(diff, se, n - 1.0, level, alt)),
    })
}

/// Average ranks of `values` and the sizes of the tie groups.
fn ranks_and_ties(values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; values.len()];
    let mut ties = Vec::new();
    let mut s = 0;
    while s < order.len() {
        let mut e = s + 1;
        while e < order.len() && values[order[e]] == values[order[s]] {
            e += 1;
        }
        let r = s as f64 + (e - s + 1) as f64 / 2.0;
        for &i in &order[s..e] {
            ranks[i] = r;
        }
        ties.push((e - s) as f64);
        s = e;
    }
    (ranks, ties)
}

/// Number of arrangements of the Mann–Whitney U for sample sizes (m, n): `counts[u]` for u = 0..=m·n.
fn mwu_counts(m: usize, n: usize) -> Vec<f64> {
    // f(m, n, u) = f(m − 1, n, u − n) + f(m, n − 1, u), built over n for each m (generating function).
    let max = m * n;
    // Polynomial product ∏_{i=1..m} (1 − q^{n+i}) / (1 − q^i): counts of U.
    let mut c = vec![0.0f64; max + 1];
    c[0] = 1.0;
    for i in 1..=m {
        // multiply by (1 − q^{n+i})
        for u in (n + i..=max).rev() {
            c[u] -= c[u - n - i];
        }
        // divide by (1 − q^i)
        for u in i..=max {
            c[u] += c[u - i];
        }
    }
    c
}

/// Mann–Whitney U test (exact for small samples without ties, else tie-corrected normal approximation with
/// continuity correction, as SciPy). Statistic: U of `a`; estimate: rank-biserial correlation.
pub fn mann_whitney_test(a: &[f64], b: &[f64], alt: Alternative) -> Result<TestResult> {
    vec_check(a, "a", 1)?;
    vec_check(b, "b", 1)?;
    let (n1, n2) = (a.len(), b.len());
    let all: Vec<f64> = a.iter().chain(b).copied().collect();
    let (ranks, ties) = ranks_and_ties(&all);
    let r1: f64 = np_sum(&ranks[..n1]);
    let u = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let (n1f, n2f) = (n1 as f64, n2 as f64);
    let has_ties = ties.iter().any(|&t| t > 1.0);
    let p = if (n1 <= 8 || n2 <= 8) && !has_ties {
        let (m, n) = (n1.min(n2), n1.max(n2));
        let c = mwu_counts(m, n);
        let total: f64 = c.iter().sum();
        let u2 = n1f * n2f - u;
        let (stat, f) = match alt {
            Alternative::Greater => (u, 1.0),
            Alternative::Less => (u2, 1.0),
            Alternative::TwoSided => (u.max(u2), 2.0),
        };
        let k = stat.round() as usize;
        let sf: f64 = c[k.min(c.len())..].iter().sum::<f64>() / total;
        (f * sf).clamp(0.0, 1.0)
    } else {
        let n = n1f + n2f;
        let tie_term: f64 = ties
            .iter()
            .filter(|&&t| t > 1.0)
            .map(|t| t * t * t - t)
            .sum();
        let s = (n1f * n2f / 12.0 * ((n + 1.0) - tie_term / (n * (n - 1.0)))).sqrt();
        let mut num = u - n1f * n2f / 2.0;
        let sign = match alt {
            Alternative::Greater => 1.0,
            Alternative::Less => -1.0,
            Alternative::TwoSided => {
                if num > 0.0 {
                    1.0
                } else if num < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        };
        num -= 0.5 * sign;
        if s > 0.0 {
            z_p(num / s, alt)
        } else {
            f64::NAN
        }
    };
    Ok(TestResult {
        statistic: u,
        p_value: p,
        estimate: 2.0 * u / (n1f * n2f) - 1.0,
        ci: None,
    })
}

/// Distribution of the Wilcoxon signed-rank statistic T⁺ for n pairs without ties: counts for t = 0..=n(n+1)/2.
fn wilcoxon_counts(n: usize) -> Vec<f64> {
    let max = n * (n + 1) / 2;
    let mut c = vec![0.0f64; max + 1];
    c[0] = 1.0;
    for i in 1..=n {
        for t in (i..=max).rev() {
            c[t] += c[t - i];
        }
    }
    c
}

/// Wilcoxon signed-rank test on a − b (zeros dropped). Exact for ≤ 50 pairs without ties or zeros,
/// exact over all sign flips for ≤ 13 pairs with ties or zeros, otherwise the tie-corrected normal
/// approximation. Estimate: matched-pairs rank-biserial correlation.
pub fn wilcoxon_test(a: &[f64], b: &[f64], alt: Alternative) -> Result<TestResult> {
    vec_check(a, "a", 1)?;
    vec_check(b, "b", 1)?;
    if a.len() != b.len() {
        return invalid(format!(
            "Paired samples must have the same length; got {} and {}.",
            a.len(),
            b.len()
        ));
    }
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let nz: Vec<f64> = d.iter().copied().filter(|&v| v != 0.0).collect();
    if nz.is_empty() {
        return undefined("The Wilcoxon test is undefined: every difference is zero.");
    }
    let abs: Vec<f64> = nz.iter().map(|v| v.abs()).collect();
    let (ranks, ties) = ranks_and_ties(&abs);
    let t_plus: f64 = nz
        .iter()
        .zip(&ranks)
        .filter(|(v, _)| **v > 0.0)
        .map(|(_, r)| r)
        .sum();
    let t_minus: f64 = nz
        .iter()
        .zip(&ranks)
        .filter(|(v, _)| **v < 0.0)
        .map(|(_, r)| r)
        .sum();
    let stat = if alt == Alternative::TwoSided {
        t_plus.min(t_minus)
    } else {
        t_plus
    };
    let count = nz.len();
    let n_zero = d.len() - count;
    let has_ties = ties.iter().any(|&t| t > 1.0);
    let p = if d.len() <= 50 && !(has_ties || n_zero > 0) {
        let c = wilcoxon_counts(count);
        let total: f64 = c.iter().sum();
        let cdf = |k: f64| -> f64 {
            if k < 0.0 {
                0.0
            } else {
                c[..=(k as usize).min(c.len() - 1)].iter().sum::<f64>() / total
            }
        };
        let max = (count * (count + 1) / 2) as f64;
        let sf = |k: f64| cdf(max - k);
        match alt {
            Alternative::Less => cdf(t_plus.ceil()),
            Alternative::Greater => sf(t_plus.floor()),
            Alternative::TwoSided => {
                (2.0 * sf(t_plus.floor()).min(cdf(t_plus.ceil()))).clamp(0.0, 1.0)
            }
        }
    } else if d.len() <= 13 {
        // Exact permutation distribution over all 2^count sign assignments of the (tied) ranks.
        let total = 1u64 << count;
        let (mut le, mut ge) = (0u64, 0u64);
        let eps = 1e-10 * t_plus.abs().max(1.0);
        for mask in 0..total {
            let s: f64 = (0..count)
                .filter(|i| mask >> i & 1 == 1)
                .map(|i| ranks[i])
                .sum();
            if s <= t_plus + eps {
                le += 1;
            }
            if s >= t_plus - eps {
                ge += 1;
            }
        }
        let (pl, pg) = (le as f64 / total as f64, ge as f64 / total as f64);
        match alt {
            Alternative::Less => pl,
            Alternative::Greater => pg,
            Alternative::TwoSided => (2.0 * pl.min(pg)).min(1.0),
        }
    } else {
        let n = count as f64;
        let mn = n * (n + 1.0) * 0.25;
        let tie_term: f64 = ties.iter().map(|t| t * t * t - t).sum();
        let se = ((n * (n + 1.0) * (2.0 * n + 1.0) - tie_term / 2.0) / 24.0).sqrt();
        z_p((t_plus - mn) / se, alt)
    };
    Ok(TestResult {
        statistic: stat,
        p_value: p,
        estimate: (t_plus - t_minus) / (t_plus + t_minus),
        ci: None,
    })
}

/// Kruskal–Wallis H test across groups (tie-corrected). Estimate: epsilon-squared.
pub fn kruskal_wallis_test(groups: &[&[f64]]) -> Result<TestResult> {
    if groups.len() < 2 {
        return invalid(format!(
            "The Kruskal–Wallis test needs at least 2 groups; got {}.",
            groups.len()
        ));
    }
    for (i, g) in groups.iter().enumerate() {
        vec_check(g, &format!("group {}", i + 1), 1)?;
    }
    let all: Vec<f64> = groups.iter().flat_map(|g| g.iter().copied()).collect();
    let n = all.len() as f64;
    let (ranks, ties) = ranks_and_ties(&all);
    let mut pos = 0;
    let mut acc = 0.0;
    for g in groups {
        let s: f64 = ranks[pos..pos + g.len()].iter().sum();
        acc += s * s / g.len() as f64;
        pos += g.len();
    }
    let tie_term: f64 = ties
        .iter()
        .filter(|&&t| t > 1.0)
        .map(|t| t * t * t - t)
        .sum();
    let correction = 1.0 - tie_term / (n * n * n - n);
    if correction == 0.0 {
        return undefined("The Kruskal–Wallis test is undefined: all values are identical.");
    }
    let h = (12.0 / (n * (n + 1.0)) * acc - 3.0 * (n + 1.0)) / correction;
    let df = (groups.len() - 1) as f64;
    Ok(TestResult {
        statistic: h,
        p_value: chi2_sf(h, df),
        estimate: h / ((n * n - 1.0) / (n + 1.0)),
        ci: None,
    })
}

/// Friedman test for k ≥ 3 related samples (each slice one model's scores over the same blocks).
/// Estimate: Kendall's W.
pub fn friedman_test(groups: &[&[f64]]) -> Result<TestResult> {
    if groups.len() < 3 {
        return invalid(format!(
            "The Friedman test needs at least 3 groups; got {}.",
            groups.len()
        ));
    }
    for (i, g) in groups.iter().enumerate() {
        vec_check(g, &format!("group {}", i + 1), 1)?;
    }
    let n = groups[0].len();
    if groups.iter().any(|g| g.len() != n) {
        return invalid("Every sample in the Friedman test needs the same number of blocks.");
    }
    if n < 2 {
        return undefined("The Friedman test needs at least two blocks.");
    }
    let k = groups.len();
    let mut r = vec![0.0f64; k];
    let mut tie_sum = 0.0;
    for b in 0..n {
        for j in 0..k {
            let x = groups[j][b];
            let less = groups.iter().filter(|g| g[b] < x).count() as f64;
            let eq = groups.iter().filter(|g| g[b] == x).count() as f64;
            r[j] += 1.0 + less + (eq - 1.0) / 2.0;
            tie_sum += eq * eq - 1.0;
        }
    }
    let (nf, kf) = (n as f64, k as f64);
    let c = 1.0 - tie_sum / (nf * kf * (kf * kf - 1.0));
    if c == 0.0 {
        return undefined("The Friedman test is undefined: every block has all values tied.");
    }
    let rr: f64 = r.iter().map(|x| x * x).sum();
    let chi = (12.0 / (nf * kf * (kf + 1.0)) * rr - 3.0 * nf * (kf + 1.0)) / c;
    Ok(TestResult {
        statistic: chi,
        p_value: chi2_sf(chi, kf - 1.0),
        estimate: chi / (nf * (kf - 1.0)),
        ci: None,
    })
}

fn swilk_coefficients(n: usize) -> Vec<f64> {
    let nf = n as f64;
    let m: Vec<f64> = (1..=n)
        .map(|i| norm_ppf((i as f64 - 3.0 / 8.0) / (nf + 0.25)))
        .collect();
    let u = nf.powf(-0.5);
    let mtm: f64 = m.iter().map(|x| x * x).sum();
    let c: Vec<f64> = m.iter().map(|x| mtm.powf(-0.5) * x).collect();
    let (mnm1, mn) = (m[n - 2], m[n - 1]);
    let (cnm1, cn) = (c[n - 2], c[n - 1]);
    let an = cn + 0.221157 * u - 0.147981 * u.powi(2) - 2.071190 * u.powi(3) + 4.434685 * u.powi(4)
        - 2.706056 * u.powi(5);
    let anm1 = cnm1 + 0.042981 * u - 0.293762 * u.powi(2) - 1.752461 * u.powi(3)
        + 5.682633 * u.powi(4)
        - 3.582633 * u.powi(5);
    let phi = if n <= 5 {
        (mtm - 2.0 * mn * mn) / (1.0 - 2.0 * an * an)
    } else {
        (mtm - 2.0 * mn * mn - 2.0 * mnm1 * mnm1) / (1.0 - 2.0 * an * an - 2.0 * anm1 * anm1)
    };
    let mut a: Vec<f64> = m.iter().map(|x| phi.powf(-0.5) * x).collect();
    if n > 5 {
        a[n - 2] = anm1;
        a[1] = -anm1;
    }
    a[n - 1] = an;
    a[0] = -an;
    a
}

/// Shapiro–Wilk normality test (Royston's algorithm, as SciPy ≥ 1.12). Returns (W, p-value).
pub fn shapiro_wilk_test(x: &[f64]) -> Result<(f64, f64)> {
    vec_check(x, "x", 3)?;
    let n = x.len();
    let mut y = sorted(x);
    let lo = y[0];
    if y[n - 1] - lo == 0.0 {
        return undefined("The Shapiro–Wilk test is undefined for constant data.");
    }
    let mid = x[n / 2];
    for v in y.iter_mut() {
        *v -= mid;
    }
    let ym = mean(&y);
    let ss: f64 = y.iter().map(|v| (v - ym) * (v - ym)).sum();
    if n == 3 {
        let c = std::f64::consts::SQRT_2 / 2.0;
        let num = -c * y[0] + c * y[2];
        let w = (num * num / ss).clamp(0.75, 1.0);
        let p = (1.0 - 6.0 / std::f64::consts::PI * w.sqrt().acos()).clamp(0.0, 1.0);
        return Ok((w, p));
    }
    let a = swilk_coefficients(n);
    let dot: f64 = a.iter().zip(&y).map(|(a, b)| a * b).sum();
    let w = dot * dot / ss;
    let nf = n as f64;
    let (mu, sigma, g) = if n <= 11 {
        let gamma = -2.273 + 0.459 * nf;
        let mu = 0.5440 - 0.39978 * nf + 0.025054 * nf * nf - 0.0006714 * nf.powi(3);
        let sigma = (1.3822 - 0.77857 * nf + 0.062767 * nf * nf - 0.0020322 * nf.powi(3)).exp();
        (mu, sigma, -(gamma - (1.0 - w).ln()).ln())
    } else {
        let lu = nf.ln();
        let mu = -1.5861 - 0.31082 * lu - 0.083751 * lu * lu + 0.0038915 * lu.powi(3);
        let sigma = (-0.4803 - 0.082676 * lu + 0.0030302 * lu * lu).exp();
        (mu, sigma, (1.0 - w).ln())
    };
    Ok((w, norm_cdf(-(g - mu) / sigma)))
}

fn table_check(t: &[Vec<f64>]) -> Result<()> {
    if t.len() < 2 || t[0].len() < 2 || t.iter().any(|r| r.len() != t[0].len()) {
        return invalid("Give a contingency table with at least 2 rows and 2 columns.");
    }
    if t.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
        return invalid("Contingency table counts must be finite and non-negative.");
    }
    Ok(())
}

fn chi2_parts(t: &[Vec<f64>]) -> Result<(f64, Vec<Vec<f64>>, f64)> {
    let rows: Vec<f64> = t.iter().map(|r| r.iter().sum()).collect();
    let cols: Vec<f64> = (0..t[0].len())
        .map(|j| t.iter().map(|r| r[j]).sum())
        .collect();
    if rows.iter().chain(&cols).any(|&s| s == 0.0) {
        return undefined("The χ² test is undefined when a row or column total is zero.");
    }
    let n: f64 = rows.iter().sum();
    let expected: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| cols.iter().map(|c| r * c / n).collect())
        .collect();
    let mut chi = 0.0;
    for (i, row) in t.iter().enumerate() {
        for (j, &o) in row.iter().enumerate() {
            let d = o - expected[i][j];
            chi += d * d / expected[i][j];
        }
    }
    Ok((n, expected, chi))
}

/// Pearson χ² test of independence (Yates' correction for 2 × 2 tables unless `correction = false`).
/// Estimate: Cramér's V.
pub fn chi_square_test(table: &[Vec<f64>], correction: bool) -> Result<TestResult> {
    table_check(table)?;
    let (n, expected, uncorrected) = chi2_parts(table)?;
    let (r, k) = (table.len(), table[0].len());
    let dof = (r - 1) * (k - 1);
    let (chi, p) = if correction && dof == 1 {
        let mut c = 0.0;
        for (i, row) in table.iter().enumerate() {
            for (j, &o) in row.iter().enumerate() {
                let d = (o - expected[i][j]).abs();
                let adj = d - d.min(0.5);
                c += adj * adj / expected[i][j];
            }
        }
        (c, chi2_sf(c, 1.0))
    } else {
        (uncorrected, chi2_sf(uncorrected, dof as f64))
    };
    let v = ((uncorrected / n) / (r.min(k) - 1) as f64).sqrt();
    Ok(TestResult {
        statistic: chi,
        p_value: p,
        estimate: v,
        ci: None,
    })
}

/// Cramér's V of a contingency table (optionally with Bergsma's bias correction).
pub fn cramers_v(table: &[Vec<f64>], bias_correction: bool) -> Result<f64> {
    table_check(table)?;
    let (n, _, chi) = chi2_parts(table).map_err(|_| {
        crate::Error::Undefined(
            "Cramér's V is undefined when a row or column total is zero.".into(),
        )
    })?;
    let (r, k) = (table.len() as f64, table[0].len() as f64);
    if !bias_correction {
        return Ok((chi / (n * (r.min(k) - 1.0))).sqrt());
    }
    if n < 2.0 {
        return undefined("The bias-corrected Cramér's V needs at least two observations.");
    }
    let phi2 = (chi / n - (k - 1.0) * (r - 1.0) / (n - 1.0)).max(0.0);
    let rc = r - (r - 1.0).powi(2) / (n - 1.0);
    let kc = k - (k - 1.0).powi(2) / (n - 1.0);
    let denom = (rc - 1.0).min(kc - 1.0);
    if denom <= 0.0 {
        return undefined("The bias-corrected Cramér's V is undefined for this table size.");
    }
    Ok((phi2 / denom).sqrt())
}

fn fisher_p(a: i64, b: i64, c: i64, d: i64, alt: Alternative) -> f64 {
    let (n1, n2, n) = (a + b, c + d, a + c);
    let total = n1 + n2;
    let (lo, hi) = ((n - n2).max(0), n.min(n1));
    let mode = (n + 1) * (n1 + 1) / (total + 2);
    let (n1f, n2f, nf, tf) = (n1 as f64, n2 as f64, n as f64, total as f64);
    let sd = (nf * (n1f / tf) * (n2f / tf) * (tf - nf) / (tf - 1.0).max(1.0))
        .max(1.0)
        .sqrt();
    let wlo = lo.max((mode as f64 - 40.0 * sd) as i64 - 2);
    let whi = hi.min((mode as f64 + 40.0 * sd) as i64 + 2);
    let lp: Vec<f64> = (wlo..=whi)
        .map(|x| {
            let x = x as f64;
            -(ln_gamma(x + 1.0)
                + ln_gamma(n1f - x + 1.0)
                + ln_gamma(nf - x + 1.0)
                + ln_gamma(n2f - nf + x + 1.0))
        })
        .collect();
    let mx = lp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = lp.iter().map(|v| (v - mx).exp()).collect();
    let tot = np_sum(&w);
    let i = a - wlo;
    let len = w.len() as i64;
    let p = match alt {
        Alternative::Less => {
            if i < 0 {
                0.0
            } else if i >= len {
                1.0
            } else {
                np_sum(&w[..=i as usize]) / tot
            }
        }
        Alternative::Greater => {
            if i < 0 {
                1.0
            } else if i >= len {
                0.0
            } else {
                np_sum(&w[i as usize..]) / tot
            }
        }
        Alternative::TwoSided => {
            if !(0..len).contains(&i) {
                0.0
            } else {
                let lim = w[i as usize] * (1.0 + 1e-7);
                np_sum(&w.iter().copied().filter(|&v| v <= lim).collect::<Vec<_>>()) / tot
            }
        }
    };
    p.min(1.0)
}

/// Fisher's exact test on a 2 × 2 table [[a, b], [c, d]]. Statistic and estimate: the sample odds ratio.
pub fn fisher_exact_test(table: [[u64; 2]; 2], alt: Alternative) -> Result<TestResult> {
    let [[a, b], [c, d]] = table;
    let (a, b, c, d) = (a as i64, b as i64, c as i64, d as i64);
    let (odds, p) = if [a + b, c + d, a + c, b + d].contains(&0) {
        (f64::NAN, 1.0)
    } else {
        let odds = if b > 0 && c > 0 {
            (a * d) as f64 / (b * c) as f64
        } else {
            f64::INFINITY
        };
        (odds, fisher_p(a, b, c, d, alt))
    };
    Ok(TestResult {
        statistic: odds,
        p_value: p,
        estimate: odds,
        ci: None,
    })
}

/// McNemar's test for two classifiers on the same observations. `exact = None` uses the exact binomial test
/// below 25 discordant pairs. Estimate: accuracy difference A − B.
pub fn mcnemar_test<L: PartialEq>(
    y_true: &[L],
    pred_a: &[L],
    pred_b: &[L],
    exact: Option<bool>,
    correction: bool,
) -> Result<TestResult> {
    if y_true.is_empty() {
        return invalid("y_true is empty. Provide at least one example.");
    }
    if y_true.len() != pred_a.len() || y_true.len() != pred_b.len() {
        return invalid("y_true, y_pred_a and y_pred_b must have the same shape.");
    }
    let (mut b, mut c) = (0u64, 0u64);
    for ((t, x), y) in y_true.iter().zip(pred_a).zip(pred_b) {
        match (t == x, t == y) {
            (true, false) => b += 1,
            (false, true) => c += 1,
            _ => {}
        }
    }
    let n_disc = b + c;
    let estimate = (b as f64 - c as f64) / y_true.len() as f64;
    if n_disc == 0 {
        return Ok(TestResult {
            statistic: 0.0,
            p_value: 1.0,
            estimate,
            ci: None,
        });
    }
    if exact.unwrap_or(n_disc < 25) {
        let k = b.min(c) as f64;
        let p = (2.0 * special::binom_cdf(k, n_disc as f64, 0.5)).min(1.0);
        return Ok(TestResult {
            statistic: k,
            p_value: p,
            estimate,
            ci: None,
        });
    }
    let d = (b as f64 - c as f64).abs() - if correction { 1.0 } else { 0.0 };
    let stat = d * d / n_disc as f64;
    Ok(TestResult {
        statistic: stat,
        p_value: chi2_sf(stat, 1.0),
        estimate,
        ci: None,
    })
}

// ---- effect sizes ---------------------------------------------------------------------------------

fn finite_pair(a: &[f64], b: &[f64]) -> Result<()> {
    vec_check(a, "a", 1)?;
    vec_check(b, "b", 1)
}

/// Cohen's d: mean difference over the pooled SD, or over the SD of the differences (d_z) when `paired`.
pub fn cohens_d(a: &[f64], b: &[f64], paired: bool) -> Result<f64> {
    finite_pair(a, b)?;
    if paired {
        if a.len() != b.len() {
            return invalid("Paired samples must have the same length.");
        }
        if a.len() < 2 {
            return undefined("Cohen's d needs at least two pairs.");
        }
        let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        let sd = var1(&d).sqrt();
        if sd == 0.0 {
            return undefined("Cohen's d is undefined: the paired differences are all identical.");
        }
        return Ok(mean(&d) / sd);
    }
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    if a.len() < 2 || b.len() < 2 {
        return undefined("Cohen's d needs at least two observations per group.");
    }
    let pooled = (((n1 - 1.0) * var1(a) + (n2 - 1.0) * var1(b)) / (n1 + n2 - 2.0)).sqrt();
    if pooled == 0.0 {
        return undefined("Cohen's d is undefined: both groups have zero variance.");
    }
    Ok((mean(a) - mean(b)) / pooled)
}

/// Hedges' g: Cohen's d with the small-sample correction J.
pub fn hedges_g(a: &[f64], b: &[f64], paired: bool) -> Result<f64> {
    finite_pair(a, b)?;
    let df = if paired {
        a.len() as f64 - 1.0
    } else {
        (a.len() + b.len()) as f64 - 2.0
    };
    if df < 2.0 {
        return undefined("Hedges' g needs at least three observations.");
    }
    let j = (ln_gamma(df / 2.0) - ln_gamma((df - 1.0) / 2.0)).exp() / (df / 2.0).sqrt();
    Ok(j * cohens_d(a, b, paired)?)
}

/// Cliff's delta: P(a > b) − P(a < b).
pub fn cliffs_delta(a: &[f64], b: &[f64]) -> Result<f64> {
    finite_pair(a, b)?;
    let ys = sorted(b);
    let (mut less, mut greater) = (0u64, 0u64);
    for &x in a {
        less += ys.partition_point(|&y| y < x) as u64;
        greater += (ys.len() - ys.partition_point(|&y| y <= x)) as u64;
    }
    Ok((less as f64 - greater as f64) / (a.len() * b.len()) as f64)
}

/// Multiple-testing correction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Adjust {
    Bonferroni,
    #[default]
    Holm,
    Hochberg,
    /// Benjamini–Hochberg false discovery rate.
    Bh,
    /// Benjamini–Yekutieli false discovery rate.
    By,
}

/// Adjusted p-values, in input order.
pub fn adjust_pvalues(p: &[f64], method: Adjust) -> Result<Vec<f64>> {
    vec_check(p, "p_values", 1)?;
    if p.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return invalid("p-values must be between 0 and 1.");
    }
    let m = p.len();
    let mf = m as f64;
    if method == Adjust::Bonferroni {
        return Ok(p.iter().map(|v| (v * mf).min(1.0)).collect());
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].partial_cmp(&p[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut adj: Vec<f64> = match method {
        Adjust::Holm | Adjust::Hochberg => order
            .iter()
            .enumerate()
            .map(|(i, &j)| p[j] * (m - i) as f64)
            .collect(),
        _ => {
            let factor = if method == Adjust::By {
                np_sum(&(1..=m).map(|k| 1.0 / k as f64).collect::<Vec<_>>())
            } else {
                1.0
            };
            order
                .iter()
                .enumerate()
                .map(|(i, &j)| p[j] * mf * factor / (i + 1) as f64)
                .collect()
        }
    };
    if method == Adjust::Holm {
        let mut run = f64::NEG_INFINITY;
        for x in adj.iter_mut() {
            run = run.max(*x);
            *x = run;
        }
    } else {
        let mut run = f64::INFINITY;
        for x in adj.iter_mut().rev() {
            run = run.min(*x);
            *x = run;
        }
    }
    let mut out = vec![0.0; m];
    for (k, &j) in order.iter().enumerate() {
        out[j] = adj[k].min(1.0);
    }
    Ok(out)
}

// ---- intervals ------------------------------------------------------------------------------------

/// Method for a binomial proportion interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProportionMethod {
    #[default]
    Wilson,
    ClopperPearson,
    /// Wald (normal approximation); shown for comparison, not recommended.
    Normal,
}

/// Confidence interval for a binomial proportion.
pub fn proportion_ci(
    successes: u64,
    n: u64,
    level: f64,
    method: ProportionMethod,
) -> Result<Interval> {
    check_level(level)?;
    if n == 0 {
        return invalid("n must be a positive integer.");
    }
    if successes > n {
        return invalid("successes must be between 0 and n.");
    }
    let (s, nf) = (successes as f64, n as f64);
    let p = s / nf;
    let alpha = 1.0 - level;
    let z = norm_ppf(1.0 - alpha / 2.0);
    let (low, high) = match method {
        ProportionMethod::Wilson => {
            let denom = 1.0 + z * z / nf;
            let centre = (p + z * z / (2.0 * nf)) / denom;
            let half = z * (p * (1.0 - p) / nf + z * z / (4.0 * nf * nf)).sqrt() / denom;
            (centre - half, centre + half)
        }
        ProportionMethod::ClopperPearson => (
            if successes == 0 {
                0.0
            } else {
                special::beta_inc_inv(s, nf - s + 1.0, alpha / 2.0)
            },
            if successes == n {
                1.0
            } else {
                special::beta_inc_inv(s + 1.0, nf - s, 1.0 - alpha / 2.0)
            },
        ),
        ProportionMethod::Normal => {
            let half = z * (p * (1.0 - p) / nf).sqrt();
            ((p - half).max(0.0), (p + half).min(1.0))
        }
    };
    Ok(Interval {
        estimate: p,
        low,
        high,
    })
}

/// Confidence interval for accuracy (a binomial proportion of correct predictions).
pub fn accuracy_ci<L: PartialEq>(
    y_true: &[L],
    y_pred: &[L],
    level: f64,
    method: ProportionMethod,
) -> Result<Interval> {
    if y_true.len() != y_pred.len() {
        return invalid("y_true and y_pred must have the same shape.");
    }
    let k = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count() as u64;
    proportion_ci(k, y_true.len() as u64, level, method)
}

fn binary01(y: &[u8]) -> Result<Vec<bool>> {
    if y.iter().any(|&v| v > 1) {
        return invalid("DeLong's method is for binary 0/1 targets.");
    }
    Ok(y.iter().map(|&v| v == 1).collect())
}

/// AUCs and DeLong structural components of several score columns.
fn delong(pos: &[bool], cols: &[&[f64]]) -> Result<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    let m = pos.iter().filter(|p| **p).count();
    let n = pos.len() - m;
    if m == 0 || n == 0 {
        return undefined(
            "DeLong's method needs at least one positive and one negative observation.",
        );
    }
    let (mf, nf) = (m as f64, n as f64);
    let (mut auc, mut v10, mut v01) = (Vec::new(), Vec::new(), Vec::new());
    for s in cols {
        let (tzp, tzn, tx, ty) = evalsuite_core::ranks::delong_ranks(pos, s);
        auc.push((np_sum(&tzp) - mf * (mf + 1.0) / 2.0) / (mf * nf));
        v10.push(tzp.iter().zip(&tx).map(|(z, x)| (z - x) / nf).collect());
        v01.push(
            tzn.iter()
                .zip(&ty)
                .map(|(z, y)| 1.0 - (z - y) / mf)
                .collect(),
        );
    }
    Ok((auc, v10, v01))
}

fn cov(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (a.len() - 1) as f64
}

/// DeLong confidence interval for one ROC AUC (binary 0/1 `y_true`).
pub fn roc_auc_ci(y_true: &[u8], y_prob: &[f64], level: f64) -> Result<Interval> {
    check_level(level)?;
    if y_true.len() != y_prob.len() {
        return invalid("y_true and y_prob must contain the same number of observations.");
    }
    let pos = binary01(y_true)?;
    let (auc, v10, v01) = delong(&pos, &[y_prob])?;
    let (m, n) = (v10[0].len() as f64, v01[0].len() as f64);
    let s10 = if m > 1.0 { cov(&v10[0], &v10[0]) } else { 0.0 };
    let s01 = if n > 1.0 { cov(&v01[0], &v01[0]) } else { 0.0 };
    let se = (s10 / m + s01 / n).sqrt();
    let z = norm_ppf(1.0 - (1.0 - level) / 2.0);
    Ok(Interval {
        estimate: auc[0],
        low: (auc[0] - z * se).max(0.0),
        high: (auc[0] + z * se).min(1.0),
    })
}

/// DeLong's test for two correlated ROC AUCs on the same observations. Estimate: AUC_A − AUC_B.
pub fn delong_test(
    y_true: &[u8],
    prob_a: &[f64],
    prob_b: &[f64],
    level: f64,
) -> Result<TestResult> {
    check_level(level)?;
    if prob_a.len() != prob_b.len() || y_true.len() != prob_a.len() {
        return invalid("y_true, y_prob_a and y_prob_b must have the same length.");
    }
    let pos = binary01(y_true)?;
    let (auc, v10, v01) = delong(&pos, &[prob_a, prob_b])?;
    let (m, n) = (v10[0].len() as f64, v01[0].len() as f64);
    let c = |i: usize, j: usize| {
        let s10 = if m > 1.0 { cov(&v10[i], &v10[j]) } else { 0.0 };
        let s01 = if n > 1.0 { cov(&v01[i], &v01[j]) } else { 0.0 };
        s10 / m + s01 / n
    };
    let diff = auc[0] - auc[1];
    let var = c(0, 0) + c(1, 1) - 2.0 * c(0, 1);
    let (z, p, se) = if var <= 0.0 {
        (0.0, 1.0, 0.0)
    } else {
        let se = var.sqrt();
        let z = diff / se;
        (z, 2.0 * norm_cdf(-z.abs()), se)
    };
    let zc = norm_ppf(1.0 - (1.0 - level) / 2.0);
    Ok(TestResult {
        statistic: z,
        p_value: p,
        estimate: diff,
        ci: Some((diff - zc * se, diff + zc * se)),
    })
}

// ---- bootstrap ------------------------------------------------------------------------------------

/// Bootstrap interval method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BootstrapMethod {
    /// Bias-corrected and accelerated (Efron 1987).
    #[default]
    Bca,
    Percentile,
    Basic,
}

/// Bootstrap settings.
#[derive(Debug, Clone)]
pub struct Bootstrap {
    pub n_resamples: usize,
    pub level: f64,
    pub method: BootstrapMethod,
    pub seed: u64,
    /// Resample within strata (for example the class of each observation), keeping their sizes.
    pub strata: Option<Vec<u64>>,
    /// Jackknife blocks for the BCa acceleration (leave-one-out when n ≤ this).
    pub jackknife_groups: usize,
}

impl Default for Bootstrap {
    fn default() -> Self {
        Bootstrap {
            n_resamples: 2000,
            level: 0.95,
            method: BootstrapMethod::Bca,
            seed: 0,
            strata: None,
            jackknife_groups: 1000,
        }
    }
}

/// Bootstrap result: the interval, the bootstrap distribution's size and the number of failed resamples.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapInterval {
    pub estimate: f64,
    pub low: f64,
    pub high: f64,
    pub failed_resamples: usize,
}

/// Resample indices (`n_resamples × n`, row-major), drawn with replacement within strata, from the seeded
/// generator of EvalSuite's compiled core (the same stream as Python's `resampling="native"`).
pub fn resample_indices(n: usize, opts: &Bootstrap) -> Vec<i64> {
    match &opts.strata {
        None => evalsuite_core::resample::bootstrap_indices(n, opts.n_resamples, opts.seed, None),
        Some(s) => {
            let mut values: Vec<u64> = s.clone();
            values.sort_unstable();
            values.dedup();
            let mut members = Vec::with_capacity(n);
            let mut offsets = vec![0usize];
            for v in values {
                members.extend((0..n).filter(|&i| s[i] == v).map(|i| i as i64));
                offsets.push(members.len());
            }
            evalsuite_core::resample::bootstrap_indices(
                n,
                opts.n_resamples,
                opts.seed,
                Some((&members, &offsets)),
            )
        }
    }
}

fn check_boot(n: usize, o: &Bootstrap) -> Result<()> {
    check_level(o.level)?;
    if o.n_resamples < 100 {
        return invalid(
            "n_resamples must be an integer of at least 100 (1000 or more recommended).",
        );
    }
    if n < 2 {
        return undefined(format!(
            "Resampling needs at least 2 observations; got {n}."
        ));
    }
    if let Some(s) = &o.strata {
        if s.len() != n {
            return invalid("strata must have one entry per observation.");
        }
    }
    Ok(())
}

fn distribution(
    metric: &(dyn Fn(&[usize]) -> Result<f64> + Sync),
    idx: &[i64],
    n: usize,
    rows: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    idx.par_chunks(n)
        .take(rows)
        .map(|row| {
            let r: Vec<usize> = row.iter().map(|&i| i as usize).collect();
            metric(&r).unwrap_or(f64::NAN)
        })
        .collect()
}

/// Bootstrap confidence interval of `metric`, a function of the observation indices it is evaluated on
/// (the full sample is `0..n`). Undefined resamples (errors) are dropped and counted; more than 10% is an error.
///
/// ```
/// use evalsuite::{classification, stats::{bootstrap_ci, Bootstrap}};
/// let y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];
/// let p = [0, 1, 0, 0, 1, 1, 1, 1, 0, 1];
/// let acc = |idx: &[usize]| {
///     let (t, q): (Vec<i32>, Vec<i32>) = idx.iter().map(|&i| (y[i], p[i])).unzip();
///     classification::accuracy(&t, &q, None)
/// };
/// let ci = bootstrap_ci(y.len(), acc, &Bootstrap { seed: 1, ..Default::default() }).unwrap();
/// assert!(ci.low <= ci.estimate && ci.estimate <= ci.high);
/// ```
pub fn bootstrap_ci(
    n: usize,
    metric: impl Fn(&[usize]) -> Result<f64> + Sync,
    opts: &Bootstrap,
) -> Result<BootstrapInterval> {
    check_boot(n, opts)?;
    let all: Vec<usize> = (0..n).collect();
    let estimate = metric(&all)?;
    let idx = resample_indices(n, opts);
    let boot = distribution(&metric, &idx, n, opts.n_resamples);
    let failed = boot.iter().filter(|v| v.is_nan()).count();
    if failed as f64 > 0.1 * opts.n_resamples as f64 {
        return undefined(format!(
            "The metric was undefined on {failed} of {} bootstrap resamples. The sample is probably too small or \
             too unbalanced.",
            opts.n_resamples
        ));
    }
    let valid = sorted(
        &boot
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect::<Vec<_>>(),
    );
    let alpha = 1.0 - opts.level;
    let (low, high) = match opts.method {
        BootstrapMethod::Percentile => (
            quantile(&valid, alpha / 2.0),
            quantile(&valid, 1.0 - alpha / 2.0),
        ),
        BootstrapMethod::Basic => (
            2.0 * estimate - quantile(&valid, 1.0 - alpha / 2.0),
            2.0 * estimate - quantile(&valid, alpha / 2.0),
        ),
        BootstrapMethod::Bca => {
            let below = valid.iter().filter(|&&v| v < estimate).count() as f64;
            let equal = valid.iter().filter(|&&v| v == estimate).count() as f64;
            let share = (below + 0.5 * equal) / valid.len() as f64;
            if share <= 0.0 || share >= 1.0 {
                (
                    quantile(&valid, alpha / 2.0),
                    quantile(&valid, 1.0 - alpha / 2.0),
                )
            } else {
                let z0 = norm_ppf(share);
                let jack = jackknife(n, &metric, opts);
                let jm = mean(&jack);
                let diffs: Vec<f64> = jack.iter().map(|j| jm - j).collect();
                let denom =
                    6.0 * np_sum(&diffs.iter().map(|d| d * d).collect::<Vec<_>>()).powf(1.5);
                let accel = if denom > 0.0 {
                    np_sum(&diffs.iter().map(|d| d * d * d).collect::<Vec<_>>()) / denom
                } else {
                    0.0
                };
                let q = |za: f64| {
                    let num = z0 + za;
                    norm_cdf(z0 + num / (1.0 - accel * num))
                };
                (
                    quantile(&valid, q(norm_ppf(alpha / 2.0))),
                    quantile(&valid, q(norm_ppf(1.0 - alpha / 2.0))),
                )
            }
        }
    };
    Ok(BootstrapInterval {
        estimate,
        low,
        high,
        failed_resamples: failed,
    })
}

fn jackknife(
    n: usize,
    metric: &(dyn Fn(&[usize]) -> Result<f64> + Sync),
    opts: &Bootstrap,
) -> Vec<f64> {
    use rayon::prelude::*;
    let blocks: Vec<Vec<usize>> = if n <= opts.jackknife_groups {
        (0..n).map(|i| vec![i]).collect()
    } else {
        // A seeded permutation (the compiled core's generator) split into nearly equal blocks.
        let perm = evalsuite_core::resample::bootstrap_indices(
            n,
            1,
            opts.seed ^ 0x9E37_79B9_7F4A_7C15,
            None,
        );
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (perm[i], i));
        crate::calibration::array_split(&order, opts.jackknife_groups)
    };
    let out: Vec<f64> = blocks
        .par_iter()
        .map(|block| {
            let keep: Vec<usize> = (0..n).filter(|i| !block.contains(i)).collect();
            metric(&keep).unwrap_or(f64::NAN)
        })
        .collect();
    out.into_iter().filter(|v| !v.is_nan()).collect()
}

/// Paired bootstrap test of metric(A) − metric(B) on the same resampled observations (shifted-null p-value
/// with the +1 correction; percentile CI of the difference).
pub fn paired_bootstrap_test(
    n: usize,
    metric_a: impl Fn(&[usize]) -> Result<f64> + Sync,
    metric_b: impl Fn(&[usize]) -> Result<f64> + Sync,
    alt: Alternative,
    opts: &Bootstrap,
) -> Result<TestResult> {
    check_boot(n, opts)?;
    let all: Vec<usize> = (0..n).collect();
    let observed = metric_a(&all)? - metric_b(&all)?;
    let idx = resample_indices(n, opts);
    let da = distribution(&metric_a, &idx, n, opts.n_resamples);
    let db = distribution(&metric_b, &idx, n, opts.n_resamples);
    let diff: Vec<f64> = da
        .iter()
        .zip(&db)
        .map(|(a, b)| a - b)
        .filter(|v| !v.is_nan())
        .collect();
    let extreme = diff
        .iter()
        .filter(|&&d| {
            let c = d - observed;
            match alt {
                Alternative::TwoSided => c.abs() >= observed.abs(),
                Alternative::Greater => c >= observed,
                Alternative::Less => c <= observed,
            }
        })
        .count() as f64;
    let p = (1.0 + extreme) / (diff.len() as f64 + 1.0);
    let s = sorted(&diff);
    let alpha = 1.0 - opts.level;
    Ok(TestResult {
        statistic: observed,
        p_value: p,
        estimate: observed,
        ci: Some((quantile(&s, alpha / 2.0), quantile(&s, 1.0 - alpha / 2.0))),
    })
}

/// Which model pairs [`compare`] tests.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Pairs {
    /// Every pair of models.
    #[default]
    All,
    /// Every model against the named baseline.
    Baseline(String),
}

/// One model's bootstrap interval in [`compare`].
#[derive(Debug, Clone, PartialEq)]
pub struct ModelEstimate {
    pub model: String,
    pub estimate: f64,
    pub low: f64,
    pub high: f64,
    pub failed_resamples: usize,
}

/// One paired comparison in [`compare`].
#[derive(Debug, Clone, PartialEq)]
pub struct PairTest {
    pub model_a: String,
    pub model_b: String,
    pub difference: f64,
    pub low: f64,
    pub high: f64,
    pub p_value: f64,
    pub p_adjusted: f64,
    pub significant: bool,
}

/// A metric function of observation indices, for [`compare`].
pub type IndexMetric<'a> = &'a (dyn Fn(&[usize]) -> Result<f64> + Sync);

/// Compare several models on one metric with paired bootstrap resampling (every model on the same
/// resamples): a percentile interval per model and a paired bootstrap test per pair, with p-values adjusted
/// by `correction`. `significant` uses the adjusted p-value and `alpha`. (For accuracy and binary ROC AUC the
/// Python `compare` uses McNemar's and DeLong's tests; call [`mcnemar_test`] / [`delong_test`] for those.)
pub fn compare(
    n: usize,
    models: &[(&str, IndexMetric<'_>)],
    pairs: &Pairs,
    correction: Adjust,
    alpha: f64,
    opts: &Bootstrap,
) -> Result<(Vec<ModelEstimate>, Vec<PairTest>)> {
    check_boot(n, opts)?;
    if models.len() < 2 {
        return invalid("compare() needs at least two models.");
    }
    if let Pairs::Baseline(b) = pairs {
        if !models.iter().any(|(m, _)| m == b) {
            return invalid(format!("baseline={b:?} is not one of the models."));
        }
    }
    let all: Vec<usize> = (0..n).collect();
    let idx = resample_indices(n, opts);
    let a = (1.0 - opts.level) / 2.0;
    let mut dists = Vec::with_capacity(models.len());
    let mut observed = Vec::with_capacity(models.len());
    let mut estimates = Vec::with_capacity(models.len());
    for (name, f) in models {
        let est = f(&all)?;
        let d = distribution(*f, &idx, n, opts.n_resamples);
        let failed = d.iter().filter(|v| v.is_nan()).count();
        let s = sorted(
            &d.iter()
                .copied()
                .filter(|v| !v.is_nan())
                .collect::<Vec<_>>(),
        );
        estimates.push(ModelEstimate {
            model: name.to_string(),
            estimate: est,
            low: quantile(&s, a),
            high: quantile(&s, 1.0 - a),
            failed_resamples: failed,
        });
        dists.push(d);
        observed.push(est);
    }
    let index: Vec<(usize, usize)> = match pairs {
        Pairs::All => (0..models.len())
            .flat_map(|i| (i + 1..models.len()).map(move |j| (i, j)))
            .collect(),
        Pairs::Baseline(b) => {
            let bi = models.iter().position(|(m, _)| m == b).unwrap();
            (0..models.len())
                .filter(|&i| i != bi)
                .map(|i| (i, bi))
                .collect()
        }
    };
    let mut tests = Vec::with_capacity(index.len());
    for &(i, j) in &index {
        let diff: Vec<f64> = dists[i]
            .iter()
            .zip(&dists[j])
            .map(|(x, y)| x - y)
            .filter(|v| !v.is_nan())
            .collect();
        let obs = observed[i] - observed[j];
        let extreme = diff
            .iter()
            .filter(|&&d| (d - obs).abs() >= obs.abs())
            .count() as f64;
        let s = sorted(&diff);
        tests.push(PairTest {
            model_a: models[i].0.to_string(),
            model_b: models[j].0.to_string(),
            difference: obs,
            low: quantile(&s, a),
            high: quantile(&s, 1.0 - a),
            p_value: (1.0 + extreme) / (diff.len() as f64 + 1.0),
            p_adjusted: 0.0,
            significant: false,
        });
    }
    let adj = adjust_pvalues(
        &tests.iter().map(|t| t.p_value).collect::<Vec<_>>(),
        correction,
    )?;
    for (t, p) in tests.iter_mut().zip(adj) {
        t.p_adjusted = p;
        t.significant = p < alpha;
    }
    Ok((estimates, tests))
}
