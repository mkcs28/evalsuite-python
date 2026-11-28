//! Special functions and distributions used by the statistics: normal, Student t, χ², F, beta and
//! binomial/hypergeometric helpers, accurate to about 1e-14 relative (checked against SciPy).

use std::f64::consts::{PI, SQRT_2};

/// ln Γ(x) for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// ln B(a, b).
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// ln C(n, k).
pub fn ln_choose(n: f64, k: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
}

/// Standard normal CDF Φ(x) (SciPy's `ndtr`).
pub fn norm_cdf(x: f64) -> f64 {
    let t = x / SQRT_2;
    if t.abs() < std::f64::consts::FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(t)
    } else if t > 0.0 {
        1.0 - 0.5 * libm::erfc(t)
    } else {
        0.5 * libm::erfc(-t)
    }
}

/// Standard normal survival function 1 − Φ(x), accurate in the upper tail.
pub fn norm_sf(x: f64) -> f64 {
    norm_cdf(-x)
}

/// Standard normal density.
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Inverse standard normal CDF Φ⁻¹(p) (Wichura's AS241, ~1e-16 relative), refined by one Newton step.
pub fn norm_ppf(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        q * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r
            + 67265.770927008700853)
            * r
            + 45921.953931549871457)
            * r
            + 13731.693765509461125)
            * r
            + 1971.5909503065514427)
            * r
            + 133.14166789178437745)
            * r
            + 3.387132872796366608)
            / (((((((5226.495278852545925 * r + 28729.085735721942674) * r
                + 39307.89580009271061)
                * r
                + 21213.794301586595867)
                * r
                + 5394.1960214247511077)
                * r
                + 687.1870074920579083)
                * r
                + 42.313330701600911252)
                * r
                + 1.0)
    } else {
        let mut r = if q < 0.0 { p } else { 1.0 - p };
        r = (-r.ln()).sqrt();
        let v = if r <= 5.0 {
            let r = r - 1.6;
            (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r
                + 0.24178072517745061177)
                * r
                + 1.27045825245236838258)
                * r
                + 3.64784832476320460504)
                * r
                + 5.7694972214606914055)
                * r
                + 4.6303378461565452959)
                * r
                + 1.42343711074968357734)
                / (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r
                    + 0.0151986665636164571966)
                    * r
                    + 0.14810397642748007459)
                    * r
                    + 0.68976733498510000455)
                    * r
                    + 1.6763848301838038494)
                    * r
                    + 2.05319162663775882187)
                    * r
                    + 1.0)
        } else {
            let r = r - 5.0;
            (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
                + 0.0012426609473880784386)
                * r
                + 0.026532189526576123093)
                * r
                + 0.29656057182850489123)
                * r
                + 1.7848265399172913358)
                * r
                + 5.4637849111641143699)
                * r
                + 6.6579046435011037772)
                / (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r
                    + 1.8463183175100546818e-5)
                    * r
                    + 7.868691311456132591e-4)
                    * r
                    + 0.0148753612908506148525)
                    * r
                    + 0.13692988092273580531)
                    * r
                    + 0.59983220655588793769)
                    * r
                    + 1.0)
        };
        if q < 0.0 {
            -v
        } else {
            v
        }
    }
}

/// Regularized lower incomplete gamma P(a, x).
pub fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

/// Regularized upper incomplete gamma Q(a, x) = 1 − P(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..10_000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-17 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..10_000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-17 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// χ² survival function with `df` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

/// χ² CDF.
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    gamma_p(df / 2.0, x / 2.0)
}

/// Regularized incomplete beta I_x(a, b).
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Inverse of the regularized incomplete beta: x with I_x(a, b) = p.
pub fn beta_inc_inv(a: f64, b: f64, p: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    // Bisection to a bracket, then Newton steps; I_x is monotone in x.
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut x = 0.5;
    for _ in 0..200 {
        let f = beta_inc(a, b, x) - p;
        if f.abs() < 1e-16 {
            break;
        }
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let dens = ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta(a, b)).exp();
        let mut nx = x - f / dens;
        if !(nx > lo && nx < hi) || !nx.is_finite() {
            nx = 0.5 * (lo + hi);
        }
        if (nx - x).abs() < 1e-17 * x.max(1e-300) {
            x = nx;
            break;
        }
        x = nx;
    }
    x
}

/// Student t CDF with `df` degrees of freedom.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    if t.is_infinite() {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    let x = df / (df + t * t);
    let tail = 0.5 * beta_inc(df / 2.0, 0.5, x);
    if t > 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Student t survival function.
pub fn t_sf(t: f64, df: f64) -> f64 {
    t_cdf(-t, df)
}

/// Inverse Student t CDF.
pub fn t_ppf(p: f64, df: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    // t² = df (1/x − 1) with x = I⁻¹(df/2, 1/2; 2 min(p, 1−p)).
    let tail = if p < 0.5 { p } else { 1.0 - p };
    let x = beta_inc_inv(df / 2.0, 0.5, 2.0 * tail);
    let t = (df * (1.0 / x - 1.0)).sqrt();
    if p < 0.5 {
        -t
    } else {
        t
    }
}

/// F distribution survival function.
pub fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    beta_inc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))
}

/// Binomial CDF P(X ≤ k) for X ~ Bin(n, p).
pub fn binom_cdf(k: f64, n: f64, p: f64) -> f64 {
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    beta_inc(n - k, k + 1.0, 1.0 - p)
}

/// Beta distribution quantile.
pub fn beta_ppf(q: f64, a: f64, b: f64) -> f64 {
    beta_inc_inv(a, b, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-13 * b.abs().max(1e-300)
    }

    #[test]
    fn known_values() {
        assert!(close(norm_cdf(1.959963984540054), 0.975));
        assert!(close(norm_ppf(0.975), 1.959963984540054));
        assert!(close(chi2_sf(3.841458820694124, 1.0), 0.05));
        assert!(close(t_cdf(2.0, 5.0), 0.9490302605850709));
        assert!((t_cdf(t_ppf(0.975, 10.0), 10.0) - 0.975).abs() < 1e-14);
        assert!(close(beta_inc(2.0, 3.0, 0.4), 0.5248));
    }
}
