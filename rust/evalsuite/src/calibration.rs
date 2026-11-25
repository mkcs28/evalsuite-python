//! Calibration of binary risk predictions, as EvalSuite's Python `evalsuite.calibration`: maximum calibration
//! error, logistic recalibration slope and intercept, and the Hosmer–Lemeshow test. The calibration curve, ECE
//! and Brier score are in [`crate::classification::binary`].

use crate::classification::binary::{self, Strategy};
use crate::classification::{check_probs, Ctx, Kind};
use crate::error::{invalid, undefined, unsupported, Result};
use crate::labels::Label;
use crate::special::chi2_sf;

const EPS: f64 = 1e-15;

fn target<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let ctx = Ctx::new(y_true, None, None, w)?;
    if ctx.kind != Kind::Binary {
        return unsupported(
            "Calibration measures here are for binary outcomes; use one-vs-rest per class.",
        );
    }
    check_probs(y_prob, ctx.n(), "y_prob")?;
    let y = match ctx.positive(pos_label)? {
        None => vec![0.0; ctx.n()],
        Some(i) => ctx
            .true_idx
            .iter()
            .map(|&t| f64::from(u8::from(t == i)))
            .collect(),
    };
    Ok((y, ctx.w))
}

/// Largest gap between observed and predicted risk over the calibration bins.
pub fn maximum_calibration_error<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    n_bins: usize,
    strategy: Strategy,
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<f64> {
    let (t, p, _) = binary::calibration_curve(y_true, y_prob, n_bins, strategy, pos_label, w)?;
    Ok(t.iter()
        .zip(&p)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs())))
}

fn logit(p: f64) -> f64 {
    let q = p.clamp(EPS, 1.0 - EPS);
    (q / (1.0 - q)).ln()
}

/// Weighted logistic regression by Newton–Raphson. `offset = true` fits logit P = a + x (returns `[a]`),
/// otherwise logit P = a + b·x (returns `[a, b]`).
pub fn logistic_fit(y: &[f64], x: &[f64], w: &[f64], offset: bool) -> Result<Vec<f64>> {
    let k = if offset { 1 } else { 2 };
    let mut beta = vec![0.0; k];
    for _ in 0..100 {
        let mut grad = [0.0f64; 2];
        let mut hess = [[0.0f64; 2]; 2];
        for i in 0..y.len() {
            let eta = if offset {
                beta[0] + x[i]
            } else {
                beta[0] + beta[1] * x[i]
            };
            let mu = 1.0 / (1.0 + (-eta.clamp(-700.0, 700.0)).exp());
            let row = [1.0, x[i]];
            let r = w[i] * (y[i] - mu);
            let h = w[i] * mu * (1.0 - mu);
            for a in 0..k {
                grad[a] += row[a] * r;
                for b in 0..k {
                    hess[a][b] += row[a] * row[b] * h;
                }
            }
        }
        let step: Vec<f64> = if k == 1 {
            if hess[0][0] == 0.0 {
                return undefined("The logistic recalibration model is singular (perfect separation or constant risk).");
            }
            vec![grad[0] / hess[0][0]]
        } else {
            let det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
            if det == 0.0 || !det.is_finite() {
                return undefined("The logistic recalibration model is singular (perfect separation or constant risk).");
            }
            vec![
                (hess[1][1] * grad[0] - hess[0][1] * grad[1]) / det,
                (hess[0][0] * grad[1] - hess[1][0] * grad[0]) / det,
            ]
        };
        for (b, s) in beta.iter_mut().zip(&step) {
            *b += s;
        }
        if step.iter().fold(0.0f64, |m, s| m.max(s.abs())) < 1e-12 {
            return Ok(beta);
        }
    }
    undefined("The logistic recalibration did not converge (the predictions may separate the outcomes perfectly).")
}

fn need_both(y: &[f64], what: &str) -> Result<()> {
    let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lo == hi {
        return invalid(format!("{what} needs both outcomes (0 and 1) in y_true."));
    }
    Ok(())
}

/// Calibration slope: b in logit P(y) = a + b·logit(p) (ideal 1). Returns (slope, intercept).
pub fn calibration_slope<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<(f64, f64)> {
    let (y, w) = target(y_true, y_prob, pos_label, w)?;
    need_both(&y, "The calibration slope")?;
    let x: Vec<f64> = y_prob.iter().map(|&p| logit(p)).collect();
    let b = logistic_fit(&y, &x, &w, false)?;
    Ok((b[1], b[0]))
}

/// Calibration-in-the-large: a in logit P(y) = a + logit(p) (ideal 0).
pub fn calibration_intercept<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<f64> {
    let (y, w) = target(y_true, y_prob, pos_label, w)?;
    need_both(&y, "The calibration intercept")?;
    let x: Vec<f64> = y_prob.iter().map(|&p| logit(p)).collect();
    Ok(logistic_fit(&y, &x, &w, true)?[0])
}

/// `np.array_split(order, k)`: k nearly equal contiguous groups (the first n mod k one longer).
pub(crate) fn array_split<T: Copy>(v: &[T], k: usize) -> Vec<Vec<T>> {
    let (q, r) = (v.len() / k, v.len() % k);
    let mut out = Vec::with_capacity(k);
    let mut pos = 0;
    for i in 0..k {
        let len = q + usize::from(i < r);
        out.push(v[pos..pos + len].to_vec());
        pos += len;
    }
    out
}

/// Hosmer–Lemeshow goodness-of-fit test over `n_groups` risk groups: (statistic, p-value on n_groups − 2 df).
pub fn hosmer_lemeshow<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    n_groups: usize,
    pos_label: Option<&L>,
) -> Result<(f64, f64)> {
    let (y, _) = target(y_true, y_prob, pos_label, None)?;
    if n_groups < 3 {
        return invalid("n_groups must be an integer of at least 3.");
    }
    if y.len() < n_groups {
        return invalid(format!(
            "Need at least n_groups={n_groups} observations; got {}.",
            y.len()
        ));
    }
    let order =
        evalsuite_core::curve::descending_order(&y_prob.iter().map(|p| -p).collect::<Vec<_>>());
    let mut stat = Vec::with_capacity(n_groups);
    for g in array_split(&order, n_groups) {
        let o: f64 = crate::util::np_sum(&g.iter().map(|&i| y[i]).collect::<Vec<_>>());
        let e: f64 = crate::util::np_sum(&g.iter().map(|&i| y_prob[i]).collect::<Vec<_>>());
        let var = e * (1.0 - e / g.len() as f64);
        if var <= 0.0 {
            return undefined(
                "A risk group has expected count 0 or equal to its size; use fewer groups.",
            );
        }
        stat.push((o - e) * (o - e) / var);
    }
    let s = crate::util::np_sum(&stat);
    Ok((s, chi2_sf(s, (n_groups - 2) as f64)))
}
