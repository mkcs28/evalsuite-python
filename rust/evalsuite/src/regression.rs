//! Regression metrics, as EvalSuite's Python `evalsuite.regression` (scikit-learn compatible).
//!
//! Single-output functions take `&[f64]` slices; [`multioutput`] has the same metrics for 2-D targets (one
//! row per observation, one column per output) with uniform, raw or weighted averaging over outputs.

use crate::classification::Score;
use crate::error::{invalid, undefined, Result};
use crate::util::{np_sum, same_len, weighted_mean, weights};

struct In<'a> {
    t: &'a [f64],
    p: &'a [f64],
    w: Vec<f64>,
    weighted: bool,
}

fn inputs<'a>(t: &'a [f64], p: &'a [f64], w: Option<&[f64]>) -> Result<In<'a>> {
    if t.is_empty() {
        return invalid("y_true is empty. Provide at least one example.");
    }
    same_len(t.len(), p.len(), "y_true and y_pred")?;
    for (x, name) in [(t, "y_true"), (p, "y_pred")] {
        if x.iter().any(|v| !v.is_finite()) {
            return invalid(format!("{name} contains NaN or infinite values."));
        }
    }
    Ok(In { t, p, w: weights(w, t.len())?, weighted: w.is_some() })
}

impl In<'_> {
    fn mean(&self, v: &[f64]) -> f64 {
        if self.weighted {
            weighted_mean(v, &self.w)
        } else {
            np_sum(v) / v.len() as f64
        }
    }
    fn err(&self) -> Vec<f64> {
        self.p.iter().zip(self.t).map(|(p, t)| p - t).collect()
    }
}

/// Mean absolute error.
pub fn mae(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    Ok(i.mean(&i.err().iter().map(|e| e.abs()).collect::<Vec<_>>()))
}

/// Mean squared error.
pub fn mse(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    Ok(i.mean(&i.err().iter().map(|e| e * e).collect::<Vec<_>>()))
}

/// Root mean squared error (rescaled when squaring would overflow).
pub fn rmse(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let e = i.err();
    let r = i.mean(&e.iter().map(|e| e * e).collect::<Vec<_>>()).sqrt();
    if r.is_finite() {
        return Ok(r);
    }
    let scale = e.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    Ok(scale * i.mean(&e.iter().map(|x| (x / scale) * (x / scale)).collect::<Vec<_>>()).sqrt())
}

/// Coefficient of determination R² (1 for a perfect fit of a constant target, 0 for an imperfect one).
pub fn r2(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    Ok(r2_of(&i))
}

fn r2_of(i: &In<'_>) -> f64 {
    let m = i.mean(i.t);
    let e = i.err();
    let (res, tot): (Vec<f64>, Vec<f64>) = if i.weighted {
        (
            e.iter().zip(&i.w).map(|(e, w)| w * (e * e)).collect(),
            i.t.iter().zip(&i.w).map(|(t, w)| w * (t - m) * (t - m)).collect(),
        )
    } else {
        (e.iter().map(|e| e * e).collect(), i.t.iter().map(|t| (t - m) * (t - m)).collect())
    };
    let (ss_res, ss_tot) = (np_sum(&res), np_sum(&tot));
    if ss_tot == 0.0 {
        return if ss_res == 0.0 { 1.0 } else { 0.0 };
    }
    1.0 - ss_res / ss_tot
}

/// Adjusted R² for `n_features` predictors.
pub fn adjusted_r2(y_true: &[f64], y_pred: &[f64], n_features: usize, w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let n = i.t.len();
    if n as i64 - n_features as i64 - 1 <= 0 {
        return undefined(format!(
            "Adjusted R² needs more observations than predictors + 1 (n={n}, n_features={n_features})."
        ));
    }
    let r = r2_of(&i);
    Ok(1.0 - (1.0 - r) * (n - 1) as f64 / (n - n_features - 1) as f64)
}

/// Mean absolute percentage error (as a fraction); undefined when `y_true` contains zeros.
pub fn mape(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let zeros = i.t.iter().filter(|&&t| t == 0.0).count();
    if zeros > 0 {
        return undefined(format!("MAPE is undefined because y_true contains {zeros} zero value(s)."));
    }
    Ok(i.mean(&i.err().iter().zip(i.t).map(|(e, t)| e.abs() / t.abs()).collect::<Vec<_>>()))
}

/// Symmetric MAPE: mean of 2|e| / (|y| + |ŷ|) (0 where both are 0).
pub fn smape(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let terms: Vec<f64> = i
        .err()
        .iter()
        .zip(i.t.iter().zip(i.p))
        .map(|(e, (t, p))| {
            let den = t.abs() + p.abs();
            if den != 0.0 {
                2.0 * e.abs() / den
            } else {
                0.0
            }
        })
        .collect();
    Ok(i.mean(&terms))
}

fn log_sq(i: &In<'_>, name: &str) -> Result<Vec<f64>> {
    if i.t.iter().chain(i.p).any(|&v| v < 0.0) {
        return undefined(format!("{name} needs non-negative y_true and y_pred (it compares log(1 + y))."));
    }
    Ok(i.p.iter().zip(i.t).map(|(p, t)| (p.ln_1p() - t.ln_1p()).powi(2)).collect())
}

/// Mean squared logarithmic error.
pub fn msle(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let v = log_sq(&i, "MSLE")?;
    Ok(i.mean(&v))
}

/// Root mean squared logarithmic error.
pub fn rmsle(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let v = log_sq(&i, "RMSLE")?;
    Ok(i.mean(&v).sqrt())
}

fn median(v: &[f64]) -> f64 {
    let s = crate::util::sorted(v);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

fn weighted_median(v: &[f64], w: &[f64]) -> f64 {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cw = Vec::with_capacity(v.len());
    let mut acc = 0.0;
    for &i in &order {
        acc += w[i];
        cw.push(acc);
    }
    let half = 0.5 * acc;
    let k = cw.partition_point(|&c| c < half);
    v[order[k.min(order.len() - 1)]]
}

/// Median absolute error (weighted median with sample weights).
pub fn median_absolute_error(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let a: Vec<f64> = i.err().iter().map(|e| e.abs()).collect();
    Ok(if i.weighted { weighted_median(&a, &i.w) } else { median(&a) })
}

/// Explained variance score.
pub fn explained_variance(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    let resid: Vec<f64> = i.t.iter().zip(i.p).map(|(t, p)| t - p).collect();
    let mr = i.mean(&resid);
    let var_res = i.mean(&resid.iter().map(|r| (r - mr) * (r - mr)).collect::<Vec<_>>());
    let mt = i.mean(i.t);
    let var_true = i.mean(&i.t.iter().map(|t| (t - mt) * (t - mt)).collect::<Vec<_>>());
    if var_true == 0.0 {
        return Ok(if var_res == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - var_res / var_true)
}

/// Largest absolute error.
pub fn max_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    let i = inputs(y_true, y_pred, None)?;
    Ok(i.err().iter().fold(0.0f64, |m, e| m.max(e.abs())))
}

/// Mean bias error (mean of ŷ − y).
pub fn mean_bias_error(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    let i = inputs(y_true, y_pred, w)?;
    Ok(i.mean(&i.err()))
}

/// Quantile (pinball) loss at quantile `alpha` ∈ (0, 1).
pub fn quantile_loss(y_true: &[f64], y_pred: &[f64], alpha: f64, w: Option<&[f64]>) -> Result<f64> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return invalid("alpha must be strictly between 0 and 1.");
    }
    let i = inputs(y_true, y_pred, w)?;
    let loss: Vec<f64> = i.t.iter().zip(i.p).map(|(t, p)| {
        let d = t - p;
        (alpha * d).max((alpha - 1.0) * d)
    }).collect();
    Ok(i.mean(&loss))
}

/// Huber loss with threshold `delta`.
pub fn huber_loss(y_true: &[f64], y_pred: &[f64], delta: f64, w: Option<&[f64]>) -> Result<f64> {
    if !(delta > 0.0) {
        return invalid("delta must be positive.");
    }
    let i = inputs(y_true, y_pred, w)?;
    let loss: Vec<f64> = i.err().iter().map(|e| {
        let a = e.abs();
        if a <= delta {
            0.5 * a * a
        } else {
            delta * (a - 0.5 * delta)
        }
    }).collect();
    Ok(i.mean(&loss))
}

fn relative(i: &In<'_>, power: i32, name: &str) -> Result<f64> {
    let m = i.mean(i.t);
    let num: Vec<f64> = i.err().iter().zip(&i.w).map(|(e, w)| w * e.abs().powi(power)).collect();
    let den: Vec<f64> = i.t.iter().zip(&i.w).map(|(t, w)| w * (t - m).abs().powi(power)).collect();
    let d = np_sum(&den);
    if d == 0.0 {
        return undefined(format!("{name} is undefined when y_true is constant (zero baseline error)."));
    }
    Ok(np_sum(&num) / d)
}

/// Relative absolute error (vs predicting the mean).
pub fn rae(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    relative(&inputs(y_true, y_pred, w)?, 1, "RAE")
}

/// Relative squared error (vs predicting the mean).
pub fn rse(y_true: &[f64], y_pred: &[f64], w: Option<&[f64]>) -> Result<f64> {
    relative(&inputs(y_true, y_pred, w)?, 2, "RSE")
}

/// Multi-output regression: one row per observation, one column per output.
pub mod multioutput {
    use super::*;

    /// How per-output values are combined.
    #[derive(Debug, Clone, PartialEq, Default)]
    pub enum Multioutput {
        #[default]
        UniformAverage,
        RawValues,
        Weights(Vec<f64>),
    }

    fn columns(rows: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let k = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != k) || k == 0 {
            return invalid("Every row must have the same, non-zero number of outputs.");
        }
        Ok((0..k).map(|j| rows.iter().map(|r| r[j]).collect()).collect())
    }

    fn apply(
        t: &[Vec<f64>],
        p: &[Vec<f64>],
        how: &Multioutput,
        f: impl Fn(&[f64], &[f64]) -> Result<f64>,
    ) -> Result<Score> {
        let (ct, cp) = (columns(t)?, columns(p)?);
        if ct.len() != cp.len() {
            return invalid("y_true and y_pred must have the same number of outputs.");
        }
        let per: Vec<f64> = ct.iter().zip(&cp).map(|(a, b)| f(a, b)).collect::<Result<_>>()?;
        match how {
            Multioutput::RawValues => Ok(Score::PerClass(per)),
            Multioutput::UniformAverage => Ok(Score::Scalar(np_sum(&per) / per.len() as f64)),
            Multioutput::Weights(w) => {
                if w.len() != per.len() || w.iter().any(|&x| x < 0.0) || np_sum(w) == 0.0 {
                    return invalid("multioutput weights must be non-negative, one per output, not all zero.");
                }
                Ok(Score::Scalar(weighted_mean(&per, w)))
            }
        }
    }

    macro_rules! multi {
        ($($name:ident),*) => {$(
            #[doc = concat!("Multi-output `", stringify!($name), "`.")]
            pub fn $name(t: &[Vec<f64>], p: &[Vec<f64>], w: Option<&[f64]>, how: &Multioutput) -> Result<Score> {
                apply(t, p, how, |a, b| super::$name(a, b, w))
            }
        )*};
    }
    multi!(mae, mse, rmse, r2, mape, smape, msle, rmsle, median_absolute_error, explained_variance, mean_bias_error, rae, rse);
}
