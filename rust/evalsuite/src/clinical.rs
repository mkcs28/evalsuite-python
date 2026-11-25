//! Diagnostic-accuracy and decision-curve metrics for binary clinical outcomes, as EvalSuite's Python
//! `evalsuite.clinical`.
//!
//! Likelihood and odds ratios that divide by zero are reported as `+∞` (positive numerator) or NaN (0/0),
//! never replaced by 0, since a ratio of 0 means the opposite of an infinite one.

use crate::classification::{check_probs, Ctx, Kind, ZeroDivision};
use crate::error::{invalid, unsupported, Result};
use crate::labels::Label;

/// Weighted (TP, FP, FN, TN) of a binary test against a binary reference standard.
pub fn binary_counts<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    sample_weight: Option<&[f64]>,
) -> Result<(f64, f64, f64, f64)> {
    let ctx = Ctx::new(y_true, Some(y_pred), None, sample_weight)?;
    if ctx.kind != Kind::Binary {
        return unsupported(
            "Diagnostic metrics are defined for binary outcomes; this target is multiclass.",
        );
    }
    let total: f64 = crate::util::np_sum(&ctx.w);
    match ctx.positive(pos_label)? {
        None => Ok((0.0, 0.0, 0.0, total)),
        Some(i) => {
            let c = ctx.counts()?;
            Ok((c.tp[i], c.fp[i], c.fn_[i], c.tn[i]))
        }
    }
}

fn zd(num: f64, den: f64, z: ZeroDivision) -> f64 {
    if den == 0.0 {
        match z {
            ZeroDivision::Warn => 0.0,
            ZeroDivision::Value(v) => v,
        }
    } else {
        num / den
    }
}

fn ratio(num: f64, den: f64) -> f64 {
    if den != 0.0 {
        num / den
    } else if num > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

fn need_both(tp: f64, fp: f64, fn_: f64, tn: f64, what: &str) -> Result<()> {
    if tp + fn_ == 0.0 || tn + fp == 0.0 {
        return invalid(format!(
            "{what} needs both people with and without the condition in y_true; got {} positive and {} negative.",
            tp + fn_,
            tn + fp
        ));
    }
    Ok(())
}

fn sens_spec(tp: f64, fp: f64, fn_: f64, tn: f64) -> (f64, f64) {
    let s = if tp + fn_ != 0.0 {
        tp / (tp + fn_)
    } else {
        f64::NAN
    };
    let p = if tn + fp != 0.0 {
        tn / (tn + fp)
    } else {
        f64::NAN
    };
    (s, p)
}

/// Sensitivity (true positive rate).
pub fn sensitivity<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
    z: ZeroDivision,
) -> Result<f64> {
    let (tp, _, fn_, _) = binary_counts(y_true, y_pred, pos_label, w)?;
    Ok(zd(tp, tp + fn_, z))
}

/// Specificity (true negative rate).
pub fn specificity<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
    z: ZeroDivision,
) -> Result<f64> {
    let (_, fp, _, tn) = binary_counts(y_true, y_pred, pos_label, w)?;
    Ok(zd(tn, tn + fp, z))
}

/// Positive predictive value.
pub fn ppv<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
    z: ZeroDivision,
) -> Result<f64> {
    let (tp, fp, _, _) = binary_counts(y_true, y_pred, pos_label, w)?;
    Ok(zd(tp, tp + fp, z))
}

/// Negative predictive value.
pub fn npv<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
    z: ZeroDivision,
) -> Result<f64> {
    let (_, _, fn_, tn) = binary_counts(y_true, y_pred, pos_label, w)?;
    Ok(zd(tn, tn + fn_, z))
}

/// Positive likelihood ratio: sensitivity / (1 − specificity).
pub fn lr_positive<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<f64> {
    let (tp, fp, fn_, tn) = binary_counts(y_true, y_pred, pos_label, w)?;
    need_both(tp, fp, fn_, tn, "LR+")?;
    let (s, p) = sens_spec(tp, fp, fn_, tn);
    Ok(ratio(s, 1.0 - p))
}

/// Negative likelihood ratio: (1 − sensitivity) / specificity.
pub fn lr_negative<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<f64> {
    let (tp, fp, fn_, tn) = binary_counts(y_true, y_pred, pos_label, w)?;
    need_both(tp, fp, fn_, tn, "LR−")?;
    let (s, p) = sens_spec(tp, fp, fn_, tn);
    Ok(ratio(1.0 - s, p))
}

/// Diagnostic odds ratio (TP·TN)/(FP·FN); `correction` (e.g. 0.5) is added to every cell when one is zero.
pub fn diagnostic_odds_ratio<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
    correction: f64,
) -> Result<f64> {
    let (mut tp, mut fp, mut fn_, mut tn) = binary_counts(y_true, y_pred, pos_label, w)?;
    need_both(tp, fp, fn_, tn, "The diagnostic odds ratio")?;
    if correction < 0.0 {
        return invalid("correction must be 0 or positive (0.5 is the usual choice).");
    }
    if correction != 0.0 && tp.min(fp).min(fn_).min(tn) == 0.0 {
        tp += correction;
        fp += correction;
        fn_ += correction;
        tn += correction;
    }
    Ok(ratio(tp * tn, fp * fn_))
}

/// Youden's J: sensitivity + specificity − 1.
pub fn youden_j<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<f64> {
    let (tp, fp, fn_, tn) = binary_counts(y_true, y_pred, pos_label, w)?;
    need_both(tp, fp, fn_, tn, "Youden's J")?;
    let (s, p) = sens_spec(tp, fp, fn_, tn);
    Ok(s + p - 1.0)
}

fn risk<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let ctx = Ctx::new(y_true, None, None, w)?;
    if ctx.kind != Kind::Binary {
        return unsupported("Decision curves are defined for binary outcomes.");
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

fn check_threshold(t: f64) -> Result<()> {
    if !(t.is_finite() && t > 0.0 && t < 1.0) {
        return invalid("Threshold probabilities must be strictly between 0 and 1.");
    }
    Ok(())
}

/// Net benefit of treating when the predicted risk is at least `threshold` (Vickers & Elkin).
pub fn net_benefit<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    threshold: f64,
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<f64> {
    let (y, w) = risk(y_true, y_prob, pos_label, w)?;
    check_threshold(threshold)?;
    let n = crate::util::np_sum(&w);
    let (mut tp, mut fp) = (Vec::new(), Vec::new());
    for i in 0..y.len() {
        if y_prob[i] >= threshold {
            tp.push(w[i] * y[i]);
            fp.push(w[i] * (1.0 - y[i]));
        }
    }
    let (tp, fp) = (crate::util::np_sum(&tp), crate::util::np_sum(&fp));
    Ok(tp / n - fp / n * (threshold / (1.0 - threshold)))
}

/// Decision curve: net benefit of the model and of treating everyone at each threshold.
pub fn decision_curve<L: Label>(
    y_true: &[L],
    y_prob: &[f64],
    thresholds: &[f64],
    pos_label: Option<&L>,
    w: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (y, w) = risk(y_true, y_prob, pos_label, w)?;
    if thresholds.is_empty() {
        return invalid("Threshold probabilities must be strictly between 0 and 1.");
    }
    for &t in thresholds {
        check_threshold(t)?;
    }
    let n = crate::util::np_sum(&w);
    let mut order: Vec<usize> = (0..y.len()).collect();
    order.sort_by(|&a, &b| {
        y_prob[a]
            .partial_cmp(&y_prob[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted: Vec<f64> = order.iter().map(|&i| y_prob[i]).collect();
    let mut below_pos = vec![0.0];
    let mut below_neg = vec![0.0];
    for &i in &order {
        below_pos.push(below_pos.last().unwrap() + w[i] * y[i]);
        below_neg.push(below_neg.last().unwrap() + w[i] * (1.0 - y[i]));
    }
    let (pos_all, neg_all) = (*below_pos.last().unwrap(), *below_neg.last().unwrap());
    let prevalence = pos_all / n;
    let mut model = Vec::with_capacity(thresholds.len());
    let mut all = Vec::with_capacity(thresholds.len());
    for &t in thresholds {
        let k = sorted.partition_point(|&p| p < t);
        let odds = t / (1.0 - t);
        model.push((pos_all - below_pos[k]) / n - (neg_all - below_neg[k]) / n * odds);
        all.push(prevalence - (1.0 - prevalence) * odds);
    }
    Ok((model, all))
}

/// A diagnostic-accuracy report: the 2×2 table and every measure with its confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticReport {
    /// (TP, FP, FN, TN).
    pub table: (u64, u64, u64, u64),
    pub sensitivity: crate::stats::Interval,
    pub specificity: crate::stats::Interval,
    pub ppv: crate::stats::Interval,
    pub npv: crate::stats::Interval,
    /// Log method of Simel et al. (1991).
    pub lr_positive: crate::stats::Interval,
    pub lr_negative: crate::stats::Interval,
    /// Woolf's log method, with 0.5 added to every cell when one is zero.
    pub diagnostic_odds_ratio: crate::stats::Interval,
    /// Wald interval from independent sensitivity and specificity variances, clipped to [−1, 1].
    pub youden_j: crate::stats::Interval,
    pub accuracy: crate::stats::Interval,
    pub prevalence: crate::stats::Interval,
}

/// Diagnostic accuracy of a binary test with intervals (as the Python package's `diagnostic_report`);
/// proportions use `method` (Wilson by default).
pub fn diagnostic_report<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    pos_label: Option<&L>,
    level: f64,
    method: crate::stats::ProportionMethod,
) -> Result<DiagnosticReport> {
    use crate::stats::{proportion_ci, Interval};
    if !(level > 0.0 && level < 1.0) {
        return invalid("level must be between 0 and 1, for example 0.95.");
    }
    let (tp, fp, fn_, tn) = binary_counts(y_true, y_pred, pos_label, None)?;
    need_both(tp, fp, fn_, tn, "A diagnostic report")?;
    let (a, b, c, d) = (
        tp.round() as u64,
        fp.round() as u64,
        fn_.round() as u64,
        tn.round() as u64,
    );
    let z = crate::special::norm_ppf(1.0 - (1.0 - level) / 2.0);
    let nan = Interval {
        estimate: f64::NAN,
        low: f64::NAN,
        high: f64::NAN,
    };
    let prop = |k: u64, n: u64| {
        if n == 0 {
            Ok(nan)
        } else {
            proportion_ci(k, n, level, method)
        }
    };
    let log_ci = |est: f64, se: f64| {
        if est.is_finite() && est > 0.0 && se.is_finite() {
            Interval {
                estimate: est,
                low: (est.ln() - z * se).exp(),
                high: (est.ln() + z * se).exp(),
            }
        } else {
            Interval {
                estimate: est,
                low: f64::NAN,
                high: f64::NAN,
            }
        }
    };
    let (n1, n0, n) = (a + c, b + d, a + b + c + d);
    let (sens, spec) = (a as f64 / n1 as f64, d as f64 / n0 as f64);
    let lr_pos = if spec < 1.0 {
        sens / (1.0 - spec)
    } else if sens > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let lr_neg = if spec > 0.0 {
        (1.0 - sens) / spec
    } else if sens < 1.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let inv = |x: u64| 1.0 / x as f64;
    let se_pos = if a > 0 && b > 0 {
        (inv(a) - inv(n1) + inv(b) - inv(n0)).sqrt()
    } else {
        f64::NAN
    };
    let se_neg = if c > 0 && d > 0 {
        (inv(c) - inv(n1) + inv(d) - inv(n0)).sqrt()
    } else {
        f64::NAN
    };
    let corr = if a.min(b).min(c).min(d) == 0 {
        0.5
    } else {
        0.0
    };
    let (ca, cb, cc, cd) = (
        a as f64 + corr,
        b as f64 + corr,
        c as f64 + corr,
        d as f64 + corr,
    );
    let dor = ca * cd / (cb * cc);
    let se_dor = (1.0 / ca + 1.0 / cb + 1.0 / cc + 1.0 / cd).sqrt();
    let j = sens + spec - 1.0;
    let se_j = (sens * (1.0 - sens) / n1 as f64 + spec * (1.0 - spec) / n0 as f64).sqrt();
    Ok(DiagnosticReport {
        table: (a, b, c, d),
        sensitivity: prop(a, n1)?,
        specificity: prop(d, n0)?,
        ppv: prop(a, a + b)?,
        npv: prop(d, c + d)?,
        lr_positive: log_ci(lr_pos, se_pos),
        lr_negative: log_ci(lr_neg, se_neg),
        diagnostic_odds_ratio: log_ci(dor, se_dor),
        youden_j: Interval {
            estimate: j,
            low: (j - z * se_j).max(-1.0),
            high: (j + z * se_j).min(1.0),
        },
        accuracy: prop(a + d, n)?,
        prevalence: prop(n1, n)?,
    })
}
