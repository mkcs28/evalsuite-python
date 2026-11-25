//! Uncertainty and selective prediction for LLM answers, as EvalSuite's Python `evalsuite.llmsys.uncertainty`:
//! per question, whether the answer was correct and the model's confidence in it.

use crate::calibration::array_split;
use crate::error::{invalid, Result};
use crate::util::{np_sum, same_len};

fn inputs(correct: &[bool], confidence: &[f64]) -> Result<()> {
    if correct.is_empty() {
        return invalid("correct is empty. Provide at least one example.");
    }
    same_len(correct.len(), confidence.len(), "correct and confidence")?;
    if confidence.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
        return invalid("confidence must contain finite values in [0, 1].");
    }
    Ok(())
}

/// Stable order by decreasing confidence (`np.argsort(-p, kind="mergesort")`).
fn by_confidence(p: &[f64]) -> Vec<usize> {
    evalsuite_core::curve::descending_order(p)
}

/// Risk–coverage curve: (coverage, risk, confidence threshold) at each distinct confidence, most confident first.
pub fn risk_coverage_curve(correct: &[bool], confidence: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    inputs(correct, confidence)?;
    let order = by_confidence(confidence);
    let n = correct.len();
    let (mut cov, mut risk, mut thr) = (Vec::new(), Vec::new(), Vec::new());
    let mut errors = 0.0f64;
    for (pos, &i) in order.iter().enumerate() {
        errors += f64::from(u8::from(!correct[i]));
        let last = pos + 1 == n || confidence[order[pos + 1]] != confidence[i];
        if last {
            let k = (pos + 1) as f64;
            cov.push(k / n as f64);
            risk.push(errors / k);
            thr.push(confidence[i]);
        }
    }
    Ok((cov, risk, thr))
}

/// Adaptive (equal-mass) calibration error over `n_bins` bins.
pub fn adaptive_calibration_error(correct: &[bool], confidence: &[f64], n_bins: usize) -> Result<f64> {
    inputs(correct, confidence)?;
    if n_bins < 1 {
        return invalid("n_bins must be a positive integer.");
    }
    let r = n_bins.min(correct.len());
    let mut order: Vec<usize> = (0..confidence.len()).collect();
    order.sort_by(|&a, &b| confidence[a].partial_cmp(&confidence[b]).unwrap_or(std::cmp::Ordering::Equal));
    let gaps: Vec<f64> = array_split(&order, r)
        .iter()
        .map(|g| {
            let c: Vec<f64> = g.iter().map(|&i| f64::from(u8::from(correct[i]))).collect();
            let p: Vec<f64> = g.iter().map(|&i| confidence[i]).collect();
            (np_sum(&c) / c.len() as f64 - np_sum(&p) / p.len() as f64).abs()
        })
        .collect();
    Ok(np_sum(&gaps) / gaps.len() as f64)
}

/// Error rate among the most confident `coverage` share of answers.
pub fn selective_risk(correct: &[bool], confidence: &[f64], coverage: f64) -> Result<f64> {
    if !(coverage > 0.0 && coverage <= 1.0) {
        return invalid("coverage must be in (0, 1].");
    }
    inputs(correct, confidence)?;
    let k = (coverage * correct.len() as f64).ceil() as usize;
    let order = by_confidence(confidence);
    Ok(order[..k].iter().filter(|&&i| !correct[i]).count() as f64 / k as f64)
}

/// Area under the risk–coverage curve, and its excess over the optimal ordering (E-AURC).
pub fn aurc(correct: &[bool], confidence: &[f64]) -> Result<(f64, f64)> {
    inputs(correct, confidence)?;
    let order = by_confidence(confidence);
    let n = correct.len();
    let mut cum = 0.0;
    let mut curve = Vec::with_capacity(n);
    for (pos, &i) in order.iter().enumerate() {
        cum += f64::from(u8::from(!correct[i]));
        curve.push(cum / (pos + 1) as f64);
    }
    let value = np_sum(&curve) / n as f64;
    let errors = correct.iter().filter(|c| !**c).count();
    let opt: Vec<f64> = (1..=n).map(|k| (k as f64 - (n - errors) as f64).max(0.0) / k as f64).collect();
    let opt = np_sum(&opt) / n as f64;
    Ok((value, value - opt))
}

/// Risk at the smallest curve point whose coverage reaches `coverage`.
pub fn risk_at_coverage(correct: &[bool], confidence: &[f64], coverage: f64) -> Result<f64> {
    if !(coverage > 0.0 && coverage <= 1.0) {
        return invalid("coverage must be in (0, 1].");
    }
    let (cov, risk, _) = risk_coverage_curve(correct, confidence)?;
    let i = cov.partition_point(|&c| c < coverage - 1e-12);
    Ok(risk[i.min(risk.len() - 1)])
}

/// Largest coverage whose risk is at most `risk` (0 if none).
pub fn coverage_at_risk(correct: &[bool], confidence: &[f64], risk: f64) -> Result<f64> {
    if !(0.0..1.0).contains(&risk) {
        return invalid("risk must be in [0, 1).");
    }
    let (cov, r, _) = risk_coverage_curve(correct, confidence)?;
    Ok(r.iter().rposition(|&x| x <= risk + 1e-12).map_or(0.0, |i| cov[i]))
}

/// AUROC of confidence for separating correct from incorrect answers.
pub fn confidence_accuracy_correlation(correct: &[bool], confidence: &[f64]) -> Result<f64> {
    inputs(correct, confidence)?;
    let n1 = correct.iter().filter(|c| **c).count();
    if n1 == 0 || n1 == correct.len() {
        return invalid("Need both correct and incorrect answers.");
    }
    let ranks = evalsuite_core::ranks::midranks(confidence);
    let s: f64 = np_sum(&ranks.iter().zip(correct).filter(|(_, c)| **c).map(|(r, _)| *r).collect::<Vec<_>>());
    let n1f = n1 as f64;
    Ok((s - n1f * (n1f + 1.0) / 2.0) / (n1f * (correct.len() - n1) as f64))
}
