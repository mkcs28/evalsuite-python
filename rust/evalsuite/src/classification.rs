//! Classification metrics for binary, multiclass and multilabel targets, as EvalSuite's Python
//! `evalsuite.classification` (scikit-learn compatible definitions).
//!
//! Single-label functions take label slices (`&[L]` for any [`Label`]); probability metrics take
//! `P(positive)` for binary tasks ([`binary`] functions) or one probability row per observation for
//! multiclass tasks. Multilabel indicator matrices are handled in [`multilabel`].
//!
//! ```
//! use evalsuite::classification::{self, Average, Opts};
//! let y_true = [0, 1, 1, 0, 1];
//! let y_pred = [0, 1, 0, 0, 1];
//! let f1 = classification::f1(&y_true, &y_pred, &Opts::default()).unwrap();
//! assert!((f1.value().unwrap() - 0.8).abs() < 1e-12);
//! let macro_f1 = classification::f1(&y_true, &y_pred, &Opts::default().average(Average::Macro)).unwrap();
//! assert!(macro_f1.value().unwrap() > 0.79);
//! ```

use std::collections::{BTreeSet, HashMap};

use crate::error::{invalid, undefined, unsupported, Result};
use crate::labels::Label;
use crate::util::{np_sum, weighted_mean, weights};

/// How per-class values are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Average {
    /// `Binary` for binary targets, `Macro` otherwise.
    #[default]
    Auto,
    /// The positive class only (binary targets).
    Binary,
    /// Pool the counts of every class.
    Micro,
    /// Unweighted mean over classes.
    Macro,
    /// Mean over classes weighted by support.
    Weighted,
    /// Mean over observations (multilabel only).
    Samples,
    /// No averaging: one value per class.
    None,
}

/// The value used when a ratio's denominator is zero. `Warn` gives 0 (Python warns; Rust returns 0 silently,
/// matching scikit-learn's value); `Value(v)` gives `v` (0, 1 or NaN).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ZeroDivision {
    #[default]
    Warn,
    Value(f64),
}

impl ZeroDivision {
    fn value(self) -> f64 {
        match self {
            ZeroDivision::Warn => 0.0,
            ZeroDivision::Value(v) => v,
        }
    }
}

/// A metric value: one number, or one per class (in label order) with `Average::None`.
#[derive(Debug, Clone, PartialEq)]
pub enum Score {
    Scalar(f64),
    PerClass(Vec<f64>),
}

impl Score {
    /// The scalar value, if averaged.
    pub fn value(&self) -> Option<f64> {
        match self {
            Score::Scalar(v) => Some(*v),
            Score::PerClass(_) => None,
        }
    }

    /// The per-class values, if not averaged.
    pub fn per_class(&self) -> Option<&[f64]> {
        match self {
            Score::Scalar(_) => None,
            Score::PerClass(v) => Some(v),
        }
    }
}

/// Options shared by the classification metrics (all optional; `Opts::default()` matches the Python defaults).
#[derive(Debug, Clone)]
pub struct Opts<L> {
    pub average: Average,
    /// Explicit label order (and set); default: the sorted labels present in `y_true` or `y_pred`.
    pub labels: Option<Vec<L>>,
    /// The positive class for binary averaging; default: see [`Label::default_positive`].
    pub pos_label: Option<L>,
    pub sample_weight: Option<Vec<f64>>,
    pub zero_division: ZeroDivision,
}

impl<L> Default for Opts<L> {
    fn default() -> Self {
        Opts {
            average: Average::Auto,
            labels: None,
            pos_label: None,
            sample_weight: None,
            zero_division: ZeroDivision::Warn,
        }
    }
}

impl<L> Opts<L> {
    pub fn average(mut self, a: Average) -> Self {
        self.average = a;
        self
    }
    pub fn labels(mut self, l: Vec<L>) -> Self {
        self.labels = Some(l);
        self
    }
    pub fn pos_label(mut self, l: L) -> Self {
        self.pos_label = Some(l);
        self
    }
    pub fn sample_weight(mut self, w: Vec<f64>) -> Self {
        self.sample_weight = Some(w);
        self
    }
    pub fn zero_division(mut self, z: ZeroDivision) -> Self {
        self.zero_division = z;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Binary,
    Multiclass,
    Multilabel,
}

/// Validated inputs with label encoding and weighted per-class counts.
pub(crate) struct Ctx<L> {
    pub kind: Kind,
    pub labels: Vec<L>,
    pub true_idx: Vec<usize>,
    pub pred_idx: Option<Vec<usize>>,
    pub w: Vec<f64>,
}

pub(crate) struct Counts {
    pub tp: Vec<f64>,
    pub fp: Vec<f64>,
    pub fn_: Vec<f64>,
    pub tn: Vec<f64>,
    pub support: Vec<f64>,
}

impl<L: Label> Ctx<L> {
    pub fn new(
        y_true: &[L],
        y_pred: Option<&[L]>,
        labels: Option<&[L]>,
        w: Option<&[f64]>,
    ) -> Result<Self> {
        let n = y_true.len();
        if n == 0 {
            return invalid("y_true is empty. Provide at least one example.");
        }
        if let Some(p) = y_pred {
            if p.len() != n {
                return invalid(format!(
                    "y_true and y_pred must contain the same number of observations. Received {n} and {}.",
                    p.len()
                ));
            }
        }
        // One hashed pass assigns first-seen ids; only the distinct labels are sorted afterwards.
        let mut ids: HashMap<&L, usize, crate::util::FastHash> = HashMap::default();
        let mut distinct: Vec<&L> = Vec::new();
        fn encode<'a, L: Label>(
            v: &'a [L],
            ids: &mut HashMap<&'a L, usize, crate::util::FastHash>,
            distinct: &mut Vec<&'a L>,
        ) -> Vec<usize> {
            let mut last: Option<(&L, usize)> = None;
            v.iter()
                .map(|x| {
                    if let Some((l, i)) = last {
                        if l == x {
                            return i;
                        }
                    }
                    // A handful of classes (the common case): a linear scan beats hashing.
                    if distinct.len() <= 8 {
                        if let Some(i) = distinct.iter().position(|d| *d == x) {
                            last = Some((x, i));
                            return i;
                        }
                    }
                    let n = distinct.len();
                    let i = *ids.entry(x).or_insert_with(|| {
                        distinct.push(x);
                        n
                    });
                    last = Some((x, i));
                    i
                })
                .collect()
        }
        let raw_true = encode(y_true, &mut ids, &mut distinct);
        let n_true_unique = distinct.len();
        let raw_pred = y_pred.map(|p| encode(p, &mut ids, &mut distinct));
        let labels: Vec<L> = match labels {
            Some(l) => {
                let set: BTreeSet<&L> = l.iter().collect();
                if set.len() != l.len() {
                    return invalid("labels contains duplicates.");
                }
                let missing: Vec<&L> = distinct
                    .iter()
                    .copied()
                    .filter(|x| !set.contains(x))
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if !missing.is_empty() {
                    return invalid(format!(
                        "y_true/y_pred contain label(s) {missing:?} that are not in labels. Include every label that \
                         occurs, or filter the data first."
                    ));
                }
                l.to_vec()
            }
            None => {
                let mut d: Vec<&L> = distinct.clone();
                d.sort();
                d.into_iter().cloned().collect()
            }
        };
        let kind = if n_true_unique <= 2 && labels.len() <= 2 {
            Kind::Binary
        } else {
            Kind::Multiclass
        };
        let position: HashMap<&L, usize, crate::util::FastHash> =
            labels.iter().enumerate().map(|(i, l)| (l, i)).collect();
        let remap: Vec<usize> = distinct.iter().map(|l| position[l]).collect();
        let true_idx = raw_true.into_iter().map(|i| remap[i]).collect();
        let pred_idx = raw_pred.map(|p| p.into_iter().map(|i| remap[i]).collect());
        Ok(Ctx {
            kind,
            labels: labels.clone(),
            true_idx,
            pred_idx,
            w: weights(w, n)?,
        })
    }

    pub fn n(&self) -> usize {
        self.true_idx.len()
    }

    pub fn pred(&self) -> Result<&[usize]> {
        match &self.pred_idx {
            Some(p) => Ok(p),
            None => invalid("This metric needs y_pred (predicted labels)."),
        }
    }

    /// Weighted confusion matrix, rows = true, columns = predicted (row-major k × k).
    pub fn confusion(&self) -> Result<Vec<f64>> {
        let k = self.labels.len();
        let p = self.pred()?;
        let mut cm = vec![0.0f64; k * k];
        for (i, (&t, &q)) in self.true_idx.iter().zip(p).enumerate() {
            cm[t * k + q] += self.w[i];
        }
        Ok(cm)
    }

    pub fn counts(&self) -> Result<Counts> {
        let k = self.labels.len();
        let cm = self.confusion()?;
        let total = np_sum(&cm);
        let mut c = Counts {
            tp: vec![0.0; k],
            fp: vec![0.0; k],
            fn_: vec![0.0; k],
            tn: vec![0.0; k],
            support: vec![0.0; k],
        };
        for j in 0..k {
            let tp = cm[j * k + j];
            let col: f64 = (0..k).map(|i| cm[i * k + j]).sum();
            let row: f64 = (0..k).map(|i| cm[j * k + i]).sum();
            c.tp[j] = tp;
            c.fp[j] = col - tp;
            c.fn_[j] = row - tp;
            c.tn[j] = total - tp - (col - tp) - (row - tp);
            c.support[j] = tp + (row - tp);
        }
        Ok(c)
    }

    /// Index of the positive class (None when the only label is not the positive one).
    pub fn positive(&self, pos_label: Option<&L>) -> Result<Option<usize>> {
        let pos = match pos_label {
            Some(p) => p.clone(),
            None => match L::default_positive(&self.labels) {
                Some(p) => p,
                None => {
                    return invalid(format!(
                        "Labels are {:?}; specify which one is positive with pos_label.",
                        self.labels
                    ))
                }
            },
        };
        if let Some(i) = self.labels.iter().position(|l| *l == pos) {
            return Ok(Some(i));
        }
        if self.labels.len() <= 1 {
            return Ok(None);
        }
        invalid(format!(
            "pos_label={pos:?} is not one of the labels {:?}.",
            self.labels
        ))
    }

    pub fn resolve_average(&self, a: Average) -> Result<Average> {
        resolve_average(self.kind, a, self.labels.len())
    }
}

pub(crate) fn resolve_average(kind: Kind, a: Average, n_labels: usize) -> Result<Average> {
    match a {
        Average::Auto => Ok(if kind == Kind::Binary { Average::Binary } else { Average::Macro }),
        Average::Binary if kind != Kind::Binary => unsupported(format!(
            "average=Binary needs a binary target, but this target is {} ({n_labels} labels). Choose Macro, Micro or \
             Weighted, or None for per-class values.",
            if kind == Kind::Multilabel { "multilabel" } else { "multiclass" }
        )),
        Average::Samples if kind != Kind::Multilabel => unsupported("average=Samples is only defined for multilabel targets."),
        other => Ok(other),
    }
}

fn div(num: f64, den: f64, zd: ZeroDivision) -> f64 {
    if den == 0.0 {
        zd.value()
    } else {
        num / den
    }
}

/// A per-class count formula f(tp, fp, fn, tn).
pub(crate) type CountFn = dyn Fn(f64, f64, f64, f64, ZeroDivision) -> f64;

pub(crate) fn combine(
    avg: Average,
    c: &Counts,
    f: &CountFn,
    zd: ZeroDivision,
    positive: Option<Option<usize>>,
    total_weight: f64,
) -> Score {
    match avg {
        Average::Binary => match positive.flatten() {
            Some(i) => Score::Scalar(f(c.tp[i], c.fp[i], c.fn_[i], c.tn[i], zd)),
            None => Score::Scalar(f(0.0, 0.0, 0.0, total_weight, zd)),
        },
        Average::Micro => Score::Scalar(f(
            np_sum(&c.tp),
            np_sum(&c.fp),
            np_sum(&c.fn_),
            np_sum(&c.tn),
            zd,
        )),
        _ => {
            let per: Vec<f64> = (0..c.tp.len())
                .map(|j| f(c.tp[j], c.fp[j], c.fn_[j], c.tn[j], zd))
                .collect();
            match avg {
                Average::None => Score::PerClass(per),
                Average::Macro => Score::Scalar(np_sum(&per) / per.len() as f64),
                _ => {
                    let num: Vec<f64> = per.iter().zip(&c.support).map(|(a, b)| a * b).collect();
                    Score::Scalar(div(np_sum(&num), np_sum(&c.support), zd))
                }
            }
        }
    }
}

fn averaged<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>, f: &CountFn) -> Result<Score> {
    let ctx = Ctx::new(
        y_true,
        Some(y_pred),
        o.labels.as_deref(),
        o.sample_weight.as_deref(),
    )?;
    let avg = ctx.resolve_average(o.average)?;
    let positive = if avg == Average::Binary {
        Some(ctx.positive(o.pos_label.as_ref())?)
    } else {
        None
    };
    let c = ctx.counts()?;
    Ok(combine(
        avg,
        &c,
        f,
        o.zero_division,
        positive,
        np_sum(&ctx.w),
    ))
}

pub(crate) fn f_precision(tp: f64, fp: f64, _: f64, _: f64, z: ZeroDivision) -> f64 {
    div(tp, tp + fp, z)
}
pub(crate) fn f_recall(tp: f64, _: f64, fn_: f64, _: f64, z: ZeroDivision) -> f64 {
    div(tp, tp + fn_, z)
}
pub(crate) fn f_specificity(_: f64, fp: f64, _: f64, tn: f64, z: ZeroDivision) -> f64 {
    div(tn, tn + fp, z)
}
pub(crate) fn f_npv(_: f64, _: f64, fn_: f64, tn: f64, z: ZeroDivision) -> f64 {
    div(tn, tn + fn_, z)
}
pub(crate) fn f_jaccard(tp: f64, fp: f64, fn_: f64, _: f64, z: ZeroDivision) -> f64 {
    div(tp, tp + fp + fn_, z)
}
pub(crate) fn f_fbeta(beta: f64) -> impl Fn(f64, f64, f64, f64, ZeroDivision) -> f64 {
    let b2 = beta * beta;
    move |tp, fp, fn_, _, z| div((1.0 + b2) * tp, (1.0 + b2) * tp + b2 * fn_ + fp, z)
}

/// Proportion of correct predictions.
pub fn accuracy<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    if sample_weight.is_none() && !y_true.is_empty() && y_true.len() == y_pred.len() {
        // Unweighted: the share of equal pairs needs no label encoding (0/1 sums are exact in any order).
        let k = y_true.iter().zip(y_pred).filter(|(a, b)| a == b).count();
        return Ok(k as f64 / y_true.len() as f64);
    }
    let ctx = Ctx::new(y_true, Some(y_pred), None, sample_weight)?;
    let p = ctx.pred()?;
    let correct: Vec<f64> = ctx
        .true_idx
        .iter()
        .zip(p)
        .map(|(a, b)| f64::from(u8::from(a == b)))
        .collect();
    Ok(weighted_mean(&correct, &ctx.w))
}

/// Precision: TP / (TP + FP).
pub fn precision<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>) -> Result<Score> {
    averaged(y_true, y_pred, o, &f_precision)
}

/// Recall (sensitivity): TP / (TP + FN).
pub fn recall<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>) -> Result<Score> {
    averaged(y_true, y_pred, o, &f_recall)
}

/// Specificity: TN / (TN + FP).
pub fn specificity<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>) -> Result<Score> {
    averaged(y_true, y_pred, o, &f_specificity)
}

/// Negative predictive value: TN / (TN + FN).
pub fn npv<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>) -> Result<Score> {
    averaged(y_true, y_pred, o, &f_npv)
}

/// F1 score.
pub fn f1<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>) -> Result<Score> {
    averaged(y_true, y_pred, o, &f_fbeta(1.0))
}

/// F-beta score.
pub fn fbeta<L: Label>(y_true: &[L], y_pred: &[L], beta: f64, o: &Opts<L>) -> Result<Score> {
    if !(beta > 0.0 && beta.is_finite()) {
        return invalid("beta must be a positive number.");
    }
    averaged(y_true, y_pred, o, &f_fbeta(beta))
}

/// Jaccard index: TP / (TP + FP + FN).
pub fn jaccard<L: Label>(y_true: &[L], y_pred: &[L], o: &Opts<L>) -> Result<Score> {
    averaged(y_true, y_pred, o, &f_jaccard)
}

/// Balanced accuracy (mean recall over the classes present in `y_true`); `adjusted` rescales chance to 0.
pub fn balanced_accuracy<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    adjusted: bool,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    let ctx = Ctx::new(y_true, Some(y_pred), None, sample_weight)?;
    let c = ctx.counts()?;
    let r: Vec<f64> = (0..c.tp.len())
        .filter(|&j| c.support[j] > 0.0)
        .map(|j| c.tp[j] / c.support[j])
        .collect();
    let mut score = np_sum(&r) / r.len() as f64;
    if adjusted {
        let k = r.len();
        let chance = 1.0 / k as f64;
        score = if k > 1 {
            (score - chance) / (1.0 - chance)
        } else {
            0.0
        };
    }
    Ok(score)
}

/// Matthews correlation coefficient (Gorodkin's multiclass form).
pub fn mcc<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<f64> {
    let ctx = Ctx::new(y_true, Some(y_pred), None, sample_weight)?;
    let cm = ctx.confusion()?;
    let k = ctx.labels.len();
    let t: Vec<f64> = (0..k)
        .map(|i| (0..k).map(|j| cm[i * k + j]).sum())
        .collect();
    let p: Vec<f64> = (0..k)
        .map(|j| (0..k).map(|i| cm[i * k + j]).sum())
        .collect();
    let c: f64 = (0..k).map(|i| cm[i * k + i]).sum();
    let s = np_sum(&cm);
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let cov_tp = c * s - dot(&t, &p);
    let cov_pp = s * s - dot(&p, &p);
    let cov_tt = s * s - dot(&t, &t);
    Ok(div(cov_tp, (cov_tt * cov_pp).sqrt(), zero_division))
}

/// Weighting of disagreements for Cohen's kappa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KappaWeights {
    Unweighted,
    Linear,
    Quadratic,
}

/// Cohen's kappa, optionally linearly or quadratically weighted (ordinal labels, in label order).
pub fn cohen_kappa<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    weights: KappaWeights,
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
    zero_division: ZeroDivision,
) -> Result<f64> {
    let ctx = Ctx::new(y_true, Some(y_pred), labels, sample_weight)?;
    let cm = ctx.confusion()?;
    let k = ctx.labels.len();
    let total = np_sum(&cm);
    let rows: Vec<f64> = (0..k)
        .map(|i| (0..k).map(|j| cm[i * k + j]).sum())
        .collect();
    let cols: Vec<f64> = (0..k)
        .map(|j| (0..k).map(|i| cm[i * k + j]).sum())
        .collect();
    let wt = |i: usize, j: usize| -> f64 {
        let d = (i as f64 - j as f64).abs();
        match weights {
            KappaWeights::Unweighted => f64::from(u8::from(i != j)),
            KappaWeights::Linear => d,
            KappaWeights::Quadratic => d * d,
        }
    };
    let mut obs = Vec::with_capacity(k * k);
    let mut exp = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            obs.push(wt(i, j) * cm[i * k + j]);
            exp.push(wt(i, j) * (rows[i] * cols[j] / total));
        }
    }
    Ok(1.0 - div(np_sum(&obs), np_sum(&exp), zero_division))
}

/// Hamming loss (share of wrong labels).
pub fn hamming_loss<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    let ctx = Ctx::new(y_true, Some(y_pred), None, sample_weight)?;
    let p = ctx.pred()?;
    let wrong: Vec<f64> = ctx
        .true_idx
        .iter()
        .zip(p)
        .map(|(a, b)| f64::from(u8::from(a != b)))
        .collect();
    Ok(weighted_mean(&wrong, &ctx.w))
}

/// Normalisation of a confusion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalize {
    None,
    True,
    Pred,
    All,
}

/// Confusion matrix (rows: true, columns: predicted, in `labels` order) and the label order.
pub fn confusion_matrix<L: Label>(
    y_true: &[L],
    y_pred: &[L],
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
    normalize: Normalize,
) -> Result<(Vec<Vec<f64>>, Vec<L>)> {
    let ctx = Ctx::new(y_true, Some(y_pred), labels, sample_weight)?;
    let k = ctx.labels.len();
    let cm = ctx.confusion()?;
    let mut out: Vec<Vec<f64>> = (0..k).map(|i| cm[i * k..(i + 1) * k].to_vec()).collect();
    let total = np_sum(&cm);
    let cols: Vec<f64> = (0..k)
        .map(|j| (0..k).map(|i| cm[i * k + j]).sum())
        .collect();
    for (i, row) in out.iter_mut().enumerate() {
        let rs: f64 = cm[i * k..(i + 1) * k].iter().sum();
        for (j, v) in row.iter_mut().enumerate() {
            let den = match normalize {
                Normalize::None => 1.0,
                Normalize::True => rs,
                Normalize::Pred => cols[j],
                Normalize::All => total,
            };
            let x = *v / den;
            *v = if x.is_finite() { x } else { 0.0 };
        }
    }
    Ok((out, ctx.labels))
}

// ---- probability metrics --------------------------------------------------------------------------

/// Cumulative weighted false/true positives at each distinct score threshold (descending) and the thresholds.
pub(crate) fn binary_curve(y: &[f64], score: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let order = evalsuite_core::curve::descending_order(score);
    let (mut fps, mut tps, mut thr) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tp, mut fp) = (0.0f64, 0.0f64);
    for (pos, &i) in order.iter().enumerate() {
        tp += y[i] * w[i];
        fp += (1.0 - y[i]) * w[i];
        if pos + 1 == order.len() || score[i] - score[order[pos + 1]] != 0.0 {
            tps.push(tp);
            fps.push(fp);
            thr.push(score[i]);
        }
    }
    (fps, tps, thr)
}

pub(crate) fn auc_or_err(y: &[f64], s: &[f64], w: &[f64]) -> Result<f64> {
    let (auc, _, tp, fp) = evalsuite_core::curve::curve_scores(y, s, w);
    if tp == 0.0 || fp == 0.0 {
        let missing = if tp == 0.0 { "positive" } else { "negative" };
        return undefined(format!(
            "ROC AUC is undefined because y_true contains no {missing} observations. It needs at least one positive \
             and one negative example."
        ));
    }
    Ok(auc)
}

pub(crate) fn ap_or_err(y: &[f64], s: &[f64], w: &[f64]) -> Result<f64> {
    let (_, ap, tp, _) = evalsuite_core::curve::curve_scores(y, s, w);
    if tp == 0.0 {
        return undefined(
            "Average precision is undefined because y_true contains no positive observations.",
        );
    }
    Ok(ap)
}

pub(crate) fn check_probs(p: &[f64], n: usize, name: &str) -> Result<()> {
    if p.len() != n {
        return invalid(format!(
            "{name} must contain one row per observation. Received {} for {n} observations.",
            p.len()
        ));
    }
    if p.iter().any(|v| !v.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite values."));
    }
    if p.iter().any(|&v| !(0.0..=1.0).contains(&v)) {
        return invalid(format!(
            "{name} must contain probabilities in [0, 1]. If these are scores or logits, convert them to probabilities first."
        ));
    }
    Ok(())
}

pub(crate) fn check_rows(p: &[Vec<f64>], n: usize, k: usize, sum_to_one: bool) -> Result<()> {
    if p.len() != n {
        return invalid(format!(
            "y_prob must contain one row per observation. Received {} for {n} observations.",
            p.len()
        ));
    }
    for row in p {
        if row.len() != k {
            return invalid(format!(
                "y_prob has {} columns but there are {k} classes. Provide one probability column per class, in label order.",
                row.len()
            ));
        }
        check_probs(row, k, "y_prob")?;
        if sum_to_one && (row.iter().sum::<f64>() - 1.0).abs() > 1e-6 + 1e-5 {
            return invalid("Each row of y_prob must sum to 1 for multiclass probabilities.");
        }
    }
    Ok(())
}

/// Metrics on `P(positive)` scores of a binary task.
pub mod binary {
    use super::*;

    fn target<L: Label>(
        y_true: &[L],
        pos_label: Option<&L>,
        w: Option<&[f64]>,
        p: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let ctx = Ctx::new(y_true, None, None, w)?;
        if ctx.kind != Kind::Binary {
            return unsupported("This metric needs a binary target; use the multiclass version.");
        }
        check_probs(p, ctx.n(), "y_prob")?;
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

    /// ROC AUC of binary scores.
    pub fn roc_auc<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<f64> {
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        auc_or_err(&y, y_prob, &w)
    }

    /// Average precision (step-wise PR AUC) of binary scores.
    pub fn average_precision<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<f64> {
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        ap_or_err(&y, y_prob, &w)
    }

    /// ROC curve: (false positive rates, true positive rates, thresholds), starting at (0, 0) with threshold +∞.
    pub fn roc_curve<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        let (fps, tps, thr) = binary_curve(&y, y_prob, &w);
        let (tpl, fpl) = (*tps.last().unwrap(), *fps.last().unwrap());
        if tpl == 0.0 || fpl == 0.0 {
            let missing = if tpl == 0.0 { "positive" } else { "negative" };
            return undefined(format!(
                "The ROC curve is undefined because y_true contains no {missing} observations."
            ));
        }
        let mut fpr = vec![0.0];
        fpr.extend(fps.iter().map(|f| f / fpl));
        let mut tpr = vec![0.0];
        tpr.extend(tps.iter().map(|t| t / tpl));
        let mut th = vec![f64::INFINITY];
        th.extend(thr);
        Ok((fpr, tpr, th))
    }

    /// Precision–recall curve: (precision, recall, thresholds) by increasing threshold, ending at (1, 0).
    pub fn pr_curve<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        let (fps, tps, thr) = binary_curve(&y, y_prob, &w);
        let tpl = *tps.last().unwrap();
        if tpl == 0.0 {
            return undefined(
                "The PR curve is undefined because y_true contains no positive observations.",
            );
        }
        let mut prec: Vec<f64> = tps
            .iter()
            .zip(&fps)
            .map(|(t, f)| t / (t + f))
            .rev()
            .collect();
        let mut rec: Vec<f64> = tps.iter().map(|t| t / tpl).rev().collect();
        prec.push(1.0);
        rec.push(0.0);
        Ok((prec, rec, thr.into_iter().rev().collect()))
    }

    /// Log loss (cross-entropy) of binary probabilities, clipped to [ε, 1 − ε].
    pub fn log_loss<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<f64> {
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        let eps = f64::EPSILON;
        let losses: Vec<f64> = y
            .iter()
            .zip(y_prob)
            .map(|(&t, &p)| {
                let p = p.clamp(eps, 1.0 - eps);
                -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            })
            .collect();
        Ok(weighted_mean(&losses, &w))
    }

    /// Brier score of binary probabilities.
    pub fn brier_score<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<f64> {
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        let err: Vec<f64> = y
            .iter()
            .zip(y_prob)
            .map(|(t, p)| (p - t) * (p - t))
            .collect();
        Ok(weighted_mean(&err, &w))
    }

    /// Binning strategy for calibration.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Strategy {
        Uniform,
        Quantile,
    }

    /// Reliability diagram: (observed positive rate, mean predicted probability, bin weight) per non-empty bin.
    pub fn calibration_curve<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        n_bins: usize,
        strategy: Strategy,
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        if n_bins < 1 {
            return invalid("n_bins must be a positive integer.");
        }
        let (y, w) = target(y_true, pos_label, sample_weight, y_prob)?;
        let edges: Vec<f64> = match strategy {
            Strategy::Uniform => crate::util::linspace(0.0, 1.0, n_bins + 1),
            Strategy::Quantile => {
                let s = crate::util::sorted(y_prob);
                crate::util::linspace(0.0, 1.0, n_bins + 1)
                    .iter()
                    .map(|&q| crate::util::quantile(&s, q))
                    .collect()
            }
        };
        let [bw, bt, bp] =
            evalsuite_core::groups::calibration_bins(&y, y_prob, &w, &edges[1..edges.len() - 1]);
        let keep: Vec<usize> = (0..bw.len()).filter(|&b| bw[b] > 0.0).collect();
        Ok((
            keep.iter().map(|&b| bt[b] / bw[b]).collect(),
            keep.iter().map(|&b| bp[b] / bw[b]).collect(),
            keep.iter().map(|&b| bw[b]).collect(),
        ))
    }

    /// Expected calibration error.
    pub fn expected_calibration_error<L: Label>(
        y_true: &[L],
        y_prob: &[f64],
        n_bins: usize,
        strategy: Strategy,
        pos_label: Option<&L>,
        sample_weight: Option<&[f64]>,
    ) -> Result<f64> {
        let (t, p, w) =
            calibration_curve(y_true, y_prob, n_bins, strategy, pos_label, sample_weight)?;
        let terms: Vec<f64> = (0..t.len()).map(|i| w[i] * (t[i] - p[i]).abs()).collect();
        Ok(np_sum(&terms) / np_sum(&w))
    }
}

/// Multiclass ROC AUC strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MultiClass {
    /// One-vs-rest (macro, weighted or per class).
    #[default]
    Ovr,
    /// One-vs-one (Hand & Till), macro only.
    Ovo,
}

fn multiclass_ctx<L: Label>(
    y_true: &[L],
    p: &[Vec<f64>],
    labels: Option<&[L]>,
    w: Option<&[f64]>,
) -> Result<Ctx<L>> {
    let ctx = Ctx::new(y_true, None, labels, w)?;
    check_rows(p, ctx.n(), ctx.labels.len(), ctx.kind == Kind::Multiclass)?;
    Ok(ctx)
}

fn one_vs_rest<L: Label>(
    ctx: &Ctx<L>,
    p: &[Vec<f64>],
    avg: Average,
    f: fn(&[f64], &[f64], &[f64]) -> Result<f64>,
) -> Result<Score> {
    let k = ctx.labels.len();
    let mut scores = Vec::with_capacity(k);
    let mut support = Vec::with_capacity(k);
    for j in 0..k {
        let y: Vec<f64> = ctx
            .true_idx
            .iter()
            .map(|&t| f64::from(u8::from(t == j)))
            .collect();
        let s: Vec<f64> = p.iter().map(|r| r[j]).collect();
        scores.push(f(&y, &s, &ctx.w)?);
        support.push(y.iter().zip(&ctx.w).map(|(a, b)| a * b).sum::<f64>());
    }
    Ok(match avg {
        Average::None => Score::PerClass(scores),
        Average::Macro => Score::Scalar(np_sum(&scores) / k as f64),
        Average::Weighted => Score::Scalar(weighted_mean(&scores, &support)),
        _ => return unsupported("Use Macro, Weighted or None for multiclass probability metrics."),
    })
}

/// Multiclass ROC AUC from one probability row per observation (columns in label order).
pub fn roc_auc<L: Label>(
    y_true: &[L],
    y_prob: &[Vec<f64>],
    average: Average,
    multi_class: MultiClass,
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
) -> Result<Score> {
    let ctx = multiclass_ctx(y_true, y_prob, labels, sample_weight)?;
    let avg = ctx.resolve_average(average)?;
    if multi_class == MultiClass::Ovo {
        if avg != Average::Macro {
            return unsupported("One-vs-one ROC AUC supports average=Macro only.");
        }
        let k = ctx.labels.len();
        let mut pairs = Vec::new();
        for a in 0..k {
            for b in a + 1..k {
                let idx: Vec<usize> = (0..ctx.n())
                    .filter(|&i| ctx.true_idx[i] == a || ctx.true_idx[i] == b)
                    .collect();
                let ya: Vec<f64> = idx
                    .iter()
                    .map(|&i| f64::from(u8::from(ctx.true_idx[i] == a)))
                    .collect();
                let yb: Vec<f64> = ya.iter().map(|v| 1.0 - v).collect();
                let w: Vec<f64> = idx.iter().map(|&i| ctx.w[i]).collect();
                let pa: Vec<f64> = idx.iter().map(|&i| y_prob[i][a]).collect();
                let pb: Vec<f64> = idx.iter().map(|&i| y_prob[i][b]).collect();
                pairs.push((auc_or_err(&ya, &pa, &w)? + auc_or_err(&yb, &pb, &w)?) / 2.0);
            }
        }
        return Ok(Score::Scalar(np_sum(&pairs) / pairs.len() as f64));
    }
    if avg == Average::Micro {
        return unsupported(
            "average=Micro is not defined for multiclass ROC AUC; use Macro, Weighted or None.",
        );
    }
    if ctx.kind == Kind::Binary {
        return unsupported(
            "Use classification::binary::roc_auc with P(positive) for binary targets.",
        );
    }
    one_vs_rest(&ctx, y_prob, avg, auc_or_err)
}

/// Multiclass average precision (one-vs-rest).
pub fn average_precision<L: Label>(
    y_true: &[L],
    y_prob: &[Vec<f64>],
    average: Average,
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
) -> Result<Score> {
    let ctx = multiclass_ctx(y_true, y_prob, labels, sample_weight)?;
    let avg = ctx.resolve_average(average)?;
    if avg == Average::Micro {
        let k = ctx.labels.len();
        let (mut y, mut s, mut w) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..ctx.n() {
            for j in 0..k {
                y.push(f64::from(u8::from(ctx.true_idx[i] == j)));
                s.push(y_prob[i][j]);
                w.push(ctx.w[i]);
            }
        }
        return Ok(Score::Scalar(ap_or_err(&y, &s, &w)?));
    }
    one_vs_rest(&ctx, y_prob, avg, ap_or_err)
}

/// Multiclass log loss (probabilities clipped to [ε, 1 − ε]).
pub fn log_loss<L: Label>(
    y_true: &[L],
    y_prob: &[Vec<f64>],
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    let ctx = multiclass_ctx(y_true, y_prob, labels, sample_weight)?;
    let eps = f64::EPSILON;
    let losses: Vec<f64> = ctx
        .true_idx
        .iter()
        .zip(y_prob)
        .map(|(&t, r)| -r[t].clamp(eps, 1.0 - eps).ln())
        .collect();
    Ok(weighted_mean(&losses, &ctx.w))
}

/// Multiclass Brier score (squared error against the one-hot truth, summed over classes).
pub fn brier_score<L: Label>(
    y_true: &[L],
    y_prob: &[Vec<f64>],
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    let ctx = multiclass_ctx(y_true, y_prob, labels, sample_weight)?;
    let err: Vec<f64> = ctx
        .true_idx
        .iter()
        .zip(y_prob)
        .map(|(&t, r)| {
            r.iter()
                .enumerate()
                .map(|(j, &p)| (p - f64::from(u8::from(j == t))).powi(2))
                .sum()
        })
        .collect();
    Ok(weighted_mean(&err, &ctx.w))
}

/// Top-k accuracy: the true class's probability is among the `k` largest (strictly fewer than `k` higher).
pub fn top_k_accuracy<L: Label>(
    y_true: &[L],
    y_prob: &[Vec<f64>],
    k: usize,
    labels: Option<&[L]>,
    sample_weight: Option<&[f64]>,
) -> Result<f64> {
    let ctx = multiclass_ctx(y_true, y_prob, labels, sample_weight)?;
    if ctx.kind != Kind::Multiclass {
        return unsupported(
            "top_k_accuracy needs a multiclass target with one probability column per class.",
        );
    }
    let n_classes = ctx.labels.len();
    if !(1..=n_classes).contains(&k) {
        return invalid(format!(
            "k must be an integer between 1 and the number of classes ({n_classes})."
        ));
    }
    let hit: Vec<f64> = ctx
        .true_idx
        .iter()
        .zip(y_prob)
        .map(|(&t, r)| f64::from(u8::from(r.iter().filter(|&&p| p > r[t]).count() < k)))
        .collect();
    Ok(weighted_mean(&hit, &ctx.w))
}

/// Multilabel targets: indicator matrices with one `bool` column per label.
pub mod multilabel {
    use super::*;

    fn check(t: &[Vec<bool>], p: &[Vec<bool>]) -> Result<usize> {
        if t.is_empty() {
            return invalid("y_true is empty. Provide at least one example.");
        }
        crate::util::same_len(t.len(), p.len(), "y_true and y_pred")?;
        let k = t[0].len();
        if t.iter().chain(p).any(|r| r.len() != k) {
            return invalid(
                "Every row of a multilabel indicator matrix must have one column per label.",
            );
        }
        Ok(k)
    }

    fn counts(t: &[Vec<bool>], p: &[Vec<bool>], w: &[f64], k: usize) -> Counts {
        let mut c = Counts {
            tp: vec![0.0; k],
            fp: vec![0.0; k],
            fn_: vec![0.0; k],
            tn: vec![0.0; k],
            support: vec![0.0; k],
        };
        for j in 0..k {
            let (mut tp, mut fp, mut fn_, mut tn) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for i in 0..t.len() {
                let (a, b) = (t[i][j], p[i][j]);
                tp.push(f64::from(u8::from(a && b)) * w[i]);
                fp.push(f64::from(u8::from(!a && b)) * w[i]);
                fn_.push(f64::from(u8::from(a && !b)) * w[i]);
                tn.push(f64::from(u8::from(!a && !b)) * w[i]);
            }
            c.tp[j] = np_sum(&tp);
            c.fp[j] = np_sum(&fp);
            c.fn_[j] = np_sum(&fn_);
            c.tn[j] = np_sum(&tn);
            c.support[j] = c.tp[j] + c.fn_[j];
        }
        c
    }

    fn averaged(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
        f: &CountFn,
    ) -> Result<Score> {
        let k = check(t, p)?;
        let w = weights(w, t.len())?;
        let avg = resolve_average(Kind::Multilabel, average, k)?;
        if avg == Average::Samples {
            let per: Vec<f64> = t
                .iter()
                .zip(p)
                .map(|(a, b)| {
                    let (mut tp, mut fp, mut fn_, mut tn) = (0.0, 0.0, 0.0, 0.0);
                    for (&x, &y) in a.iter().zip(b) {
                        match (x, y) {
                            (true, true) => tp += 1.0,
                            (false, true) => fp += 1.0,
                            (true, false) => fn_ += 1.0,
                            (false, false) => tn += 1.0,
                        }
                    }
                    f(tp, fp, fn_, tn, zd)
                })
                .collect();
            return Ok(Score::Scalar(weighted_mean(&per, &w)));
        }
        Ok(combine(avg, &counts(t, p, &w, k), f, zd, None, np_sum(&w)))
    }

    pub fn precision(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_precision)
    }
    pub fn recall(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_recall)
    }
    pub fn f1(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_fbeta(1.0))
    }
    pub fn fbeta(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        beta: f64,
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_fbeta(beta))
    }
    pub fn jaccard(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_jaccard)
    }
    pub fn specificity(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_specificity)
    }
    pub fn npv(
        t: &[Vec<bool>],
        p: &[Vec<bool>],
        average: Average,
        w: Option<&[f64]>,
        zd: ZeroDivision,
    ) -> Result<Score> {
        averaged(t, p, average, w, zd, &f_npv)
    }

    /// Exact-match accuracy (every label right).
    pub fn accuracy(t: &[Vec<bool>], p: &[Vec<bool>], w: Option<&[f64]>) -> Result<f64> {
        check(t, p)?;
        let w = weights(w, t.len())?;
        let c: Vec<f64> = t
            .iter()
            .zip(p)
            .map(|(a, b)| f64::from(u8::from(a == b)))
            .collect();
        Ok(weighted_mean(&c, &w))
    }

    /// Hamming loss: share of wrong label assignments.
    pub fn hamming_loss(t: &[Vec<bool>], p: &[Vec<bool>], w: Option<&[f64]>) -> Result<f64> {
        let k = check(t, p)?;
        let w = weights(w, t.len())?;
        let wrong: Vec<f64> = t
            .iter()
            .zip(p)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x != y).count() as f64 / k as f64)
            .collect();
        Ok(weighted_mean(&wrong, &w))
    }

    fn per_label(
        t: &[Vec<bool>],
        prob: &[Vec<f64>],
        average: Average,
        w: Option<&[f64]>,
        f: fn(&[f64], &[f64], &[f64]) -> Result<f64>,
    ) -> Result<Score> {
        if t.is_empty() {
            return invalid("y_true is empty. Provide at least one example.");
        }
        let k = t[0].len();
        check_rows(prob, t.len(), k, false)?;
        let w = weights(w, t.len())?;
        let avg = resolve_average(Kind::Multilabel, average, k)?;
        let y: Vec<Vec<f64>> = (0..k)
            .map(|j| t.iter().map(|r| f64::from(u8::from(r[j]))).collect())
            .collect();
        if avg == Average::Micro {
            let (mut yy, mut ss, mut ww) = (Vec::new(), Vec::new(), Vec::new());
            for i in 0..t.len() {
                for j in 0..k {
                    yy.push(y[j][i]);
                    ss.push(prob[i][j]);
                    ww.push(w[i]);
                }
            }
            return Ok(Score::Scalar(f(&yy, &ss, &ww)?));
        }
        let mut scores = Vec::with_capacity(k);
        let mut support = Vec::with_capacity(k);
        for j in 0..k {
            let s: Vec<f64> = prob.iter().map(|r| r[j]).collect();
            scores.push(f(&y[j], &s, &w)?);
            support.push(y[j].iter().zip(&w).map(|(a, b)| a * b).sum::<f64>());
        }
        Ok(match avg {
            Average::None => Score::PerClass(scores),
            Average::Macro => Score::Scalar(np_sum(&scores) / k as f64),
            Average::Weighted => Score::Scalar(weighted_mean(&scores, &support)),
            _ => return unsupported("average=Samples is not defined for this metric."),
        })
    }

    /// Per-label ROC AUC of independent label probabilities.
    pub fn roc_auc(
        t: &[Vec<bool>],
        prob: &[Vec<f64>],
        average: Average,
        w: Option<&[f64]>,
    ) -> Result<Score> {
        per_label(t, prob, average, w, auc_or_err)
    }

    /// Per-label average precision.
    pub fn average_precision(
        t: &[Vec<bool>],
        prob: &[Vec<f64>],
        average: Average,
        w: Option<&[f64]>,
    ) -> Result<Score> {
        per_label(t, prob, average, w, ap_or_err)
    }
}
