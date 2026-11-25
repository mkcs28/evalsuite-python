//! Monitoring: sliding-window metrics for live traffic, score drift against a reference (population
//! stability index, Kolmogorov–Smirnov) and threshold alerts, as EvalSuite's Python `evalsuite.monitor`.
//!
//! ```
//! use evalsuite::monitor::{Monitor, Op};
//! let mut m = Monitor::new(500).alert("accuracy", Op::Lt, 0.8);
//! let alerts = m.update(&[1, 0, 1, 1], Some(&[1, 0, 0, 1]), None).unwrap();
//! assert_eq!(alerts[0].name, "accuracy");
//! ```

use std::collections::VecDeque;

use crate::classification::{self as cls, Average, Opts};
use crate::error::{invalid, Result};
use crate::util::{quantile, sorted};

/// Population stability index of `current` against `reference`, with bins at the reference's quantiles
/// (`bins` + 1 linearly interpolated quantiles, duplicates merged) and bin shares floored at `eps`:
/// Σ (c − r)·ln(c / r). Below 0.1 is usually read as stable, above 0.25 as a major shift.
pub fn population_stability_index(
    reference: &[f64],
    current: &[f64],
    bins: usize,
    eps: f64,
) -> Result<f64> {
    if reference.len() < 2
        || current.is_empty()
        || reference.iter().chain(current).any(|v| !v.is_finite())
    {
        return invalid("PSI needs finite reference (>= 2) and current (>= 1) values.");
    }
    if bins < 2 {
        return invalid("bins must be at least 2.");
    }
    let r = sorted(reference);
    let mut edges: Vec<f64> = crate::util::linspace(0.0, 1.0, bins + 1)
        .iter()
        .map(|&q| quantile(&r, q))
        .collect();
    edges.dedup();
    if edges.len() < 2 {
        edges = vec![r[0] - 0.5, r[0] + 0.5];
    }
    let inner = &edges[1..edges.len() - 1];
    let k = edges.len() - 1;
    let share = |v: &[f64]| -> Vec<f64> {
        let mut c = vec![0.0; k];
        for x in v {
            // `searchsorted(inner, x, side="right")`
            c[inner.partition_point(|e| e <= x)] += 1.0;
        }
        c.iter().map(|n| (n / v.len() as f64).max(eps)).collect()
    };
    let (rp, cp) = (share(reference), share(current));
    Ok(rp
        .iter()
        .zip(&cp)
        .map(|(r, c)| (c - r) * (c / r).ln())
        .sum())
}

/// Two-sample Kolmogorov–Smirnov statistic D and its asymptotic p-value (Kolmogorov distribution with the
/// effective sample size n·m / (n + m)); for small samples SciPy's exact p-value is more accurate.
pub fn ks_drift(reference: &[f64], current: &[f64]) -> Result<(f64, f64)> {
    if reference.is_empty()
        || current.is_empty()
        || reference.iter().chain(current).any(|v| v.is_nan())
    {
        return invalid("ks_drift needs non-empty samples without NaN.");
    }
    let (a, b) = (sorted(reference), sorted(current));
    let (n, m) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut d) = (0usize, 0usize, 0.0f64);
    while i < a.len() && j < b.len() {
        let x = a[i].min(b[j]);
        while i < a.len() && a[i] <= x {
            i += 1;
        }
        while j < b.len() && b[j] <= x {
            j += 1;
        }
        d = d.max((i as f64 / n - j as f64 / m).abs());
    }
    let en = (n * m / (n + m)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    // Q_KS(λ) = 2 Σ (−1)^{k−1} e^{−2k²λ²}
    let mut p = 0.0;
    if lambda < 1e-3 {
        p = 1.0;
    } else {
        for k in 1..=100 {
            let t =
                2.0 * (-1.0f64).powi(k - 1) * (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
            p += t;
            if t.abs() < 1e-12 {
                break;
            }
        }
    }
    Ok((d, p.clamp(0.0, 1.0)))
}

/// Comparison operator of an alert rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    fn holds(self, a: f64, b: f64) -> bool {
        match self {
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
        }
    }
}

/// A threshold crossed by a monitored quantity.
#[derive(Debug, Clone, PartialEq)]
pub struct Alert {
    pub name: String,
    pub value: f64,
    pub op: Op,
    pub threshold: f64,
    /// Observations in the window when it fired.
    pub n: usize,
}

/// Sliding-window metrics, drift and alerts for a deployed binary classifier (labels 0 / 1).
#[derive(Debug, Clone)]
pub struct Monitor {
    window: usize,
    truth: VecDeque<i64>,
    pred: VecDeque<i64>,
    prob: VecDeque<f64>,
    scores: VecDeque<f64>,
    reference: Option<Vec<f64>>,
    rules: Vec<(String, Op, f64)>,
    min_n: usize,
    /// Observations seen so far.
    pub n_seen: usize,
}

impl Monitor {
    /// A monitor keeping the last `window` labelled observations.
    pub fn new(window: usize) -> Monitor {
        Monitor {
            window: window.max(1),
            truth: VecDeque::new(),
            pred: VecDeque::new(),
            prob: VecDeque::new(),
            scores: VecDeque::new(),
            reference: None,
            rules: vec![],
            min_n: 1,
            n_seen: 0,
        }
    }

    /// Adds an alert rule on a metric (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `brier_score`) or a
    /// drift quantity (`psi`, `ks`, `ks_p`).
    pub fn alert(mut self, name: &str, op: Op, threshold: f64) -> Monitor {
        self.rules.push((name.to_owned(), op, threshold));
        self
    }

    /// Reference scores (e.g. validation-set probabilities) for drift detection.
    pub fn reference(mut self, scores: &[f64]) -> Monitor {
        self.reference = Some(scores.to_vec());
        self
    }

    /// No metric alert before this many labelled observations.
    pub fn min_n(mut self, n: usize) -> Monitor {
        self.min_n = n.max(1);
        self
    }

    fn push<T>(q: &mut VecDeque<T>, v: impl IntoIterator<Item = T>, cap: usize) {
        for x in v {
            if q.len() == cap {
                q.pop_front();
            }
            q.push_back(x);
        }
    }

    /// Records scores without labels (drift only); returns the drift alerts that fire.
    pub fn observe(&mut self, scores: &[f64]) -> Result<Vec<Alert>> {
        Self::push(&mut self.scores, scores.iter().copied(), self.window);
        let v = self.drift()?;
        Ok(self.fire(&v))
    }

    /// Adds labelled observations (with predicted labels and/or positive-class probabilities); returns the
    /// alerts that fire on the window.
    pub fn update(
        &mut self,
        y_true: &[i64],
        y_pred: Option<&[i64]>,
        y_prob: Option<&[f64]>,
    ) -> Result<Vec<Alert>> {
        if y_pred.is_none() && y_prob.is_none() {
            return invalid("update() needs y_pred, y_prob, or both.");
        }
        if y_pred.is_some_and(|p| p.len() != y_true.len())
            || y_prob.is_some_and(|p| p.len() != y_true.len())
        {
            return invalid("y_pred / y_prob must have one value per observation in y_true.");
        }
        if (y_pred.is_some() && !self.truth.is_empty() && self.pred.len() != self.truth.len())
            || (y_prob.is_some() && !self.truth.is_empty() && self.prob.len() != self.truth.len())
        {
            return invalid(
                "Every update() must pass the same inputs (y_pred and/or y_prob) as the first.",
            );
        }
        Self::push(&mut self.truth, y_true.iter().copied(), self.window);
        if let Some(p) = y_pred {
            Self::push(&mut self.pred, p.iter().copied(), self.window);
        }
        if let Some(p) = y_prob {
            Self::push(&mut self.prob, p.iter().copied(), self.window);
            Self::push(&mut self.scores, p.iter().copied(), self.window);
        }
        self.n_seen += y_true.len();
        let mut values = if self.truth.len() >= self.min_n {
            self.metrics()
        } else {
            vec![]
        };
        values.extend(self.drift()?);
        Ok(self.fire(&values))
    }

    /// Metrics on the current window (those defined for it).
    pub fn metrics(&self) -> Vec<(String, f64)> {
        let t: Vec<i64> = self.truth.iter().copied().collect();
        let mut out = vec![];
        if self.pred.len() == t.len() && !t.is_empty() {
            let p: Vec<i64> = self.pred.iter().copied().collect();
            let o = Opts {
                average: Average::Binary,
                pos_label: Some(1),
                ..Opts::default()
            };
            if let Ok(v) = cls::accuracy(&t, &p, None) {
                out.push(("accuracy".into(), v));
            }
            for (name, r) in [
                ("precision", cls::precision(&t, &p, &o)),
                ("recall", cls::recall(&t, &p, &o)),
                ("f1", cls::f1(&t, &p, &o)),
            ] {
                if let Some(v) = r.ok().and_then(|s| s.value()) {
                    out.push((name.into(), v));
                }
            }
        }
        if self.prob.len() == t.len() && !t.is_empty() {
            let q: Vec<f64> = self.prob.iter().copied().collect();
            if let Ok(v) = cls::binary::roc_auc(&t, &q, Some(&1), None) {
                out.push(("roc_auc".into(), v));
            }
            if let Ok(v) = cls::binary::brier_score(&t, &q, Some(&1), None) {
                out.push(("brier_score".into(), v));
            }
        }
        out
    }

    /// PSI and KS of the recent scores against the reference (empty without a reference).
    pub fn drift(&self) -> Result<Vec<(String, f64)>> {
        let Some(r) = &self.reference else {
            return Ok(vec![]);
        };
        if self.scores.len() < 2 {
            return Ok(vec![]);
        }
        let c: Vec<f64> = self.scores.iter().copied().collect();
        let psi = population_stability_index(r, &c, 10, 1e-4)?;
        let (d, p) = ks_drift(r, &c)?;
        Ok(vec![
            ("psi".into(), psi),
            ("ks".into(), d),
            ("ks_p".into(), p),
        ])
    }

    fn fire(&self, values: &[(String, f64)]) -> Vec<Alert> {
        let n = if self.truth.is_empty() {
            self.scores.len()
        } else {
            self.truth.len()
        };
        self.rules
            .iter()
            .filter_map(|(name, op, thr)| {
                let v = values.iter().find(|(k, _)| k == name)?.1;
                (!v.is_nan() && op.holds(v, *thr)).then(|| Alert {
                    name: name.clone(),
                    value: v,
                    op: *op,
                    threshold: *thr,
                    n,
                })
            })
            .collect()
    }
}
