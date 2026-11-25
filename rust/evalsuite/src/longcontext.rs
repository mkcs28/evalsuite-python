//! Long-context and summarisation metrics: retrieval accuracy by context length (effective length),
//! needle-in-a-haystack grids, position effects (lost in the middle), context utilisation, summary coverage,
//! compression ratio, citation coverage and cross-document consistency, as EvalSuite's Python
//! `evalsuite.llmsys.longcontext`.
//!
//! ```
//! use evalsuite::longcontext::lost_in_the_middle;
//! let r = lost_in_the_middle(&[1.0, 0.0, 1.0], &[0.0, 0.5, 1.0], 0.2).unwrap();
//! assert_eq!(r.value, 1.0);
//! ```

use std::collections::HashSet;

use crate::error::{invalid, Result};
use crate::generation::py_split;
use crate::summary::{floats, nonempty, rate, same, Rate};
use crate::util::{linspace, mean, np_sum, percentile, sorted};

fn uniq(x: &[f64]) -> Vec<f64> {
    let mut v = sorted(x);
    v.dedup();
    v
}

fn mean_where(c: &[f64], m: impl Fn(usize) -> bool) -> Option<f64> {
    let v: Vec<f64> = (0..c.len()).filter(|&i| m(i)).map(|i| c[i]).collect();
    (!v.is_empty()).then(|| mean(&v))
}

fn as_f(c: &[bool]) -> Vec<f64> {
    c.iter().map(|&b| f64::from(u8::from(b))).collect()
}

/// Accuracy by context length and the effective length (the longest length up to which every length passes
/// the threshold).
#[derive(Debug, Clone, PartialEq)]
pub struct ByLength {
    pub value: f64,
    /// (length, accuracy), by increasing length.
    pub by_length: Vec<(f64, f64)>,
    pub effective_length: Option<f64>,
    pub n_passing_lengths: usize,
}

/// Retrieval accuracy overall and per context length.
pub fn retrieval_accuracy_by_length(correct: &[bool], context_lengths: &[f64], threshold: f64) -> Result<ByLength> {
    nonempty(correct, "correct")?;
    floats(context_lengths, "context_lengths", Some(0.0), None)?;
    same(correct.len(), context_lengths.len(), "correct", "context_lengths")?;
    let c = as_f(correct);
    let by: Vec<(f64, f64)> = uniq(context_lengths).into_iter().map(|v| (v, mean_where(&c, |i| context_lengths[i] == v).unwrap())).collect();
    let mut effective = None;
    for &(v, a) in &by {
        if a >= threshold {
            effective = Some(v);
        } else {
            break;
        }
    }
    Ok(ByLength { value: mean(&c), n_passing_lengths: by.iter().filter(|(_, a)| *a >= threshold).count(), by_length: by, effective_length: effective })
}

/// Needle-in-a-haystack accuracy grid (lengths × depths; NaN for empty cells) and its worst cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Haystack {
    pub value: f64,
    pub lengths: Vec<f64>,
    pub depths: Vec<f64>,
    pub grid: Vec<Vec<f64>>,
    pub worst_cell: f64,
}

/// Needle retrieval accuracy (`correct` in [0, 1]) by context length and needle depth (in [0, 1]).
pub fn needle_in_haystack(correct: &[f64], context_lengths: &[f64], depths: &[f64]) -> Result<Haystack> {
    floats(correct, "correct", Some(0.0), Some(1.0))?;
    floats(context_lengths, "context_lengths", Some(0.0), None)?;
    floats(depths, "depths", Some(0.0), Some(1.0))?;
    same(correct.len(), context_lengths.len(), "correct", "context_lengths")?;
    same(correct.len(), depths.len(), "correct", "depths")?;
    let (ls, ds) = (uniq(context_lengths), uniq(depths));
    let grid: Vec<Vec<f64>> = ls
        .iter()
        .map(|&l| ds.iter().map(|&d| mean_where(correct, |i| context_lengths[i] == l && depths[i] == d).unwrap_or(f64::NAN)).collect())
        .collect();
    let worst = grid.iter().flatten().copied().filter(|x| !x.is_nan()).fold(f64::INFINITY, f64::min);
    Ok(Haystack { value: mean(correct), lengths: ls, depths: ds, grid, worst_cell: worst })
}

fn check_positions(correct: &[f64], positions: &[f64]) -> Result<()> {
    floats(correct, "correct", Some(0.0), Some(1.0))?;
    floats(positions, "positions", Some(0.0), Some(1.0))?;
    same(correct.len(), positions.len(), "correct", "positions")
}

/// Position sensitivity: the gap between the best and worst accuracy over `n_bins` equal position bins;
/// returns (gap, per-bin accuracies of the non-empty bins as (low edge, high edge, accuracy)).
pub fn position_accuracy(correct: &[f64], positions: &[f64], n_bins: usize) -> Result<(f64, Vec<(f64, f64, f64)>)> {
    check_positions(correct, positions)?;
    if n_bins < 2 {
        return invalid("n_bins must be an integer >= 2.");
    }
    let e = linspace(0.0, 1.0, n_bins + 1);
    let last = e[n_bins];
    let mut by = Vec::new();
    for k in 0..n_bins {
        let (lo, hi) = (e[k], e[k + 1]);
        let a = mean_where(correct, |i| positions[i] >= lo && if hi != last { positions[i] < hi } else { positions[i] <= hi });
        if let Some(a) = a {
            by.push((lo, hi, a));
        }
    }
    let mx = by.iter().map(|b| b.2).fold(f64::NEG_INFINITY, f64::max);
    let mn = by.iter().map(|b| b.2).fold(f64::INFINITY, f64::min);
    Ok((mx - mn, by))
}

/// Lost-in-the-middle effect: mean edge accuracy − middle accuracy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LostInMiddle {
    pub value: f64,
    pub start_accuracy: f64,
    pub middle_accuracy: f64,
    pub end_accuracy: f64,
}

/// (start + end)/2 − middle accuracy, with positions ≤ `edge` at the start and ≥ 1 − `edge` at the end.
pub fn lost_in_the_middle(correct: &[f64], positions: &[f64], edge: f64) -> Result<LostInMiddle> {
    check_positions(correct, positions)?;
    if !(edge > 0.0 && edge < 0.5) {
        return invalid("edge must be in (0, 0.5).");
    }
    let s = mean_where(correct, |i| positions[i] <= edge);
    let e = mean_where(correct, |i| positions[i] >= 1.0 - edge);
    let m = mean_where(correct, |i| positions[i] > edge && positions[i] < 1.0 - edge);
    let (Some(s), Some(e), Some(m)) = (s, e, m) else {
        return invalid("Need examples at the start, the middle and the end of the context.");
    };
    Ok(LostInMiddle { value: (s + e) / 2.0 - m, start_accuracy: s, middle_accuracy: m, end_accuracy: e })
}

/// Share of provided facts the output used (pooled over examples with facts) and the mean per example.
pub fn context_utilization<S: AsRef<str>>(used_facts: &[Vec<S>], provided_facts: &[Vec<S>]) -> Result<(f64, f64)> {
    nonempty(used_facts, "used_facts")?;
    nonempty(provided_facts, "provided_facts")?;
    same(used_facts.len(), provided_facts.len(), "used_facts", "provided_facts")?;
    let (mut hit, mut tot, mut per) = (0usize, 0usize, Vec::new());
    for (u, p) in used_facts.iter().zip(provided_facts) {
        let ps: HashSet<&str> = p.iter().map(AsRef::as_ref).collect();
        let us: HashSet<&str> = u.iter().map(AsRef::as_ref).collect();
        if ps.is_empty() {
            continue;
        }
        let h = ps.intersection(&us).count();
        hit += h;
        tot += ps.len();
        per.push(h as f64 / ps.len() as f64);
    }
    if tot == 0 {
        return invalid("No example provides any fact.");
    }
    Ok((hit as f64 / tot as f64, mean(&per)))
}

/// Summary coverage: the mean (weighted) share of key points covered, with the factual consistency of the
/// summaries' claims when given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coverage {
    pub value: f64,
    /// Share of supported claims, pooled (NaN without claims).
    pub factual_consistency: Option<f64>,
    /// Share of summaries whose claims are all supported.
    pub fully_consistent_rate: Option<f64>,
}

/// Key-point coverage of summaries; `weights` are optional pyramid weights per key point.
pub fn summary_coverage(key_points_covered: &[Vec<bool>], weights: Option<&[Vec<f64>]>, claims_supported: Option<&[Vec<bool>]>) -> Result<Coverage> {
    nonempty(key_points_covered, "key_points_covered")?;
    if let Some(w) = weights {
        same(key_points_covered.len(), w.len(), "key_points_covered", "weights")?;
    }
    let mut per = Vec::new();
    for (i, r) in key_points_covered.iter().enumerate() {
        let w = weights.map_or_else(|| vec![1.0; r.len()], |w| w[i].clone());
        if r.len() != w.len() || r.is_empty() {
            return invalid(format!("Summary {i}: key points and weights must be non-empty and the same length."));
        }
        floats(&w, &format!("weights[{i}]"), Some(0.0), None)?;
        let sw = np_sum(&w);
        per.push(if sw != 0.0 { w.iter().zip(r).map(|(w, &c)| if c { *w } else { 0.0 }).sum::<f64>() / sw } else { 0.0 });
    }
    let (mut fc, mut full) = (None, None);
    if let Some(cl) = claims_supported {
        same(key_points_covered.len(), cl.len(), "key_points_covered", "claims_supported")?;
        let flat: Vec<bool> = cl.iter().flatten().copied().collect();
        fc = Some(if flat.is_empty() { f64::NAN } else { mean(&as_f(&flat)) });
        full = Some(mean(&cl.iter().map(|r| f64::from(u8::from(r.iter().all(|&x| x)))).collect::<Vec<_>>()));
    }
    Ok(Coverage { value: mean(&per), factual_consistency: fc, fully_consistent_rate: full })
}

/// Mean (and median) ratio of source to summary length in tokens (whitespace tokens by default).
pub fn compression_ratio<S: AsRef<str>>(sources: &[S], summaries: &[S], tokenizer: Option<&dyn Fn(&str) -> usize>) -> Result<(f64, f64)> {
    nonempty(sources, "sources")?;
    same(sources.len(), summaries.len(), "sources", "summaries")?;
    let count = |s: &str| tokenizer.map_or_else(|| py_split(s).len(), |f| f(s));
    let mut r = Vec::new();
    for (i, (a, b)) in sources.iter().zip(summaries).enumerate() {
        let nb = count(b.as_ref());
        if nb == 0 {
            return invalid(format!("summaries[{i}] is empty."));
        }
        r.push(count(a.as_ref()) as f64 / nb as f64);
    }
    Ok((mean(&r), percentile(&sorted(&r), 50.0)))
}

/// Share of statements with at least one citation (`citations_per_statement[i][j]` = citations of statement
/// j of output i), with the mean per output.
pub fn citation_coverage(citations_per_statement: &[Vec<f64>]) -> Result<(Rate, f64)> {
    nonempty(citations_per_statement, "citations_per_statement")?;
    let flat: Vec<bool> = citations_per_statement.iter().flatten().map(|&v| v > 0.0).collect();
    if flat.is_empty() {
        return invalid("No statements given.");
    }
    let per: Vec<f64> = citations_per_statement.iter().filter(|r| !r.is_empty()).map(|r| r.iter().filter(|&&v| v > 0.0).count() as f64 / r.len() as f64).collect();
    Ok((rate(&flat), mean(&per)))
}

/// Cross-document consistency: share of questions whose answers (one per document set or variant) all agree,
/// and the mean pairwise agreement. Strings are compared lowercased with whitespace collapsed unless a
/// `normalize` function is given.
pub fn cross_document_consistency<S: AsRef<str>>(answers: &[Vec<S>], normalize: Option<&dyn Fn(&str) -> String>) -> Result<(f64, f64)> {
    nonempty(answers, "answers")?;
    let (mut full, mut pair) = (Vec::new(), Vec::new());
    for (i, r) in answers.iter().enumerate() {
        if r.len() < 2 {
            return invalid(format!("answers[{i}] needs at least two answers."));
        }
        let v: Vec<String> = r.iter().map(|a| normalize.map_or_else(|| py_split(&a.as_ref().to_lowercase()).join(" "), |f| f(a.as_ref()))).collect();
        full.push(f64::from(u8::from(v.iter().all(|x| *x == v[0]))));
        let (mut eq, mut n) = (0usize, 0usize);
        for a in 0..v.len() {
            for b in a + 1..v.len() {
                eq += usize::from(v[a] == v[b]);
                n += 1;
            }
        }
        pair.push(eq as f64 / n as f64);
    }
    Ok((mean(&full), mean(&pair)))
}
