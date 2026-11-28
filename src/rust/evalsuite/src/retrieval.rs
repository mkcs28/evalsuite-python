//! Ranked-retrieval metrics (precision/recall/hit rate at k, MRR, MAP, NDCG), as EvalSuite's Python
//! `evalsuite.text.retrieval` (trec_eval / ranx conventions).
//!
//! Per query, `relevant` holds the judged documents with their grades ([`Judgements`]; build one from a set
//! of ids with [`judgements`]) and `retrieved` the ranked ids, best first. Scores are averaged over queries.
//!
//! ```
//! use evalsuite::retrieval::{judgements, mrr};
//! let rel = vec![judgements(["a", "c"]), judgements(["x"])];
//! let run = vec![vec!["b", "a", "c"], vec!["x", "y"]];
//! assert_eq!(mrr(&rel, &run, None).unwrap().mean, 0.75);
//! ```

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::error::{invalid, Result};

/// Relevance judgements of one query: document id → grade (> 0; ungraded relevance is grade 1).
pub type Judgements<D> = HashMap<D, f64>;

/// Judgements from a set of relevant ids (each grade 1).
pub fn judgements<D: Hash + Eq>(ids: impl IntoIterator<Item = D>) -> Judgements<D> {
    ids.into_iter().map(|d| (d, 1.0)).collect()
}

/// Graded judgements; grades must be finite and ≥ 0, and grade-0 entries are dropped (not relevant).
pub fn graded<D: Hash + Eq>(pairs: impl IntoIterator<Item = (D, f64)>) -> Result<Judgements<D>> {
    let mut out = HashMap::new();
    for (d, g) in pairs {
        if !(g.is_finite() && g >= 0.0) {
            return invalid("relevance grades must be finite and >= 0.");
        }
        if g > 0.0 {
            out.insert(d, g);
        }
    }
    Ok(out)
}

/// Per-query scores and their mean (NaN-ignoring, as `np.nanmean`).
#[derive(Debug, Clone, PartialEq)]
pub struct QueryScores {
    pub mean: f64,
    pub per_query: Vec<f64>,
}

fn finish(per_query: Vec<f64>) -> QueryScores {
    let ok: Vec<f64> = per_query.iter().copied().filter(|v| !v.is_nan()).collect();
    let mean = if ok.is_empty() {
        f64::NAN
    } else {
        crate::util::np_sum(&ok) / ok.len() as f64
    };
    QueryScores { mean, per_query }
}

fn queries<D: Hash + Eq>(relevant: &[Judgements<D>], retrieved: &[Vec<D>]) -> Result<()> {
    if relevant.is_empty() {
        return invalid("relevant is empty. Provide at least one example.");
    }
    if relevant.len() != retrieved.len() {
        return invalid(format!(
            "relevant and retrieved must have one entry per query. Received {} and {}.",
            relevant.len(),
            retrieved.len()
        ));
    }
    for (i, r) in retrieved.iter().enumerate() {
        if r.iter().collect::<HashSet<_>>().len() != r.len() {
            return invalid(format!("retrieved[{i}] lists the same id more than once."));
        }
    }
    Ok(())
}

fn need_relevant<D>(relevant: &[Judgements<D>], what: &str) -> Result<()> {
    let empty: Vec<usize> = relevant
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_empty())
        .map(|(i, _)| i)
        .take(5)
        .collect();
    if !empty.is_empty() {
        return invalid(format!(
            "{what} is undefined for queries without relevant documents (queries {empty:?})."
        ));
    }
    Ok(())
}

fn top<D>(run: &[D], k: Option<usize>) -> &[D] {
    match k {
        Some(k) => &run[..k.min(run.len())],
        None => run,
    }
}

fn check_k(k: Option<usize>) -> Result<()> {
    if k == Some(0) {
        return invalid("k must be a positive integer.");
    }
    Ok(())
}

/// Precision@k: relevant documents among the top k, divided by k.
pub fn precision_at_k<D: Hash + Eq>(
    relevant: &[Judgements<D>],
    retrieved: &[Vec<D>],
    k: usize,
) -> Result<QueryScores> {
    check_k(Some(k))?;
    queries(relevant, retrieved)?;
    Ok(finish(
        relevant
            .iter()
            .zip(retrieved)
            .map(|(r, d)| {
                top(d, Some(k)).iter().filter(|x| r.contains_key(x)).count() as f64 / k as f64
            })
            .collect(),
    ))
}

/// Recall@k: share of a query's relevant documents found in the top k.
pub fn recall_at_k<D: Hash + Eq>(
    relevant: &[Judgements<D>],
    retrieved: &[Vec<D>],
    k: usize,
) -> Result<QueryScores> {
    check_k(Some(k))?;
    queries(relevant, retrieved)?;
    need_relevant(relevant, "Recall@k")?;
    Ok(finish(
        relevant
            .iter()
            .zip(retrieved)
            .map(|(r, d)| {
                top(d, Some(k)).iter().filter(|x| r.contains_key(x)).count() as f64 / r.len() as f64
            })
            .collect(),
    ))
}

/// Hit rate@k: whether any relevant document is in the top k.
pub fn hit_rate_at_k<D: Hash + Eq>(
    relevant: &[Judgements<D>],
    retrieved: &[Vec<D>],
    k: usize,
) -> Result<QueryScores> {
    check_k(Some(k))?;
    queries(relevant, retrieved)?;
    Ok(finish(
        relevant
            .iter()
            .zip(retrieved)
            .map(|(r, d)| f64::from(u8::from(top(d, Some(k)).iter().any(|x| r.contains_key(x)))))
            .collect(),
    ))
}

/// Mean reciprocal rank of the first relevant document (within the top k, if given).
pub fn mrr<D: Hash + Eq>(
    relevant: &[Judgements<D>],
    retrieved: &[Vec<D>],
    k: Option<usize>,
) -> Result<QueryScores> {
    check_k(k)?;
    queries(relevant, retrieved)?;
    Ok(finish(
        relevant
            .iter()
            .zip(retrieved)
            .map(|(r, d)| {
                top(d, k)
                    .iter()
                    .position(|x| r.contains_key(x))
                    .map_or(0.0, |i| 1.0 / (i + 1) as f64)
            })
            .collect(),
    ))
}

/// Mean average precision (within the top k, if given), normalised by the number of relevant documents.
pub fn mean_average_precision_at_k<D: Hash + Eq>(
    relevant: &[Judgements<D>],
    retrieved: &[Vec<D>],
    k: Option<usize>,
) -> Result<QueryScores> {
    check_k(k)?;
    queries(relevant, retrieved)?;
    need_relevant(relevant, "Average precision")?;
    Ok(finish(
        relevant
            .iter()
            .zip(retrieved)
            .map(|(r, d)| {
                let (mut hits, mut total) = (0u64, 0.0f64);
                for (i, x) in top(d, k).iter().enumerate() {
                    if r.contains_key(x) {
                        hits += 1;
                        total += hits as f64 / (i + 1) as f64;
                    }
                }
                total / r.len() as f64
            })
            .collect(),
    ))
}

/// NDCG gain function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gains {
    /// gain = grade.
    #[default]
    Linear,
    /// gain = 2^grade − 1.
    Exponential,
}

/// Normalised discounted cumulative gain (within the top k, if given).
pub fn ndcg_at_k<D: Hash + Eq>(
    relevant: &[Judgements<D>],
    retrieved: &[Vec<D>],
    k: Option<usize>,
    gains: Gains,
) -> Result<QueryScores> {
    check_k(k)?;
    queries(relevant, retrieved)?;
    need_relevant(relevant, "NDCG")?;
    let g = |x: f64| {
        if gains == Gains::Linear {
            x
        } else {
            2.0f64.powf(x) - 1.0
        }
    };
    Ok(finish(
        relevant
            .iter()
            .zip(retrieved)
            .map(|(r, d)| {
                let dcg: f64 = top(d, k)
                    .iter()
                    .enumerate()
                    .map(|(i, x)| g(r.get(x).copied().unwrap_or(0.0)) / ((i + 2) as f64).log2())
                    .sum();
                let mut ideal: Vec<f64> = r.values().copied().collect();
                ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                if let Some(k) = k {
                    ideal.truncate(k);
                }
                let idcg: f64 = ideal
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| g(x) / ((i + 2) as f64).log2())
                    .sum();
                dcg / idcg
            })
            .collect(),
    ))
}
