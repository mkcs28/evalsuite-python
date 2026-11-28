//! Retrieval-augmented generation (RAG) pipeline metrics: context precision / recall / relevance from
//! per-chunk or per-claim judgements, latency summaries, end-to-end task success and failure attribution, as
//! EvalSuite's Python `evalsuite.text.rag`.
//!
//! ```
//! use evalsuite::rag::context_precision;
//! // Relevant chunks at ranks 1 and 3: (1/1 + 2/3) / 2.
//! let s = context_precision(&[vec![true, false, true]]).unwrap();
//! assert!((s.mean - 5.0 / 6.0).abs() < 1e-12);
//! ```

use crate::error::{invalid, Result};
pub use crate::retrieval::QueryScores;
use crate::summary::{rate, Rate, Summary};
use crate::util::np_sum;

fn finish(per: Vec<f64>, name: &str) -> Result<QueryScores> {
    let ok: Vec<f64> = per.iter().copied().filter(|v| !v.is_nan()).collect();
    if ok.is_empty() {
        return invalid(format!("{name} is undefined for every query."));
    }
    Ok(QueryScores {
        mean: np_sum(&ok) / ok.len() as f64,
        per_query: per,
    })
}

fn rows(x: &[Vec<bool>], name: &str) -> Result<()> {
    if x.is_empty() {
        return invalid(format!(
            "{name} must be a non-empty list with one list of booleans per query."
        ));
    }
    Ok(())
}

fn share(r: &[bool]) -> f64 {
    if r.is_empty() {
        f64::NAN
    } else {
        r.iter().filter(|&&x| x).count() as f64 / r.len() as f64
    }
}

/// Rank-aware context precision: the mean of precision@k over the ranks k of relevant chunks. A query with no
/// relevant chunk scores 0.
pub fn context_precision(chunk_relevance: &[Vec<bool>]) -> Result<QueryScores> {
    rows(chunk_relevance, "chunk_relevance")?;
    let per = chunk_relevance
        .iter()
        .map(|r| {
            let rel = r.iter().filter(|&&x| x).count();
            if rel == 0 {
                return 0.0;
            }
            let (mut hits, mut terms) = (0usize, Vec::with_capacity(rel));
            for (i, &x) in r.iter().enumerate() {
                if x {
                    hits += 1;
                    terms.push(hits as f64 / (i + 1) as f64);
                }
            }
            np_sum(&terms) / rel as f64
        })
        .collect();
    finish(per, "Context precision")
}

/// Share of the reference answer's claims attributable to the retrieved context (queries without claims are
/// undefined and skipped).
pub fn context_recall(claim_attributed: &[Vec<bool>]) -> Result<QueryScores> {
    rows(claim_attributed, "claim_attributed")?;
    finish(
        claim_attributed.iter().map(|r| share(r)).collect(),
        "Context recall",
    )
}

/// Share of retrieved units (sentences or chunks) relevant to the query.
pub fn context_relevance(unit_relevance: &[Vec<bool>]) -> Result<QueryScores> {
    rows(unit_relevance, "unit_relevance")?;
    finish(
        unit_relevance.iter().map(|r| share(r)).collect(),
        "Context relevance",
    )
}

/// Latency summary (mean, p50/p90/p95/p99, max and the rest); latencies must be finite and ≥ 0.
pub fn latency_summary(latencies: &[f64]) -> Result<Summary> {
    if latencies.is_empty() || latencies.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return invalid("latencies must be non-empty, finite and non-negative.");
    }
    Ok(Summary::of(latencies))
}

/// End-to-end task success rate with its 95% Wilson interval.
pub fn task_success_rate(success: &[bool]) -> Result<Rate> {
    if success.is_empty() {
        return invalid("success is empty.");
    }
    Ok(rate(success))
}

/// The default pipeline stages for [`failure_attribution`].
pub const STAGES: [&str; 5] = [
    "retrieval",
    "generation",
    "evidence",
    "orchestration",
    "other",
];

/// Share of failures per stage (aligned with `stage_names`). `stages[i]` is the stage blamed for request i;
/// `None`, `"none"`, `"success"` and `""` mark successes. Labels are compared trimmed and lowercased.
pub fn failure_attribution(stages: &[Option<&str>], stage_names: &[&str]) -> Result<Vec<f64>> {
    let failed: Vec<String> = stages
        .iter()
        .flatten()
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !matches!(s.as_str(), "none" | "success" | ""))
        .collect();
    let mut unknown: Vec<&String> = failed
        .iter()
        .filter(|s| !stage_names.contains(&s.as_str()))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        unknown.dedup();
        return invalid(format!(
            "Unknown stage(s) {unknown:?}; expected one of {stage_names:?}."
        ));
    }
    if failed.is_empty() {
        return invalid("There are no failures to attribute.");
    }
    Ok(stage_names
        .iter()
        .map(|n| failed.iter().filter(|s| s == n).count() as f64 / failed.len() as f64)
        .collect())
}
