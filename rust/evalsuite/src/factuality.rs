//! Factuality and grounding metrics over verification verdicts (faithfulness, hallucination rate,
//! groundedness, ALCE citation precision / recall, claim-verification accuracy, knowledge consistency, answer
//! correctness and relevance, abstention), as EvalSuite's Python `evalsuite.text.factuality`.
//!
//! The verdicts come from your verification step (annotators, an NLI model, an LLM judge); these functions
//! aggregate them consistently.
//!
//! ```
//! use evalsuite::factuality::{faithfulness, Average, Claim::*};
//! let r = faithfulness(&[vec![Supported, Unsupported], vec![Supported]], Average::Macro).unwrap();
//! assert_eq!(r.value, 0.75);
//! ```

use std::collections::HashSet;
use std::hash::Hash;

use crate::error::{invalid, Result};
use crate::util::{mean, np_sum};

/// A claim verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Claim {
    Supported,
    Contradicted,
    Unsupported,
}

impl Claim {
    /// Parses the usual verdict strings (case-insensitive): supported / support / entailed / entailment /
    /// true / yes; contradicted / contradiction / refuted / false; unsupported / not enough info / nei /
    /// neutral / unknown / no.
    pub fn parse(s: &str) -> Result<Claim> {
        match s.trim().to_lowercase().as_str() {
            "supported" | "support" | "entailed" | "entailment" | "true" | "yes" => Ok(Claim::Supported),
            "contradicted" | "contradiction" | "refuted" | "false" => Ok(Claim::Contradicted),
            "unsupported" | "not enough info" | "nei" | "neutral" | "unknown" | "no" => Ok(Claim::Unsupported),
            _ => invalid(format!("verdict must be a boolean or one of 'supported', 'contradicted', 'unsupported'; got {s:?}.")),
        }
    }
}

impl From<bool> for Claim {
    fn from(b: bool) -> Claim {
        if b {
            Claim::Supported
        } else {
            Claim::Unsupported
        }
    }
}

/// How per-answer counts are aggregated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    /// Pool all claims.
    Micro,
    /// Mean per answer, skipping answers with nothing to score.
    Macro,
}

/// An aggregated score with the per-answer values (NaN for answers with nothing to score).
#[derive(Debug, Clone, PartialEq)]
pub struct Aggregated {
    pub value: f64,
    pub per_answer: Vec<f64>,
}

fn aggregate(hits: &[f64], totals: &[f64], avg: Average, name: &str) -> Result<Aggregated> {
    let per: Vec<f64> = hits.iter().zip(totals).map(|(h, t)| if *t > 0.0 { h / t } else { f64::NAN }).collect();
    let value = match avg {
        Average::Micro => {
            let t = np_sum(totals);
            if t == 0.0 {
                return invalid(format!("{name} is undefined: there are no claims to score."));
            }
            np_sum(hits) / t
        }
        Average::Macro => {
            let ok: Vec<f64> = per.iter().copied().filter(|x| !x.is_nan()).collect();
            if ok.is_empty() {
                return invalid(format!("{name} is undefined: every answer has zero claims."));
            }
            np_sum(&ok) / ok.len() as f64
        }
    };
    Ok(Aggregated { value, per_answer: per })
}

fn claims(c: &[Vec<Claim>]) -> Result<()> {
    if c.is_empty() {
        return invalid("claim_verdicts is empty.");
    }
    Ok(())
}

fn count(c: &[Vec<Claim>], f: impl Fn(Claim) -> bool) -> (Vec<f64>, Vec<f64>) {
    (c.iter().map(|a| a.iter().filter(|&&v| f(v)).count() as f64).collect(), c.iter().map(|a| a.len() as f64).collect())
}

/// Share of an answer's claims supported by the evidence (FActScore / RAGAS faithfulness).
pub fn faithfulness(claim_verdicts: &[Vec<Claim>], avg: Average) -> Result<Aggregated> {
    claims(claim_verdicts)?;
    let (h, t) = count(claim_verdicts, |v| v == Claim::Supported);
    aggregate(&h, &t, avg, "Faithfulness")
}

/// Which claims count as hallucinated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hallucination {
    /// Every claim that is not supported.
    #[default]
    Unsupported,
    /// Only claims the evidence contradicts.
    Contradicted,
}

/// Share of claims that are hallucinated.
pub fn hallucination_rate(claim_verdicts: &[Vec<Claim>], kind: Hallucination, avg: Average) -> Result<Aggregated> {
    claims(claim_verdicts)?;
    let (h, t) = match kind {
        Hallucination::Unsupported => count(claim_verdicts, |v| v != Claim::Supported),
        Hallucination::Contradicted => count(claim_verdicts, |v| v == Claim::Contradicted),
    };
    aggregate(&h, &t, avg, "Hallucination rate")
}

/// Mean support score (each in [0, 1]) per answer, or pooled.
pub fn groundedness(support_scores: &[Vec<f64>], avg: Average) -> Result<Aggregated> {
    if support_scores.is_empty() {
        return invalid("support_scores must be a non-empty list of score lists (one per answer).");
    }
    for (i, a) in support_scores.iter().enumerate() {
        if a.iter().any(|&x| !x.is_finite() || !(0.0..=1.0).contains(&x)) {
            return invalid(format!("support_scores[{i}] must be finite values in [0, 1]."));
        }
    }
    let sums: Vec<f64> = support_scores.iter().map(|a| np_sum(a)).collect();
    let counts: Vec<f64> = support_scores.iter().map(|a| a.len() as f64).collect();
    aggregate(&sums, &counts, avg, "Groundedness")
}

/// One statement of an answer for the citation metrics: whether its citations together support it, and for
/// each citation whether that passage is relevant to it.
#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub supported: bool,
    pub citations: Vec<bool>,
}

fn citations(c: &[Vec<Statement>]) -> Result<()> {
    if c.is_empty() {
        return invalid("citations must be a non-empty list (one list of statements per answer).");
    }
    Ok(())
}

/// ALCE citation recall: share of statements with at least one citation whose citations support them.
pub fn citation_recall(answers: &[Vec<Statement>], avg: Average) -> Result<Aggregated> {
    citations(answers)?;
    let h: Vec<f64> = answers.iter().map(|a| a.iter().filter(|s| s.supported && !s.citations.is_empty()).count() as f64).collect();
    let t: Vec<f64> = answers.iter().map(|a| a.len() as f64).collect();
    aggregate(&h, &t, avg, "Citation recall")
}

/// ALCE citation precision: share of citations that are relevant.
pub fn citation_precision(answers: &[Vec<Statement>], avg: Average) -> Result<Aggregated> {
    citations(answers)?;
    let h: Vec<f64> = answers.iter().map(|a| a.iter().map(|s| s.citations.iter().filter(|&&c| c).count()).sum::<usize>() as f64).collect();
    let t: Vec<f64> = answers.iter().map(|a| a.iter().map(|s| s.citations.len()).sum::<usize>() as f64).collect();
    aggregate(&h, &t, avg, "Citation precision")
}

/// Claim-verification accuracy with the macro F1 over the verdict classes present.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verification {
    pub value: f64,
    pub macro_f1: f64,
    /// F1 of supported / contradicted / unsupported (`None` when the class is absent from both sides).
    pub per_class_f1: [Option<f64>; 3],
}

/// Accuracy of predicted claim verdicts against gold verdicts.
pub fn claim_verification_accuracy(gold: &[Claim], predicted: &[Claim]) -> Result<Verification> {
    if gold.is_empty() || gold.len() != predicted.len() {
        return invalid("gold_verdicts and predicted_verdicts must be non-empty and the same length.");
    }
    let mut per = [None; 3];
    for (k, c) in [Claim::Supported, Claim::Contradicted, Claim::Unsupported].into_iter().enumerate() {
        let tp = gold.iter().zip(predicted).filter(|(g, p)| **g == c && **p == c).count();
        let fp = gold.iter().zip(predicted).filter(|(g, p)| **g != c && **p == c).count();
        let fn_ = gold.iter().zip(predicted).filter(|(g, p)| **g == c && **p != c).count();
        if tp + fp + fn_ > 0 {
            per[k] = Some(2.0 * tp as f64 / (2 * tp + fp + fn_) as f64);
        }
    }
    let f1s: Vec<f64> = per.iter().flatten().copied().collect();
    let acc = gold.iter().zip(predicted).filter(|(g, p)| g == p).count() as f64 / gold.len() as f64;
    Ok(Verification { value: acc, macro_f1: mean(&f1s), per_class_f1: per })
}

/// Share of extracted facts found in a trusted knowledge base.
pub fn knowledge_consistency<F: Eq + Hash>(extracted_facts: &[Vec<F>], knowledge_base: &HashSet<F>, avg: Average) -> Result<Aggregated> {
    if knowledge_base.is_empty() {
        return invalid("knowledge_base is empty.");
    }
    if extracted_facts.is_empty() {
        return invalid("extracted_facts is empty.");
    }
    let h: Vec<f64> = extracted_facts.iter().map(|f| f.iter().filter(|x| knowledge_base.contains(x)).count() as f64).collect();
    let t: Vec<f64> = extracted_facts.iter().map(|f| f.len() as f64).collect();
    aggregate(&h, &t, avg, "Knowledge consistency")
}

/// Per-answer correctness: statement F1 = TP / (TP + (FP + FN)/2) (1 when nothing was to be stated or
/// stated), optionally blended with a semantic similarity in [0, 1] by `weights` (RAGAS: 0.75 / 0.25).
pub fn answer_correctness(tp: &[f64], fp: &[f64], fn_: &[f64], similarity: Option<&[f64]>, weights: (f64, f64)) -> Result<Vec<f64>> {
    if tp.is_empty() || tp.len() != fp.len() || tp.len() != fn_.len() {
        return invalid("tp, fp and fn must be non-empty arrays of the same length.");
    }
    if tp.iter().chain(fp).chain(fn_).any(|&x| x < 0.0) {
        return invalid("Statement counts must be non-negative.");
    }
    let f1: Vec<f64> = (0..tp.len())
        .map(|i| {
            let d = tp[i] + 0.5 * (fp[i] + fn_[i]);
            if d > 0.0 {
                tp[i] / d
            } else {
                1.0
            }
        })
        .collect();
    let Some(sim) = similarity else { return Ok(f1) };
    if sim.len() != tp.len() || sim.iter().any(|&s| !(0.0..=1.0).contains(&s)) {
        return invalid("similarity must hold one value in [0, 1] per answer.");
    }
    let (wf, ws) = weights;
    if wf < 0.0 || ws < 0.0 || wf + ws == 0.0 {
        return invalid("weights must be non-negative and not both zero.");
    }
    Ok(f1.iter().zip(sim).map(|(f, s)| (wf * f + ws * s) / (wf + ws)).collect())
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Per-answer relevance (RAGAS): mean cosine similarity of the question's embedding with the embeddings of
/// questions regenerated from the answer. `generated[i]` holds k ≥ 1 vectors of the question's dimension.
pub fn answer_relevance(question_embeddings: &[Vec<f64>], generated: &[Vec<Vec<f64>>]) -> Result<Vec<f64>> {
    let dim = question_embeddings.first().map_or(0, Vec::len);
    if question_embeddings.is_empty() || generated.len() != question_embeddings.len() || question_embeddings.iter().any(|q| q.len() != dim) {
        return invalid("question_embeddings must be (n, dim) with one generated set per question.");
    }
    question_embeddings
        .iter()
        .zip(generated)
        .enumerate()
        .map(|(i, (q, g))| {
            if g.is_empty() || g.iter().any(|v| v.len() != dim) {
                return invalid(format!("generated_question_embeddings[{i}] must be (k, {dim})."));
            }
            let qn = norm(q);
            if qn == 0.0 || g.iter().any(|v| norm(v) == 0.0) {
                return invalid("Embeddings must not be zero vectors.");
            }
            let cos: Vec<f64> = g.iter().map(|v| v.iter().zip(q).map(|(a, b)| a * b).sum::<f64>() / (norm(v) * qn)).collect();
            Ok(mean(&cos))
        })
        .collect()
}

/// Abstention accuracy with abstention precision / recall / rate and, with `correct`, coverage and the
/// accuracy on answered questions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Abstention {
    pub value: f64,
    pub abstention_precision: f64,
    pub abstention_recall: f64,
    pub abstention_rate: f64,
    pub coverage: Option<f64>,
    pub answered_accuracy: Option<f64>,
}

/// Share of questions where the system abstained exactly when it should have.
pub fn abstention_accuracy(should_abstain: &[bool], abstained: &[bool], correct: Option<&[bool]>) -> Result<Abstention> {
    if should_abstain.is_empty() || should_abstain.len() != abstained.len() {
        return invalid("should_abstain and abstained must be non-empty and the same length.");
    }
    let n = should_abstain.len() as f64;
    let tp = should_abstain.iter().zip(abstained).filter(|(s, a)| **s && **a).count() as f64;
    let na = abstained.iter().filter(|&&a| a).count() as f64;
    let ns = should_abstain.iter().filter(|&&s| s).count() as f64;
    let (coverage, answered_accuracy) = match correct {
        Some(c) => {
            if c.len() != abstained.len() {
                return invalid("correct must have one value per question.");
            }
            let ans: Vec<bool> = c.iter().zip(abstained).filter(|(_, a)| !**a).map(|(c, _)| *c).collect();
            let acc = if ans.is_empty() { f64::NAN } else { ans.iter().filter(|&&x| x).count() as f64 / ans.len() as f64 };
            (Some((n - na) / n), Some(acc))
        }
        None => (None, None),
    };
    Ok(Abstention {
        value: should_abstain.iter().zip(abstained).filter(|(s, a)| s == a).count() as f64 / n,
        abstention_precision: if na > 0.0 { tp / na } else { f64::NAN },
        abstention_recall: if ns > 0.0 { tp / ns } else { f64::NAN },
        abstention_rate: na / n,
        coverage,
        answered_accuracy,
    })
}
