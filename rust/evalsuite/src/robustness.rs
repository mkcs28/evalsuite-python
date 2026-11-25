//! Robustness metrics: adversarial and noisy-input accuracy, paraphrase consistency, out-of-distribution
//! accuracy, distribution-shift drop, counterfactual invariance, response stability, contradiction and error
//! rates, recovery, prompt sensitivity and truncation sensitivity, plus a typo generator, as EvalSuite's
//! Python `evalsuite.llmsys.robustness`.
//!
//! ```
//! use evalsuite::robustness::noise_robustness;
//! let r = noise_robustness(&[true, true, false], &[true, false, false]).unwrap();
//! assert_eq!(r.flip_rate, 0.5);
//! ```

use std::collections::HashMap;

use crate::error::{invalid, Result};
use crate::generation::py_split;
use crate::summary::{by_group, floats, nonempty, rate, same, Rate};
use crate::util::{mean, var};

fn pair(c: &[bool], p: &[bool], a: &str, b: &str) -> Result<()> {
    nonempty(c, a)?;
    nonempty(p, b)?;
    same(c.len(), p.len(), a, b)
}

fn share(b: &[bool]) -> f64 {
    b.iter().filter(|&&x| x).count() as f64 / b.len() as f64
}

/// Accuracy under perturbation compared with clean inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drop {
    /// Accuracy on perturbed inputs.
    pub value: f64,
    pub clean_accuracy: f64,
    pub absolute_drop: f64,
    pub relative_drop: f64,
    /// Share of clean-correct examples that fail when perturbed (for adversarial inputs: the attack success
    /// rate).
    pub flip_rate: f64,
}

fn drop_result(c: &[bool], p: &[bool]) -> Drop {
    let (ca, pa) = (share(c), share(p));
    let k = c.iter().filter(|&&x| x).count();
    let flipped = c.iter().zip(p).filter(|(a, b)| **a && !**b).count();
    Drop {
        value: pa,
        clean_accuracy: ca,
        absolute_drop: ca - pa,
        relative_drop: if ca > 0.0 { (ca - pa) / ca } else { f64::NAN },
        flip_rate: if k > 0 { flipped as f64 / k as f64 } else { f64::NAN },
    }
}

/// Robust accuracy on adversarial rewrites of the same inputs.
pub fn adversarial_robustness(clean_correct: &[bool], adversarial_correct: &[bool]) -> Result<Drop> {
    pair(clean_correct, adversarial_correct, "clean_correct", "adversarial_correct")?;
    Ok(drop_result(clean_correct, adversarial_correct))
}

/// Accuracy on noisy versions (typos, casing, …) of the same inputs.
pub fn noise_robustness(clean_correct: &[bool], noisy_correct: &[bool]) -> Result<Drop> {
    pair(clean_correct, noisy_correct, "clean_correct", "noisy_correct")?;
    Ok(drop_result(clean_correct, noisy_correct))
}

/// Adds reproducible typos: each letter is, with probability `rate`, swapped with its neighbour, deleted,
/// doubled with a random letter or substituted (equally likely). Uses this crate's seeded generator, so the
/// typos differ from the Python package's (NumPy-seeded) ones.
pub fn add_typos<S: AsRef<str>>(texts: &[S], rate_: f64, seed: u64) -> Result<Vec<String>> {
    if !(0.0..=1.0).contains(&rate_) {
        return invalid("rate must be in [0, 1].");
    }
    let mut rng = evalsuite_core::resample::Xoshiro256pp::new(seed, 0);
    let mut uniform = || rng.below(1 << 53) as f64 / (1u64 << 53) as f64;
    let letter = |u: f64| (b'a' + (u * 26.0) as u8).min(b'z') as char;
    Ok(texts
        .iter()
        .map(|t| {
            let chars: Vec<char> = t.as_ref().chars().collect();
            let mut res = String::with_capacity(chars.len() + 4);
            let mut j = 0;
            while j < chars.len() {
                let ch = chars[j];
                if ch.is_alphabetic() && uniform() < rate_ {
                    match (uniform() * 4.0) as u8 {
                        0 if j + 1 < chars.len() => {
                            res.push(chars[j + 1]);
                            res.push(ch);
                            j += 2;
                            continue;
                        }
                        1 => {
                            j += 1;
                            continue;
                        }
                        2 => {
                            res.push(ch);
                            res.push(letter(uniform()));
                        }
                        _ => res.push(letter(uniform())),
                    }
                } else {
                    res.push(ch);
                }
                j += 1;
            }
            res
        })
        .collect())
}

fn norm(a: &str, normalize: Option<&dyn Fn(&str) -> String>) -> String {
    normalize.map_or_else(|| py_split(&a.to_lowercase()).join(" "), |f| f(a))
}

fn agreement(v: &[String]) -> (bool, f64, usize) {
    let (mut eq, mut n) = (0usize, 0usize);
    for a in 0..v.len() {
        for b in a + 1..v.len() {
            eq += usize::from(v[a] == v[b]);
            n += 1;
        }
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for x in v {
        *counts.entry(x).or_default() += 1;
    }
    (v.iter().all(|x| *x == v[0]), eq as f64 / n as f64, counts.values().copied().max().unwrap_or(0))
}

/// Share of items answered identically under every paraphrase, and the mean pairwise agreement.
pub fn paraphrase_consistency<S: AsRef<str>>(answers: &[Vec<S>], normalize: Option<&dyn Fn(&str) -> String>) -> Result<(f64, f64)> {
    nonempty(answers, "answers")?;
    if answers.iter().any(|r| r.len() < 2) {
        return invalid("Every item needs answers to at least two paraphrases.");
    }
    let st: Vec<(bool, f64, usize)> = answers.iter().map(|r| agreement(&r.iter().map(|a| norm(a.as_ref(), normalize)).collect::<Vec<_>>())).collect();
    Ok((mean(&st.iter().map(|s| f64::from(u8::from(s.0))).collect::<Vec<_>>()), mean(&st.iter().map(|s| s.1).collect::<Vec<_>>())))
}

/// Out-of-distribution accuracy with the in-distribution accuracy and their gap.
pub fn ood_accuracy(correct: &[bool], is_ood: &[bool]) -> Result<(f64, f64, f64)> {
    pair(correct, is_ood, "correct", "is_ood")?;
    let ood: Vec<bool> = correct.iter().zip(is_ood).filter(|(_, &o)| o).map(|(c, _)| *c).collect();
    let ind: Vec<bool> = correct.iter().zip(is_ood).filter(|(_, &o)| !o).map(|(c, _)| *c).collect();
    if ood.is_empty() || ind.is_empty() {
        return invalid("Need both in-distribution and out-of-distribution examples.");
    }
    let (o, i) = (share(&ood), share(&ind));
    Ok((o, i, i - o))
}

/// Mean score drop from source to target distribution, with a Welch 95% interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShiftDrop {
    pub value: f64,
    pub source_mean: f64,
    pub target_mean: f64,
    pub relative_drop: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Source mean − target mean, with Welch's interval (NaN with fewer than two scores on a side).
pub fn distribution_shift_drop(source_scores: &[f64], target_scores: &[f64]) -> Result<ShiftDrop> {
    floats(source_scores, "source_scores", None, None)?;
    floats(target_scores, "target_scores", None, None)?;
    let (ms, mt) = (mean(source_scores), mean(target_scores));
    let drop = ms - mt;
    let (ns, nt) = (source_scores.len() as f64, target_scores.len() as f64);
    let (lo, hi) = if ns > 1.0 && nt > 1.0 {
        let (vs, vt) = (var(source_scores, 1) / ns, var(target_scores, 1) / nt);
        let se = (vs + vt).sqrt();
        let dof = if vs + vt > 0.0 { (vs + vt).powi(2) / (vs * vs / (ns - 1.0) + vt * vt / (nt - 1.0)) } else { f64::INFINITY };
        let q = if dof.is_finite() { crate::special::t_ppf(0.975, dof) } else { 1.959963984540054 };
        (drop - q * se, drop + q * se)
    } else {
        (f64::NAN, f64::NAN)
    };
    Ok(ShiftDrop { value: drop, source_mean: ms, target_mean: mt, relative_drop: if ms != 0.0 { drop / ms } else { f64::NAN }, ci_low: lo, ci_high: hi })
}

/// Share of counterfactual pairs (inputs differing only in a protected or irrelevant attribute) whose
/// predictions differ, with the rate per group when `groups` is given.
pub fn invariance_violation_rate<S: AsRef<str>, G: AsRef<str>>(original: &[S], counterfactual: &[S], normalize: Option<&dyn Fn(&str) -> String>, groups: Option<&[G]>) -> Result<(Rate, Vec<(String, f64)>)> {
    nonempty(original, "original_predictions")?;
    nonempty(counterfactual, "counterfactual_predictions")?;
    same(original.len(), counterfactual.len(), "original_predictions", "counterfactual_predictions")?;
    let flags: Vec<bool> = original.iter().zip(counterfactual).map(|(a, b)| norm(a.as_ref(), normalize) != norm(b.as_ref(), normalize)).collect();
    let by = match groups {
        Some(g) => by_group(&flags.iter().map(|&f| f64::from(u8::from(f))).collect::<Vec<_>>(), g, "groups")?,
        None => vec![],
    };
    Ok((rate(&flags), by))
}

/// Response stability across repeated samples of one prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stability {
    /// Mean pairwise agreement.
    pub value: f64,
    pub unanimous_rate: f64,
    /// Mean share of samples equal to the most frequent answer.
    pub modal_share: f64,
}

/// Agreement among repeated samples per prompt (at least two each).
pub fn response_stability<S: AsRef<str>>(samples: &[Vec<S>], normalize: Option<&dyn Fn(&str) -> String>) -> Result<Stability> {
    nonempty(samples, "samples")?;
    if samples.iter().any(|r| r.len() < 2) {
        return invalid("Every prompt needs at least two samples.");
    }
    let st: Vec<(bool, f64, usize, usize)> = samples
        .iter()
        .map(|r| {
            let (u, p, m) = agreement(&r.iter().map(|a| norm(a.as_ref(), normalize)).collect::<Vec<_>>());
            (u, p, m, r.len())
        })
        .collect();
    Ok(Stability {
        value: mean(&st.iter().map(|s| s.1).collect::<Vec<_>>()),
        unanimous_rate: mean(&st.iter().map(|s| f64::from(u8::from(s.0))).collect::<Vec<_>>()),
        modal_share: mean(&st.iter().map(|s| s.2 as f64 / s.3 as f64).collect::<Vec<_>>()),
    })
}

/// Share of NLI labels (`contradiction` / `entailment` / `neutral`) over response pairs that are
/// contradictions; `pair_labels[i]` holds the labels of prompt i (pooled).
pub fn contradiction_rate<S: AsRef<str>>(pair_labels: &[Vec<S>]) -> Result<Rate> {
    nonempty(pair_labels, "pair_labels")?;
    let mut flat = Vec::new();
    for (i, v) in pair_labels.iter().enumerate() {
        for l in v {
            match l.as_ref() {
                "contradiction" => flat.push(true),
                "entailment" | "neutral" => flat.push(false),
                other => return invalid(format!("pair_labels[{i}] must be one of ['contradiction', 'entailment', 'neutral']; got {other:?}.")),
            }
        }
    }
    if flat.is_empty() {
        return invalid("pair_labels contains no pairs.");
    }
    Ok(rate(&flat))
}

/// Error rate with the share of each failure status and the timeout rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Errors {
    pub rate: Rate,
    pub by_status: Vec<(String, f64)>,
    pub timeout_rate: f64,
}

/// Share of requests whose status is not in `ok` (e.g. `["ok", "success", "200", "True"]`).
pub fn error_rate<S: AsRef<str>>(statuses: &[S], ok: &[&str]) -> Result<Errors> {
    nonempty(statuses, "statuses")?;
    let fail: Vec<bool> = statuses.iter().map(|s| !ok.contains(&s.as_ref())).collect();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in statuses.iter().filter(|s| !ok.contains(&s.as_ref())) {
        *counts.entry(s.as_ref()).or_default() += 1;
    }
    let n = statuses.len() as f64;
    let mut by: Vec<(String, f64)> = counts.iter().map(|(k, v)| ((*k).to_owned(), *v as f64 / n)).collect();
    by.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Errors { rate: rate(&fail), timeout_rate: counts.get("timeout").copied().unwrap_or(0) as f64 / n, by_status: by })
}

/// Share of failed episodes that recovered, with the failure rate.
pub fn recovery_success_rate(failed: &[bool], recovered: &[bool]) -> Result<(Rate, f64)> {
    pair(failed, recovered, "failed", "recovered")?;
    if failed.iter().zip(recovered).any(|(f, r)| *r && !*f) {
        return invalid("recovered is True for an episode where failed is False.");
    }
    let r: Vec<bool> = failed.iter().zip(recovered).filter(|(f, _)| **f).map(|(_, r)| *r).collect();
    if r.is_empty() {
        return invalid("No episode failed: the recovery rate is undefined.");
    }
    Ok((rate(&r), share(failed)))
}

/// Prompt sensitivity: the spread of accuracy across prompt templates.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensitivity {
    /// Best − worst template accuracy.
    pub value: f64,
    pub std: f64,
    pub worst: f64,
    pub best: f64,
    pub mean: f64,
    pub by_template: Vec<f64>,
    /// Share of examples whose correctness changes across templates.
    pub example_disagreement_rate: f64,
}

/// Accuracy spread across templates; `correct_by_template[t][i]` = correctness (or score) of example i
/// under template t (at least two templates, same examples).
pub fn prompt_sensitivity(correct_by_template: &[Vec<f64>]) -> Result<Sensitivity> {
    for (i, r) in correct_by_template.iter().enumerate() {
        floats(r, &format!("correct_by_template[{i}]"), None, None)?;
    }
    if correct_by_template.len() < 2 {
        return invalid("Need at least two prompt templates.");
    }
    let n = correct_by_template[0].len();
    if correct_by_template.iter().any(|r| r.len() != n) {
        return invalid("Every template must be scored on the same examples.");
    }
    let acc: Vec<f64> = correct_by_template.iter().map(|r| mean(r)).collect();
    let flips = (0..n)
        .filter(|&i| {
            let col = correct_by_template.iter().map(|r| r[i]);
            col.clone().fold(f64::NEG_INFINITY, f64::max) != col.fold(f64::INFINITY, f64::min)
        })
        .count() as f64
        / n as f64;
    let mx = acc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mn = acc.iter().copied().fold(f64::INFINITY, f64::min);
    Ok(Sensitivity { value: mx - mn, std: var(&acc, 1).sqrt(), worst: mn, best: mx, mean: mean(&acc), by_template: acc, example_disagreement_rate: flips })
}

/// Truncation sensitivity: the least-squares slope of accuracy against log2(context length), with the
/// accuracy per length and the drop from the shortest to the longest.
pub fn truncation_sensitivity(correct: &[f64], context_lengths: &[f64]) -> Result<(f64, Vec<(f64, f64)>, f64)> {
    floats(correct, "correct", None, None)?;
    floats(context_lengths, "context_lengths", Some(1.0), None)?;
    same(correct.len(), context_lengths.len(), "correct", "context_lengths")?;
    let mut levels = crate::util::sorted(context_lengths);
    levels.dedup();
    if levels.len() < 2 {
        return invalid("Need at least two distinct context lengths.");
    }
    let acc: Vec<f64> = levels
        .iter()
        .map(|&v| mean(&correct.iter().zip(context_lengths).filter(|(_, &l)| l == v).map(|(c, _)| *c).collect::<Vec<_>>()))
        .collect();
    let x: Vec<f64> = levels.iter().map(|v| v.log2()).collect();
    let (mx, my) = (mean(&x), mean(&acc));
    let sxy: f64 = x.iter().zip(&acc).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let by = levels.iter().copied().zip(acc.iter().copied()).collect();
    Ok((sxy / sxx, by, acc[0] - acc[acc.len() - 1]))
}
