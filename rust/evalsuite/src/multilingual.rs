//! Multilingual metrics: language identification, bitext mining, performance parity across languages,
//! code-switching robustness, z-normalised direct assessment, cultural appropriateness, and language /
//! cross-lingual consistency, as EvalSuite's Python `evalsuite.llmsys.multilingual`.
//!
//! ```
//! use evalsuite::multilingual::language_parity;
//! let p = language_parity(&[("en", &[0.8][..]), ("sw", &[0.6][..])]).unwrap();
//! assert!((p.value - 0.75).abs() < 1e-12);
//! ```

use crate::error::{invalid, Result};
use crate::generation::py_split;
use crate::summary::{by_group, floats, nonempty, rate, same, Rate};
use crate::util::{mean, var};

/// Language-ID accuracy with macro F1 and per-language recall.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageId {
    pub value: f64,
    pub macro_f1: f64,
    /// (language, recall) for languages present in the truth, sorted.
    pub recall: Vec<(String, f64)>,
}

/// Language-identification accuracy over all labels seen (true or predicted).
pub fn language_id_accuracy<S: AsRef<str>>(
    true_languages: &[S],
    predicted: &[S],
) -> Result<LanguageId> {
    nonempty(true_languages, "true_languages")?;
    nonempty(predicted, "predicted_languages")?;
    same(
        true_languages.len(),
        predicted.len(),
        "true_languages",
        "predicted_languages",
    )?;
    let mut labels: Vec<&str> = true_languages
        .iter()
        .chain(predicted)
        .map(AsRef::as_ref)
        .collect();
    labels.sort_unstable();
    labels.dedup();
    let (mut f1s, mut recall, mut tp_all) = (Vec::new(), Vec::new(), 0usize);
    for l in &labels {
        let tp = true_languages
            .iter()
            .zip(predicted)
            .filter(|(t, p)| t.as_ref() == *l && p.as_ref() == *l)
            .count();
        let fp = true_languages
            .iter()
            .zip(predicted)
            .filter(|(t, p)| t.as_ref() != *l && p.as_ref() == *l)
            .count();
        let fn_ = true_languages
            .iter()
            .zip(predicted)
            .filter(|(t, p)| t.as_ref() == *l && p.as_ref() != *l)
            .count();
        tp_all += tp;
        f1s.push(if tp > 0 {
            2.0 * tp as f64 / (2 * tp + fp + fn_) as f64
        } else {
            0.0
        });
        if tp + fn_ > 0 {
            recall.push(((*l).to_owned(), tp as f64 / (tp + fn_) as f64));
        }
    }
    Ok(LanguageId {
        value: tp_all as f64 / true_languages.len() as f64,
        macro_f1: mean(&f1s),
        recall,
    })
}

/// How candidate translations are scored in bitext mining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scoring {
    #[default]
    Cosine,
    /// Ratio margin (Artetxe & Schwenk 2019) with `k` nearest neighbours.
    Margin(usize),
}

/// Bitext mining accuracy in both directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bitext {
    pub value: f64,
    pub source_to_target: f64,
    pub target_to_source: f64,
    pub mean_pair_cosine: f64,
}

fn unit_rows(a: &[Vec<f64>], name: &str) -> Result<Vec<Vec<f64>>> {
    let d = a.first().map_or(0, Vec::len);
    if a.len() < 2 || d == 0 || a.iter().any(|r| r.len() != d) {
        return invalid(format!(
            "{name} must be a 2-D array with at least two rows (one per sentence)."
        ));
    }
    if a.iter().flatten().any(|x| !x.is_finite()) {
        return invalid(format!("{name} contains NaN or infinite values."));
    }
    a.iter()
        .map(|r| {
            let n = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            if n == 0.0 {
                return invalid(format!("{name} contains a zero vector."));
            }
            Ok(r.iter().map(|x| x / n).collect())
        })
        .collect()
}

fn argmax(v: impl Iterator<Item = f64>) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for (i, x) in v.enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0
}

/// Share of sentences whose nearest neighbour (row i of source ↔ row i of target are translations) is its
/// translation, averaged over both directions.
pub fn bitext_mining_accuracy(
    source: &[Vec<f64>],
    target: &[Vec<f64>],
    scoring: Scoring,
) -> Result<Bitext> {
    let s = unit_rows(source, "source_embeddings")?;
    let t = unit_rows(target, "target_embeddings")?;
    if s.len() != t.len() || s[0].len() != t[0].len() {
        return invalid("source_embeddings and target_embeddings must have the same shape (row i of each is a translation pair).");
    }
    let n = s.len();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let mut sim: Vec<Vec<f64>> = s
        .iter()
        .map(|a| t.iter().map(|b| dot(a, b)).collect())
        .collect();
    let pair_cos = mean(&(0..n).map(|i| sim[i][i]).collect::<Vec<_>>());
    if let Scoring::Margin(k) = scoring {
        let kk = k.min(n).max(1);
        let top = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            mean(&v[v.len() - kk..])
        };
        let rs: Vec<f64> = (0..n).map(|i| top(sim[i].clone())).collect();
        let rt: Vec<f64> = (0..n)
            .map(|j| top((0..n).map(|i| sim[i][j]).collect()))
            .collect();
        for i in 0..n {
            for j in 0..n {
                sim[i][j] /= (rs[i] + rt[j]) / 2.0;
            }
        }
    }
    let fwd = (0..n)
        .filter(|&i| argmax(sim[i].iter().copied()) == i)
        .count() as f64
        / n as f64;
    let bwd = (0..n)
        .filter(|&j| argmax((0..n).map(|i| sim[i][j])) == j)
        .count() as f64
        / n as f64;
    Ok(Bitext {
        value: (fwd + bwd) / 2.0,
        source_to_target: fwd,
        target_to_source: bwd,
        mean_pair_cosine: pair_cos,
    })
}

/// Language parity: worst / best language mean score (0 when the worst is ≤ 0).
#[derive(Debug, Clone, PartialEq)]
pub struct Parity {
    pub value: f64,
    pub max_gap: f64,
    pub std: f64,
    pub worst_language: String,
    pub by_language: Vec<(String, f64)>,
}

/// Parity of mean scores across at least two languages (input order kept).
pub fn language_parity(scores_by_language: &[(&str, &[f64])]) -> Result<Parity> {
    if scores_by_language.len() < 2 {
        return invalid("scores_by_language must map at least two languages to scores.");
    }
    let mut means = Vec::new();
    for (lang, v) in scores_by_language {
        floats(v, &format!("scores_by_language[{lang:?}]"), None, None)?;
        means.push(((*lang).to_owned(), mean(v)));
    }
    let vals: Vec<f64> = means.iter().map(|m| m.1).collect();
    let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
    if mx <= 0.0 {
        return invalid("Scores must be positive for some language to compute parity.");
    }
    let worst = means.iter().find(|m| m.1 == mn).unwrap().0.clone();
    Ok(Parity {
        value: mn.max(0.0) / mx,
        max_gap: mx - mn,
        std: var(&vals, 1).sqrt(),
        worst_language: worst,
        by_language: means,
    })
}

/// Code-switching robustness: accuracy on code-switched inputs, with the monolingual accuracy, the drop and
/// the flip rate (monolingual-correct examples that fail when code-switched).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeSwitching {
    pub value: f64,
    pub monolingual_accuracy: f64,
    pub absolute_drop: f64,
    pub flip_rate: f64,
}

/// Robustness of correctness to code-switched rewrites of the same inputs.
pub fn code_switching_robustness(
    monolingual_correct: &[bool],
    code_switched_correct: &[bool],
) -> Result<CodeSwitching> {
    nonempty(monolingual_correct, "monolingual_correct")?;
    nonempty(code_switched_correct, "code_switched_correct")?;
    same(
        monolingual_correct.len(),
        code_switched_correct.len(),
        "monolingual_correct",
        "code_switched_correct",
    )?;
    let n = monolingual_correct.len() as f64;
    let am = monolingual_correct.iter().filter(|&&x| x).count() as f64 / n;
    let ac = code_switched_correct.iter().filter(|&&x| x).count() as f64 / n;
    let m = monolingual_correct.iter().filter(|&&x| x).count();
    let flips = monolingual_correct
        .iter()
        .zip(code_switched_correct)
        .filter(|(a, b)| **a && !**b)
        .count();
    Ok(CodeSwitching {
        value: ac,
        monolingual_accuracy: am,
        absolute_drop: am - ac,
        flip_rate: if m > 0 {
            flips as f64 / m as f64
        } else {
            f64::NAN
        },
    })
}

/// Direct assessment: mean of scores z-normalised per rater, with the raw mean and per-system means.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectAssessment {
    pub value: f64,
    pub raw_mean: f64,
    pub n_raters: usize,
    pub by_system_z: Vec<(String, f64)>,
    pub by_system_raw: Vec<(String, f64)>,
}

/// Z-normalised direct assessment (WMT style); a rater with one score or no spread contributes z = 0.
pub fn direct_assessment<R: AsRef<str>, S: AsRef<str>>(
    scores: &[f64],
    raters: &[R],
    systems: Option<&[S]>,
) -> Result<DirectAssessment> {
    floats(scores, "scores", None, None)?;
    same(scores.len(), raters.len(), "scores", "raters")?;
    let mut z = vec![0.0; scores.len()];
    let mut names: Vec<&str> = raters.iter().map(AsRef::as_ref).collect();
    names.sort_unstable();
    names.dedup();
    for r in &names {
        let idx: Vec<usize> = (0..scores.len())
            .filter(|&i| raters[i].as_ref() == *r)
            .collect();
        let v: Vec<f64> = idx.iter().map(|&i| scores[i]).collect();
        let sd = if v.len() > 1 { var(&v, 1).sqrt() } else { 0.0 };
        let m = mean(&v);
        for &i in &idx {
            z[i] = if sd > 0.0 { (scores[i] - m) / sd } else { 0.0 };
        }
    }
    let (bz, br) = match systems {
        Some(s) => (by_group(&z, s, "systems")?, by_group(scores, s, "systems")?),
        None => (vec![], vec![]),
    };
    Ok(DirectAssessment {
        value: mean(&z),
        raw_mean: mean(scores),
        n_raters: names.len(),
        by_system_z: bz,
        by_system_raw: br,
    })
}

/// Cultural appropriateness: mean rating normalised by the scale, with per-region means and their gap.
pub fn cultural_appropriateness<S: AsRef<str>>(
    ratings: &[f64],
    scale: (f64, f64),
    regions: Option<&[S]>,
) -> Result<(f64, Vec<(String, f64)>, Option<f64>)> {
    let (lo, hi) = scale;
    if hi <= lo {
        return invalid("scale must be (low, high) with high > low.");
    }
    floats(ratings, "ratings", Some(lo), Some(hi))?;
    let v: Vec<f64> = ratings.iter().map(|r| (r - lo) / (hi - lo)).collect();
    let (by, gap) = match regions {
        Some(g) => {
            let by = by_group(&v, g, "regions")?;
            let mx = by.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
            let mn = by.iter().map(|b| b.1).fold(f64::INFINITY, f64::min);
            (by, Some(mx - mn))
        }
        None => (vec![], None),
    };
    Ok((mean(&v), by, gap))
}

/// Share of outputs in the expected language.
pub fn language_consistency<S: AsRef<str>>(expected: &[S], detected: &[S]) -> Result<Rate> {
    nonempty(expected, "expected_languages")?;
    nonempty(detected, "detected_languages")?;
    same(
        expected.len(),
        detected.len(),
        "expected_languages",
        "detected_languages",
    )?;
    Ok(rate(
        &expected
            .iter()
            .zip(detected)
            .map(|(a, b)| a.as_ref() == b.as_ref())
            .collect::<Vec<_>>(),
    ))
}

/// Cross-lingual consistency: share of questions answered identically in every language, and the mean
/// pairwise agreement. Answers are compared lowercased with whitespace collapsed unless `normalize` is given.
pub fn cross_lingual_consistency<S: AsRef<str>>(
    answers_by_language: &[Vec<S>],
    normalize: Option<&dyn Fn(&str) -> String>,
) -> Result<(f64, f64)> {
    nonempty(answers_by_language, "answers_by_language")?;
    let (mut full, mut pair) = (Vec::new(), Vec::new());
    for (i, r) in answers_by_language.iter().enumerate() {
        if r.len() < 2 {
            return invalid(format!(
                "answers_by_language[{i}] must map at least two languages to answers."
            ));
        }
        let v: Vec<String> = r
            .iter()
            .map(|a| {
                normalize.map_or_else(
                    || py_split(&a.as_ref().to_lowercase()).join(" "),
                    |f| f(a.as_ref()),
                )
            })
            .collect();
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
