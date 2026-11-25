//! Question answering and reasoning: SQuAD exact match and token F1, majority-vote (self-consistency)
//! accuracy, pass@k, answer extraction and benchmark accuracy, as EvalSuite's Python `evalsuite.text.qa` and
//! `evalsuite.text.reasoning`.
//!
//! ```
//! use evalsuite::qa::{exact_match, normalize_answer, Norm};
//! assert_eq!(normalize_answer("The  Eiffel Tower!"), "eiffel tower");
//! let s = exact_match(&[vec!["Paris"]], &["paris."], Norm::Squad).unwrap();
//! assert_eq!(s, [1.0]);
//! ```

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::error::{invalid, Result};
use crate::generation::{check, py_space, py_split};

/// How answers are normalised before comparison.
#[derive(Clone, Copy, Default)]
pub enum Norm<'a> {
    /// SQuAD: lowercase, drop ASCII punctuation and the articles a/an/the, collapse whitespace.
    #[default]
    Squad,
    /// Compare as given.
    None,
    /// A custom function.
    Custom(&'a dyn Fn(&str) -> String),
}

impl Norm<'_> {
    fn apply(&self, s: &str) -> String {
        match self {
            Norm::Squad => normalize_answer(s),
            Norm::None => s.to_owned(),
            Norm::Custom(f) => f(s),
        }
    }
}

fn articles() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b(a|an|the)\b").unwrap())
}

/// SQuAD answer normalisation.
pub fn normalize_answer(text: &str) -> String {
    let t: String = text.to_lowercase().chars().filter(|c| !c.is_ascii_punctuation()).collect();
    py_split(&articles().replace_all(&t, " ")).join(" ")
}

fn f1(pred: &str, reference: &str) -> f64 {
    let (p, r) = (py_split(pred), py_split(reference));
    if p.is_empty() || r.is_empty() {
        return f64::from(u8::from(p == r));
    }
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for t in &r {
        *counts.entry(t).or_default() += 1;
    }
    let mut same = 0usize;
    for t in &p {
        if let Some(c) = counts.get_mut(t) {
            if *c > 0 {
                *c -= 1;
                same += 1;
            }
        }
    }
    if same == 0 {
        return 0.0;
    }
    let (pr, rc) = (same as f64 / p.len() as f64, same as f64 / r.len() as f64);
    2.0 * pr * rc / (pr + rc)
}

/// Per-example exact match (1 if the normalised prediction equals any normalised reference).
pub fn exact_match<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], norm: Norm) -> Result<Vec<f64>> {
    check(references, predictions)?;
    Ok(predictions
        .iter()
        .zip(references)
        .map(|(p, rs)| {
            let p = norm.apply(p.as_ref());
            f64::from(u8::from(rs.iter().any(|r| norm.apply(r.as_ref()) == p)))
        })
        .collect())
}

/// Per-example token F1 (best over the references).
pub fn token_f1<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], norm: Norm) -> Result<Vec<f64>> {
    check(references, predictions)?;
    Ok(predictions
        .iter()
        .zip(references)
        .map(|(p, rs)| {
            let p = norm.apply(p.as_ref());
            rs.iter().map(|r| f1(&p, &norm.apply(r.as_ref()))).fold(f64::NEG_INFINITY, f64::max)
        })
        .collect())
}

/// Per-question majority-vote accuracy: the most frequent normalised sample (first seen among ties) is
/// compared with the references.
pub fn majority_vote_accuracy<R: AsRef<str>, S: AsRef<str>>(references: &[Vec<R>], samples: &[Vec<S>], norm: Norm) -> Result<Vec<f64>> {
    let firsts: Vec<&str> = samples.iter().map(|_| "").collect();
    check(references, &firsts)?;
    let mut out = Vec::with_capacity(samples.len());
    for (i, (answers, rs)) in samples.iter().zip(references).enumerate() {
        if answers.is_empty() {
            return invalid(format!("samples[{i}] must be a non-empty list of answer strings."));
        }
        let normed: Vec<String> = answers.iter().map(|a| norm.apply(a.as_ref())).collect();
        let mut counts: HashMap<&str, (usize, usize)> = HashMap::new();
        for (j, a) in normed.iter().enumerate() {
            counts.entry(a).or_insert((0, j)).0 += 1;
        }
        let top = counts.iter().max_by(|a, b| a.1 .0.cmp(&b.1 .0).then(b.1 .1.cmp(&a.1 .1))).unwrap().0;
        out.push(f64::from(u8::from(rs.iter().any(|r| norm.apply(r.as_ref()) == *top))));
    }
    Ok(out)
}

/// Unbiased pass@k per problem (Chen et al. 2021) from `n` samples with `c` correct; every `n >= k`.
pub fn pass_at_k(n_samples: &[u64], n_correct: &[u64], k: u64) -> Result<Vec<f64>> {
    if k == 0 {
        return invalid("k must be a positive integer.");
    }
    if n_correct.is_empty() {
        return invalid("n_correct is empty. Provide at least one example.");
    }
    if n_samples.len() != n_correct.len() {
        return invalid(format!("n_samples and n_correct differ in length: {} and {}.", n_samples.len(), n_correct.len()));
    }
    if n_samples.iter().zip(n_correct).any(|(n, c)| c > n) {
        return invalid("Each n_correct must be between 0 and its n_samples.");
    }
    if let Some(m) = n_samples.iter().copied().filter(|&n| n < k).min() {
        return invalid(format!("pass@{k} needs at least k = {k} samples per problem; the smallest n is {m}."));
    }
    Ok(n_samples
        .iter()
        .zip(n_correct)
        .map(|(&n, &c)| {
            if n - c < k {
                return 1.0;
            }
            let prod: f64 = (n - c + 1..=n).map(|i| 1.0 - k as f64 / i as f64).product();
            1.0 - prod
        })
        .collect())
}

/// How [`extract_answer`] finds the final answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnswerStyle {
    /// The number after `####` if present, else the last number (commas, `$` and a trailing `.0` removed).
    #[default]
    Gsm8k,
    /// The content of the last `\boxed{...}` (or `\fbox{...}`).
    Boxed,
    /// The last standalone option letter A–J.
    Choice,
    /// The last non-empty line.
    LastLine,
}

fn number() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"-?\$?\d[\d,]*(?:\.\d+)?").unwrap())
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

fn clean_number(s: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let s = s.replace([',', '$'], "");
    let s = py_strip(&s);
    if RE.get_or_init(|| Regex::new(r"\A-?\d+\.0+\z").unwrap()).is_match(s) {
        return s.split('.').next().unwrap().to_owned();
    }
    s.to_owned()
}

fn boxed(text: &str) -> Option<&str> {
    let start = text.rfind("\\boxed").or_else(|| text.rfind("\\fbox"))?;
    let i = start + text[start..].find('{')?;
    let mut depth = 0i64;
    for (j, b) in text.bytes().enumerate().skip(i) {
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(&text[i + 1..j]);
            }
        }
    }
    None
}

/// Python's `str.splitlines()`.
fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && it.peek().map(|p| p.1) == Some('\n') {
                it.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

fn choice(text: &str) -> Option<String> {
    // Python: findall(r"(?:^|[^A-Za-z])\(?([A-J])\)?(?=[^A-Za-z]|$)") — non-overlapping, leftmost matches.
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let not_alpha = |c: char| !c.is_ascii_alphabetic();
    let ahead_ok = |k: usize| k >= n || not_alpha(chars[k]);
    let mut found = None;
    let mut pos = 0;
    while pos <= n {
        // Try a match starting at `pos`; returns (letter, end).
        let try_at = |s: usize| -> Option<(char, usize)> {
            let mut starts = Vec::new();
            if s == 0 {
                starts.push(0);
            }
            if s < n && not_alpha(chars[s]) {
                starts.push(s + 1);
            }
            for b in starts {
                for paren in [true, false] {
                    let mut k = b;
                    if paren {
                        if k < n && chars[k] == '(' {
                            k += 1;
                        } else {
                            continue;
                        }
                    }
                    if k < n && ('A'..='J').contains(&chars[k]) {
                        let letter = chars[k];
                        k += 1;
                        if k < n && chars[k] == ')' && ahead_ok(k + 1) {
                            return Some((letter, k + 1));
                        }
                        if ahead_ok(k) {
                            return Some((letter, k));
                        }
                    }
                }
            }
            None
        };
        match try_at(pos) {
            Some((l, end)) => {
                found = Some(l.to_string());
                pos = if end == pos { pos + 1 } else { end };
            }
            None => pos += 1,
        }
    }
    found
}

/// Extracts the final answer from a model output; `None` when nothing is found.
pub fn extract_answer(text: &str, style: AnswerStyle) -> Option<String> {
    match style {
        AnswerStyle::Gsm8k => {
            if text.contains("####") {
                let tail = text.rsplit("####").next().unwrap();
                return match number().find(tail) {
                    Some(m) => Some(clean_number(m.as_str())),
                    None => Some(py_strip(tail).to_owned()).filter(|s| !s.is_empty()),
                };
            }
            number().find_iter(text).last().map(|m| clean_number(m.as_str()))
        }
        AnswerStyle::Boxed => boxed(text).map(|b| py_strip(b).to_owned()),
        AnswerStyle::Choice => choice(text),
        AnswerStyle::LastLine => py_splitlines(py_strip(text)).into_iter().map(py_strip).filter(|l| !l.is_empty()).last().map(str::to_owned),
    }
}

/// Benchmark accuracy with the number of outputs where no answer was found.
#[derive(Debug, Clone, PartialEq)]
pub struct Benchmark {
    pub value: f64,
    pub per_example: Vec<bool>,
    pub no_answer_found: usize,
}

/// Accuracy of answers extracted from outputs against gold answers (a gold solution's `####` or `\boxed`
/// answer is used when present). `normalize` optionally maps both sides.
pub fn benchmark_accuracy<R: AsRef<str>, O: AsRef<str>>(references: &[R], outputs: &[O], style: AnswerStyle, normalize: Option<&dyn Fn(&str) -> String>) -> Result<Benchmark> {
    if outputs.is_empty() {
        return invalid("outputs is empty. Provide at least one example.");
    }
    if references.is_empty() {
        return invalid("references is empty. Provide at least one example.");
    }
    if outputs.len() != references.len() {
        return invalid("references and outputs must have the same length.");
    }
    let fix = |s: &str| normalize.map_or_else(|| s.to_owned(), |f| f(s));
    let (mut hits, mut missing) = (Vec::new(), 0usize);
    for (g, o) in references.iter().zip(outputs) {
        let g = g.as_ref();
        let mut gold = if matches!(style, AnswerStyle::Gsm8k | AnswerStyle::Boxed) && (g.contains("####") || g.contains("\\boxed")) {
            extract_answer(g, style)
        } else {
            Some(py_strip(g).to_owned())
        };
        if style == AnswerStyle::Gsm8k {
            gold = gold.map(|x| clean_number(&x));
        }
        match extract_answer(o.as_ref(), style) {
            None => {
                missing += 1;
                hits.push(false);
            }
            // Python compares fix(pred) == fix(gold); a gold of None never equals a string.
            Some(p) => hits.push(gold.is_some_and(|gd| fix(&p) == fix(&gd))),
        }
    }
    let value = hits.iter().filter(|&&h| h).count() as f64 / hits.len() as f64;
    Ok(Benchmark { value, per_example: hits, no_answer_found: missing })
}
