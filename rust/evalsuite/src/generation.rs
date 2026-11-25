//! Text generation metrics, as EvalSuite's Python `evalsuite.text.generation` (sacreBLEU, rouge-score,
//! NLTK and pycocoevalcap definitions): BLEU, sentence BLEU, self-BLEU, chrF/chrF++, TER, ROUGE-1/2/L/Lsum,
//! METEOR, CIDEr-D, distinct-n, cross-entropy and perplexity.
//!
//! `references[i]` holds one or more reference texts for `predictions[i]`.
//!
//! ```
//! use evalsuite::generation::{bleu, BleuOpts};
//! let refs = vec![vec!["the cat is on the mat".to_string()]];
//! let preds = ["the cat is on the mat"];
//! assert!((bleu(&refs, &preds, &BleuOpts::default()).unwrap() - 100.0).abs() < 1e-9);
//! ```

use std::collections::HashMap;

use crate::error::{invalid, Result};
use crate::util::np_sum;

/// Python's `str.isspace()`: Unicode whitespace plus the information separators U+001C–U+001F.
pub(crate) fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.split()` with no argument.
pub(crate) fn py_split(s: &str) -> Vec<&str> {
    s.split(py_space).filter(|t| !t.is_empty()).collect()
}

/// Python's `str.rstrip()`.
pub(crate) fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_space)
}

/// Python's `str.lower()` (full Unicode lowercase mapping).
pub(crate) fn py_lower(s: &str) -> String {
    s.to_lowercase()
}

/// Consecutive integer ids for tokens, in first-seen order.
#[derive(Default)]
pub(crate) struct Vocab<'a> {
    ids: HashMap<&'a str, u32>,
}

impl<'a> Vocab<'a> {
    pub fn id(&mut self, t: &'a str) -> u32 {
        let n = self.ids.len() as u32;
        *self.ids.entry(t).or_insert(n)
    }
    pub fn encode(&mut self, toks: &[&'a str]) -> Vec<u32> {
        toks.iter().map(|t| self.id(t)).collect()
    }
}

fn is_13a_punct(c: char) -> bool {
    matches!(c, ' '..='&' | '('..='+' | '/' | ':'..='@' | '['..='`' | '{'..='~')
}

/// sacreBLEU's default `13a` tokenizer (mteval-v13a).
pub fn tokenize_13a(line: &str) -> Vec<String> {
    let mut s = line.replace("<skipped>", "").replace("-\n", "").replace('\n', " ");
    if s.contains('&') {
        s = s.replace("&quot;", "\"").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">");
    }
    let s = format!(" {s} ");
    // 1. pad punctuation (apostrophe, comma, period and dash excluded)
    let mut a = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if is_13a_punct(c) {
            a.push(' ');
            a.push(c);
            a.push(' ');
        } else {
            a.push(c);
        }
    }
    // 2–4: two-character rules applied left to right without overlap, as `re.sub`.
    let rule = |text: &str, first: &dyn Fn(char) -> bool, second: &dyn Fn(char) -> bool, out: &dyn Fn(char, char) -> String| -> String {
        let cs: Vec<char> = text.chars().collect();
        let mut o = String::with_capacity(text.len() + 16);
        let mut i = 0;
        while i < cs.len() {
            if i + 1 < cs.len() && first(cs[i]) && second(cs[i + 1]) {
                o.push_str(&out(cs[i], cs[i + 1]));
                i += 2;
            } else {
                o.push(cs[i]);
                i += 1;
            }
        }
        o
    };
    let digit = |c: char| c.is_ascii_digit();
    let dotcomma = |c: char| c == '.' || c == ',';
    let b = rule(&a, &|c| !digit(c), &dotcomma, &|x, y| format!("{x} {y} "));
    let c = rule(&b, &dotcomma, &|c| !digit(c), &|x, y| format!(" {x} {y}"));
    let d = rule(&c, &digit, &|c| c == '-', &|x, y| format!("{x} {y} "));
    py_split(&d).into_iter().map(str::to_owned).collect()
}

/// Tokenization for BLEU-style metrics.
#[derive(Clone, Copy)]
pub enum Tokenize<'a> {
    /// sacreBLEU's `13a` (default).
    Thirteen,
    /// Whitespace (`str.split`).
    None,
    /// A custom tokenizer.
    Custom(&'a (dyn Fn(&str) -> Vec<String> + Sync)),
}

impl std::fmt::Debug for Tokenize<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Tokenize::Thirteen => "13a",
            Tokenize::None => "none",
            Tokenize::Custom(_) => "custom",
        })
    }
}

impl Tokenize<'_> {
    fn run(&self, s: &str) -> Vec<String> {
        match self {
            Tokenize::Thirteen => tokenize_13a(s),
            Tokenize::None => py_split(s).into_iter().map(str::to_owned).collect(),
            Tokenize::Custom(f) => f(s),
        }
    }
}

fn check<R: AsRef<str>, P: AsRef<str>>(refs: &[Vec<R>], preds: &[P]) -> Result<()> {
    if preds.is_empty() {
        return invalid("predictions is empty. Provide at least one example.");
    }
    if refs.len() != preds.len() {
        return invalid(format!("references must have one entry per prediction. Received {} and {}.", refs.len(), preds.len()));
    }
    if refs.iter().any(Vec::is_empty) {
        return invalid("Every prediction needs at least one reference.");
    }
    Ok(())
}

/// BLEU smoothing (sacreBLEU's methods).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Smooth {
    None,
    Floor(f64),
    AddK(f64),
    #[default]
    Exp,
}

/// BLEU options (defaults as sacreBLEU).
#[derive(Debug, Clone, Copy)]
pub struct BleuOpts<'a> {
    pub max_order: usize,
    pub smooth: Smooth,
    pub lowercase: bool,
    pub tokenize: Tokenize<'a>,
    pub effective_order: bool,
}

impl Default for BleuOpts<'_> {
    fn default() -> Self {
        BleuOpts { max_order: 4, smooth: Smooth::Exp, lowercase: false, tokenize: Tokenize::Thirteen, effective_order: false }
    }
}

fn prep(t: &str, lower: bool, tok: &Tokenize<'_>) -> Vec<String> {
    let s = if lower { py_lower(t) } else { t.to_owned() };
    tok.run(py_rstrip(&s))
}

fn bleu_score(stats: &[f64], order: usize, smooth: Smooth, effective: bool) -> f64 {
    let (sys, rl) = (stats[0], stats[1]);
    let mut correct = stats[2..2 + order].to_vec();
    let mut total = stats[2 + order..].to_vec();
    let bp = if sys >= rl { 1.0 } else if sys > 0.0 { (1.0 - rl / sys).exp() } else { 0.0 };
    if correct.iter().all(|&c| c == 0.0) {
        return 0.0;
    }
    let mut prec = vec![0.0f64; order];
    let mut mteval = 1.0;
    let mut eff = order;
    for n in 1..=order {
        if let Smooth::AddK(k) = smooth {
            if n > 1 {
                correct[n - 1] += k;
                total[n - 1] += k;
            }
        }
        if total[n - 1] == 0.0 {
            break;
        }
        if effective {
            eff = n;
        }
        if correct[n - 1] == 0.0 {
            match smooth {
                Smooth::Exp => {
                    mteval *= 2.0;
                    prec[n - 1] = 100.0 / (mteval * total[n - 1]);
                }
                Smooth::Floor(v) => prec[n - 1] = 100.0 * v / total[n - 1],
                _ => {}
            }
        } else {
            prec[n - 1] = 100.0 * correct[n - 1] / total[n - 1];
        }
    }
    let logs: f64 = prec[..eff].iter().map(|&p| if p > 0.0 { p.ln() } else { -9_999_999_999.0 }).sum();
    bp * (logs / eff as f64).exp()
}

fn segment_stats(hyps: &[Vec<String>], refs: &[Vec<Vec<String>>], order: usize) -> Vec<Vec<f64>> {
    use rayon::prelude::*;
    let mut v = Vocab::default();
    let h: Vec<Vec<u32>> = hyps.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
    let r: Vec<Vec<Vec<u32>>> = refs.iter().map(|rs| rs.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect()).collect();
    h.par_iter()
        .zip(r.par_iter())
        .map(|(h, rs)| {
            let rr: Vec<&[u32]> = rs.iter().map(Vec::as_slice).collect();
            evalsuite_core::text::bleu_stats(h, &rr, order).into_iter().map(|x| x as f64).collect()
        })
        .collect()
}

/// Corpus BLEU (0–100), statistics summed over the corpus as sacreBLEU.
pub fn bleu<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], o: &BleuOpts<'_>) -> Result<f64> {
    check(references, predictions)?;
    if o.max_order == 0 {
        return invalid("max_order must be a positive integer.");
    }
    let h: Vec<Vec<String>> = predictions.iter().map(|p| prep(p.as_ref(), o.lowercase, &o.tokenize)).collect();
    let r: Vec<Vec<Vec<String>>> = references.iter().map(|rs| rs.iter().map(|x| prep(x.as_ref(), o.lowercase, &o.tokenize)).collect()).collect();
    let mut acc = vec![0.0f64; 2 + 2 * o.max_order];
    for seg in segment_stats(&h, &r, o.max_order) {
        for (a, v) in acc.iter_mut().zip(seg) {
            *a += v;
        }
    }
    Ok(bleu_score(&acc, o.max_order, o.smooth, o.effective_order))
}

/// Sentence BLEU per prediction (effective order, as sacreBLEU's `sentence_score`).
pub fn sentence_bleu<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], o: &BleuOpts<'_>) -> Result<Vec<f64>> {
    check(references, predictions)?;
    let h: Vec<Vec<String>> = predictions.iter().map(|p| prep(p.as_ref(), o.lowercase, &o.tokenize)).collect();
    let r: Vec<Vec<Vec<String>>> = references.iter().map(|rs| rs.iter().map(|x| prep(x.as_ref(), o.lowercase, &o.tokenize)).collect()).collect();
    Ok(segment_stats(&h, &r, o.max_order).iter().map(|s| bleu_score(s, o.max_order, o.smooth, true)).collect())
}

/// Self-BLEU: mean BLEU of each prediction against all the others (lower means more diverse).
pub fn self_bleu<P: AsRef<str>>(predictions: &[P], max_order: usize, tokenize: Tokenize<'_>) -> Result<f64> {
    if predictions.len() < 2 {
        return invalid("Self-BLEU needs at least two predictions.");
    }
    let t: Vec<Vec<String>> = predictions.iter().map(|p| tokenize.run(py_rstrip(p.as_ref()))).collect();
    let refs: Vec<Vec<Vec<String>>> = (0..t.len()).map(|i| t.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, x)| x.clone()).collect()).collect();
    let scores: Vec<f64> = segment_stats(&t, &refs, max_order).iter().map(|s| bleu_score(s, max_order, Smooth::Exp, true)).collect();
    Ok(np_sum(&scores) / scores.len() as f64)
}

const PUNCTS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

fn chrf_words(sent: &str) -> Vec<String> {
    let mut out = Vec::new();
    for w in py_split(sent) {
        let cs: Vec<char> = w.chars().collect();
        if cs.len() == 1 {
            out.push(w.to_owned());
        } else if PUNCTS.contains(cs[cs.len() - 1]) {
            out.push(cs[..cs.len() - 1].iter().collect());
            out.push(cs[cs.len() - 1].to_string());
        } else if PUNCTS.contains(cs[0]) {
            out.push(cs[0].to_string());
            out.push(cs[1..].iter().collect());
        } else {
            out.push(w.to_owned());
        }
    }
    out
}

/// chrF options (defaults: chrF2 with character 6-grams; `word_order = 2` gives chrF++).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChrfOpts {
    pub char_order: usize,
    pub word_order: usize,
    pub beta: f64,
    pub lowercase: bool,
    pub whitespace: bool,
    pub eps_smoothing: bool,
}

impl Default for ChrfOpts {
    fn default() -> Self {
        ChrfOpts { char_order: 6, word_order: 0, beta: 2.0, lowercase: false, whitespace: false, eps_smoothing: false }
    }
}

/// Corpus chrF (0–100): statistics summed over the corpus, each example using its best reference.
pub fn chrf<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], o: &ChrfOpts) -> Result<f64> {
    check(references, predictions)?;
    if o.char_order == 0 {
        return invalid("char_order must be a positive integer.");
    }
    let order = o.char_order + o.word_order;
    let words_all: Vec<Vec<String>> = predictions
        .iter()
        .map(|p| p.as_ref())
        .chain(references.iter().flatten().map(|r| r.as_ref()))
        .map(|s| if o.word_order > 0 { chrf_words(&if o.lowercase { py_lower(s) } else { s.to_owned() }) } else { vec![] })
        .collect();
    let mut v = Vocab::default();
    let ids: Vec<Vec<u32>> = words_all.iter().map(|w| v.encode(&w.iter().map(String::as_str).collect::<Vec<_>>())).collect();
    let chars_of = |s: &str| -> Vec<u32> {
        let s = if o.lowercase { py_lower(s) } else { s.to_owned() };
        let line: String = if o.whitespace { s } else { py_split(&s).concat() };
        line.chars().map(u32::from).collect()
    };
    let mut total = vec![0.0f64; 3 * order];
    let mut k = predictions.len();
    for (i, (p, rs)) in predictions.iter().zip(references).enumerate() {
        let hc = chars_of(p.as_ref());
        let h = evalsuite_core::text::ChrfText { chars: &hc, words: &ids[i] };
        let rcs: Vec<Vec<u32>> = rs.iter().map(|r| chars_of(r.as_ref())).collect();
        let rws: Vec<usize> = (k..k + rs.len()).collect();
        k += rs.len();
        let rt: Vec<evalsuite_core::text::ChrfText<'_>> = rcs.iter().zip(&rws).map(|(c, &w)| evalsuite_core::text::ChrfText { chars: c, words: &ids[w] }).collect();
        if let Some(best) = evalsuite_core::text::chrf_segment(&h, &rt, o.char_order, o.word_order, o.beta, o.eps_smoothing) {
            for (t, b) in total.iter_mut().zip(best) {
                *t += b;
            }
        }
    }
    Ok(evalsuite_core::text::chrf_f(&total, order, o.beta, o.eps_smoothing))
}

/// Translation edit rate (0–100+), as sacreBLEU's TER (Tercom shifts).
pub fn ter<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], case_sensitive: bool) -> Result<f64> {
    check(references, predictions)?;
    let tok = |s: &str| -> Vec<String> {
        let s = py_rstrip(s);
        let s = if case_sensitive { s.to_owned() } else { py_lower(s) };
        py_split(&s).into_iter().map(str::to_owned).collect()
    };
    let h: Vec<Vec<String>> = predictions.iter().map(|p| tok(p.as_ref())).collect();
    let r: Vec<Vec<Vec<String>>> = references.iter().map(|rs| rs.iter().map(|x| tok(x.as_ref())).collect()).collect();
    let mut v = Vocab::default();
    let hi: Vec<Vec<u32>> = h.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
    let ri: Vec<Vec<Vec<u32>>> = r.iter().map(|rs| rs.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect()).collect();
    let (mut edits, mut ref_len) = (0.0f64, 0.0f64);
    for (h, rs) in hi.iter().zip(&ri) {
        edits += rs.iter().map(|r| evalsuite_core::ter::ter_edits(h, r)).min().unwrap() as f64;
        ref_len += rs.iter().map(Vec::len).sum::<usize>() as f64 / rs.len() as f64;
    }
    Ok(if ref_len > 0.0 { 100.0 * edits / ref_len } else if edits > 0.0 { 100.0 } else { 0.0 })
}

/// Which ROUGE value to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Measure {
    Precision,
    Recall,
    #[default]
    Fmeasure,
}

/// ROUGE variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rouge {
    N(usize),
    L,
    /// Summary-level LCS over sentences (one per line).
    Lsum,
}

/// A word stemmer (for example a Porter stemmer from the `rust-stemmers` crate).
pub type Stemmer<'a> = &'a (dyn Fn(&str) -> String + Sync);

fn rouge_tok(text: &str, stemmer: Option<Stemmer<'_>>) -> Vec<String> {
    let low = py_lower(text);
    let mut cleaned = String::with_capacity(low.len());
    let mut prev_space = false;
    for c in low.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            cleaned.push(c);
            prev_space = false;
        } else if !prev_space {
            cleaned.push(' ');
            prev_space = true;
        }
    }
    cleaned
        .split(' ')
        .filter(|t| !t.is_empty())
        .map(|t| match stemmer {
            Some(s) if t.chars().count() > 3 => s(t),
            _ => t.to_owned(),
        })
        .filter(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()))
        .collect()
}

fn f(p: f64, r: f64) -> f64 {
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// ROUGE per prediction (best reference by F-measure, first on ties), as Google's rouge-score.
pub fn rouge<R: AsRef<str>, P: AsRef<str>>(
    references: &[Vec<R>],
    predictions: &[P],
    variant: Rouge,
    measure: Measure,
    stemmer: Option<Stemmer<'_>>,
) -> Result<Vec<f64>> {
    check(references, predictions)?;
    let k = match measure {
        Measure::Precision => 0,
        Measure::Recall => 1,
        Measure::Fmeasure => 2,
    };
    let mut out = Vec::with_capacity(predictions.len());
    for (p, rs) in predictions.iter().zip(references) {
        let mut v = Vocab::default();
        let triples: Vec<[f64; 3]> = match variant {
            Rouge::Lsum => {
                let sents = |s: &str| -> Vec<Vec<String>> { s.split('\n').filter(|x| !x.is_empty()).map(|x| rouge_tok(x, stemmer)).collect() };
                let ps = sents(p.as_ref());
                let rsents: Vec<Vec<Vec<String>>> = rs.iter().map(|r| sents(r.as_ref())).collect();
                let pi: Vec<Vec<u32>> = ps.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
                let ri: Vec<Vec<Vec<u32>>> = rsents.iter().map(|ss| ss.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect()).collect();
                ri.iter()
                    .map(|ts| {
                        let tr: Vec<&[u32]> = ts.iter().map(Vec::as_slice).collect();
                        let pr: Vec<&[u32]> = pi.iter().map(Vec::as_slice).collect();
                        let (hits, m, n) = evalsuite_core::text::rouge_lsum_hits(&tr, &pr);
                        if ts.is_empty() || pi.is_empty() || m == 0 || n == 0 {
                            [0.0; 3]
                        } else {
                            let (pp, rr) = (hits as f64 / n as f64, hits as f64 / m as f64);
                            [pp, rr, f(pp, rr)]
                        }
                    })
                    .collect()
            }
            _ => {
                let pt = rouge_tok(p.as_ref(), stemmer);
                let rts: Vec<Vec<String>> = rs.iter().map(|r| rouge_tok(r.as_ref(), stemmer)).collect();
                let pi = v.encode(&pt.iter().map(String::as_str).collect::<Vec<_>>());
                let ri: Vec<Vec<u32>> = rts.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
                ri.iter()
                    .map(|t| match variant {
                        Rouge::L => {
                            if t.is_empty() || pi.is_empty() {
                                [0.0; 3]
                            } else {
                                let l = evalsuite_core::text::lcs_length(t, &pi) as f64;
                                let (pp, rr) = (l / pi.len() as f64, l / t.len() as f64);
                                [pp, rr, f(pp, rr)]
                            }
                        }
                        Rouge::N(n) => {
                            let inter = evalsuite_core::text::ngram_overlap(t, &pi, n) as f64;
                            let pp = inter / (pi.len() as f64 - n as f64 + 1.0).max(1.0);
                            let rr = inter / (t.len() as f64 - n as f64 + 1.0).max(1.0);
                            [pp, rr, f(pp, rr)]
                        }
                        Rouge::Lsum => unreachable!(),
                    })
                    .collect()
            }
        };
        let mut best = 0;
        for (i, t) in triples.iter().enumerate() {
            if t[2] > triples[best][2] {
                best = i;
            }
        }
        out.push(triples[best][k]);
    }
    Ok(out)
}

/// METEOR options (NLTK defaults).
#[derive(Clone, Copy)]
pub struct MeteorOpts<'a> {
    /// Stemmer for the stem-matching stage (NLTK uses Porter); `None` skips the stage.
    pub stemmer: Option<Stemmer<'a>>,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub tokenize: Tokenize<'a>,
}

impl std::fmt::Debug for MeteorOpts<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MeteorOpts {{ stemmer: {}, alpha: {}, beta: {}, gamma: {} }}", self.stemmer.is_some(), self.alpha, self.beta, self.gamma)
    }
}

impl Default for MeteorOpts<'_> {
    fn default() -> Self {
        MeteorOpts { stemmer: None, alpha: 0.9, beta: 3.0, gamma: 0.5, tokenize: Tokenize::None }
    }
}

/// METEOR per prediction (exact and, with a stemmer, stem matches; best reference).
pub fn meteor<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], o: &MeteorOpts<'_>) -> Result<Vec<f64>> {
    check(references, predictions)?;
    for (name, v) in [("alpha", o.alpha), ("gamma", o.gamma)] {
        if !(0.0..=1.0).contains(&v) {
            return invalid(format!("{name} must be in [0, 1]."));
        }
    }
    let low = |s: &str| -> Vec<String> { o.tokenize.run(s).iter().map(|w| py_lower(w)).collect() };
    let mut out = Vec::with_capacity(predictions.len());
    for (p, rs) in predictions.iter().zip(references) {
        let h = low(p.as_ref());
        let refs: Vec<Vec<String>> = rs.iter().map(|r| low(r.as_ref())).collect();
        let mut v = Vocab::default();
        let hi = v.encode(&h.iter().map(String::as_str).collect::<Vec<_>>());
        let ri: Vec<Vec<u32>> = refs.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
        let stems = o.stemmer.map(|st| {
            let all: Vec<String> = h.iter().chain(refs.iter().flatten()).map(|w| st(w)).collect();
            let mut sv: HashMap<String, u32> = HashMap::new();
            let ids: Vec<u32> = all.iter().map(|s| {
                let n = sv.len() as u32;
                *sv.entry(s.clone()).or_insert(n)
            }).collect();
            ids
        });
        let mut best = f64::NEG_INFINITY;
        let mut pos = h.len();
        for (j, r) in ri.iter().enumerate() {
            let s = match &stems {
                Some(ids) => {
                    let hs = &ids[..h.len()];
                    let rs_ = &ids[pos..pos + refs[j].len()];
                    Some((hs, rs_))
                }
                None => None,
            };
            pos += refs[j].len();
            best = best.max(evalsuite_core::meteor::meteor_single(&hi, r, s, o.alpha, o.beta, o.gamma));
        }
        out.push(best);
    }
    Ok(out)
}

/// CIDEr-D per prediction (document frequencies from all references; pycocoevalcap's formula).
pub fn cider<R: AsRef<str>, P: AsRef<str>>(references: &[Vec<R>], predictions: &[P], max_order: usize, sigma: f64, lowercase: bool, tokenize: Tokenize<'_>) -> Result<Vec<f64>> {
    check(references, predictions)?;
    if max_order == 0 {
        return invalid("max_order must be a positive integer.");
    }
    if !(sigma > 0.0) {
        return invalid("sigma must be positive.");
    }
    let words = |s: &str| tokenize.run(&if lowercase { py_lower(s) } else { s.to_owned() });
    let h: Vec<Vec<String>> = predictions.iter().map(|p| words(p.as_ref())).collect();
    let r: Vec<Vec<Vec<String>>> = references.iter().map(|rs| rs.iter().map(|x| words(x.as_ref())).collect()).collect();
    let mut v = Vocab::default();
    let hi: Vec<Vec<u32>> = h.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
    let ri: Vec<Vec<Vec<u32>>> = r.iter().map(|rs| rs.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect()).collect();
    let hs: Vec<&[u32]> = hi.iter().map(Vec::as_slice).collect();
    let rs: Vec<Vec<&[u32]>> = ri.iter().map(|x| x.iter().map(Vec::as_slice).collect()).collect();
    // Compensated summation, as Python's built-in sum() on 3.12+ (and more accurate).
    let totals = evalsuite_core::text::cider_totals(&hs, &rs, max_order, sigma, true);
    Ok(totals.iter().zip(references).map(|(t, rs)| np_sum(t) / t.len() as f64 / rs.len() as f64 * 10.0).collect())
}

/// Distinct-n: unique n-grams over all n-grams in the predictions (whitespace tokens).
pub fn distinct_n<P: AsRef<str>>(predictions: &[P], n: usize, lowercase: bool) -> Result<f64> {
    if predictions.is_empty() {
        return invalid("predictions is empty. Provide at least one example.");
    }
    if n == 0 {
        return invalid("n must be a positive integer.");
    }
    let toks: Vec<Vec<String>> = predictions
        .iter()
        .map(|p| {
            let s = if lowercase { py_lower(p.as_ref()) } else { p.as_ref().to_owned() };
            py_split(&s).into_iter().map(str::to_owned).collect()
        })
        .collect();
    let mut v = Vocab::default();
    let ids: Vec<Vec<u32>> = toks.iter().map(|x| v.encode(&x.iter().map(String::as_str).collect::<Vec<_>>())).collect();
    let seqs: Vec<&[u32]> = ids.iter().map(Vec::as_slice).collect();
    let (unique, total) = evalsuite_core::text::distinct_ngrams(&seqs, n);
    if total == 0 {
        return invalid(format!("The predictions contain no {n}-grams."));
    }
    Ok(unique as f64 / total as f64)
}

fn logprobs(seqs: &[Vec<f64>]) -> Result<()> {
    if seqs.is_empty() {
        return invalid("token_logprobs must be a non-empty list of per-token log-probabilities.");
    }
    for (i, s) in seqs.iter().enumerate() {
        if s.is_empty() {
            return invalid(format!("token_logprobs[{i}] is empty."));
        }
        if s.iter().any(|&v| !v.is_finite() || v > 1e-9) {
            return invalid(format!("token_logprobs[{i}] must contain finite natural-log probabilities (<= 0)."));
        }
    }
    Ok(())
}

/// Cross-entropy per token (natural-log probabilities in; result in units of `base`).
pub fn cross_entropy(token_logprobs: &[Vec<f64>], base: f64) -> Result<f64> {
    logprobs(token_logprobs)?;
    if !(base > 0.0) || base == 1.0 {
        return invalid("base must be positive and not 1.");
    }
    let flat: Vec<f64> = token_logprobs.concat();
    Ok(-np_sum(&flat) / flat.len() as f64 / base.ln())
}

/// Perplexity: pooled over all tokens (`per_sequence = false`) or the mean of per-sequence perplexities.
pub fn perplexity(token_logprobs: &[Vec<f64>], per_sequence: bool) -> Result<f64> {
    logprobs(token_logprobs)?;
    if per_sequence {
        let p: Vec<f64> = token_logprobs.iter().map(|s| (-(np_sum(s) / s.len() as f64)).exp()).collect();
        return Ok(np_sum(&p) / p.len() as f64);
    }
    let flat: Vec<f64> = token_logprobs.concat();
    Ok((-(np_sum(&flat) / flat.len() as f64)).exp())
}
