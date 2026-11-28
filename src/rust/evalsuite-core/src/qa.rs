//! SQuAD answer normalisation, exact match and token F1, as EvalSuite's Python reference (`text/qa.py`), for
//! ASCII text (callers use Python for anything else, whose Unicode case mapping and word boundaries differ).

use crate::hash::FxMap;

/// Python's `str.isspace()` for ASCII (it includes the four information separators 0x1c..0x1f).
#[inline]
fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | 0x1c..=0x1f)
}

/// `str.split()` (no argument) for ASCII text.
pub fn split_ws(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| c.is_ascii() && is_space(c as u8))
        .filter(|t| !t.is_empty())
}

/// `normalize_answer`: lowercase, drop `string.punctuation`, replace the words a/an/the by a space (with `\b`
/// boundaries on ASCII word characters), collapse whitespace. `None` for non-ASCII input.
pub fn normalize(text: &str) -> Option<String> {
    if !text.is_ascii() {
        return None;
    }
    let cleaned: Vec<u8> = text
        .bytes()
        .map(|b| b.to_ascii_lowercase())
        .filter(|b| !b.is_ascii_punctuation())
        .collect();
    // Replace whole-word articles: a word is a maximal run of [a-z0-9] (underscore is punctuation, removed).
    let mut out = Vec::with_capacity(cleaned.len());
    let mut i = 0;
    while i < cleaned.len() {
        if cleaned[i].is_ascii_alphanumeric() {
            let mut j = i;
            while j < cleaned.len() && cleaned[j].is_ascii_alphanumeric() {
                j += 1;
            }
            let w = &cleaned[i..j];
            if w == b"a" || w == b"an" || w == b"the" {
                out.push(b' ');
            } else {
                out.extend_from_slice(w);
            }
            i = j;
        } else {
            out.push(cleaned[i]);
            i += 1;
        }
    }
    let s = String::from_utf8(out).ok()?;
    Some(split_ws(&s).collect::<Vec<_>>().join(" "))
}

/// Token F1 of two normalised strings (bag of words).
pub fn f1(pred: &str, reference: &str) -> f64 {
    let p: Vec<&str> = split_ws(pred).collect();
    let r: Vec<&str> = split_ws(reference).collect();
    if p.is_empty() || r.is_empty() {
        return f64::from(u8::from(p == r));
    }
    let mut counts: FxMap<&str, i64> = FxMap::default();
    for t in &r {
        *counts.entry(t).or_insert(0) += 1;
    }
    let mut same = 0i64;
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
    let precision = same as f64 / p.len() as f64;
    let recall = same as f64 / r.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squad() {
        assert_eq!(
            normalize("The  Cat, an apple!").as_deref(),
            Some("cat apple")
        );
        assert_eq!(normalize("a-b theory").as_deref(), Some("ab theory"));
        assert_eq!(normalize("caf\u{e9}"), None);
        assert_eq!(f1("cat sat", "the cat"), 0.5);
        assert_eq!(f1("", ""), 1.0);
    }
}
