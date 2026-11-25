//! SPICE: the F-score between the semantic tuples (objects `(o)`, attributes `(o, a)`, relations
//! `(s, r, o)`) of a candidate caption's and its references' scene graphs, as EvalSuite's Python
//! `evalsuite.text.spice`. The tuples come from your scene-graph parser; matching is exact or via a synonym
//! function (SPICE uses WordNet synsets — pass a function giving a word's synonym keys).
//!
//! ```
//! use evalsuite::spice::spice;
//! let refs = vec![vec![vec!["dog"], vec!["dog", "brown"], vec!["dog", "on", "grass"]]];
//! let cand = vec![vec![vec!["dog"], vec!["dog", "brown"]]];
//! let s = spice(&refs, &cand, None).unwrap();
//! assert!((s.per_image[0] - 0.8).abs() < 1e-12);
//! ```

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::error::{invalid, Result};
use crate::util::mean;

type Tuple = Vec<String>;

/// A synonym function: a word → its synonym keys (two words match when they share a key).
pub type Synonyms<'a> = &'a dyn Fn(&str) -> Vec<String>;

fn tuples<S: AsRef<str>>(x: &[Vec<S>], name: &str) -> Result<BTreeSet<Tuple>> {
    let mut out = BTreeSet::new();
    for t in x {
        if !(1..=3).contains(&t.len()) || t.iter().any(|e| e.as_ref().trim().is_empty()) {
            return invalid(format!("{name}: every tuple needs 1-3 non-empty strings."));
        }
        out.insert(t.iter().map(|e| e.as_ref().trim().to_lowercase()).collect());
    }
    Ok(out)
}

fn match_count(
    cand: &BTreeSet<Tuple>,
    reference: &BTreeSet<Tuple>,
    syn: Option<Synonyms>,
    cache: &mut HashMap<String, HashSet<String>>,
) -> usize {
    let Some(syn) = syn else {
        return cand.intersection(reference).count();
    };
    let mut keys = |w: &str| -> HashSet<String> {
        cache
            .entry(w.to_owned())
            .or_insert_with(|| {
                let mut k: HashSet<String> = syn(w).into_iter().collect();
                k.insert(w.to_owned());
                k
            })
            .clone()
    };
    let mut same = |a: &Tuple, b: &Tuple| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x == y || !keys(x).is_disjoint(&keys(y)))
    };
    let mut pool: Vec<&Tuple> = reference.iter().collect();
    let mut order: Vec<&Tuple> = cand.iter().collect();
    order.sort_by_key(|t| !reference.contains(*t)); // exact matches first, each group in tuple order
    let mut n = 0;
    for c in order {
        if let Some(i) = pool.iter().position(|r| same(c, r)) {
            n += 1;
            pool.remove(i);
        }
    }
    n
}

/// SPICE scores: mean F-score, per image, precision and recall per image, and F-scores per tuple type
/// (object, attribute, relation; pooled over images).
#[derive(Debug, Clone, PartialEq)]
pub struct Spice {
    pub value: f64,
    pub per_image: Vec<f64>,
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub object_f: f64,
    pub attribute_f: f64,
    pub relation_f: f64,
}

fn f(p: f64, r: f64) -> f64 {
    if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    }
}

/// SPICE from tuples: `references[i]` holds the tuples of all references of image i (merged),
/// `candidates[i]` the candidate's tuples.
pub fn spice<S: AsRef<str>>(
    references: &[Vec<Vec<S>>],
    candidates: &[Vec<Vec<S>>],
    synonyms: Option<Synonyms>,
) -> Result<Spice> {
    if references.is_empty() || references.len() != candidates.len() {
        return invalid("references and candidates must be non-empty and the same length.");
    }
    let mut cache = HashMap::new();
    let (mut f1s, mut ps, mut rs) = (vec![], vec![], vec![]);
    let mut by_type = [[0usize; 3]; 3];
    for (i, (r, c)) in references.iter().zip(candidates).enumerate() {
        let rt = tuples(r, &format!("references[{i}]"))?;
        let ct = tuples(c, &format!("candidates[{i}]"))?;
        let m = match_count(&ct, &rt, synonyms, &mut cache) as f64;
        let p = if ct.is_empty() {
            0.0
        } else {
            m / ct.len() as f64
        };
        let rc = if rt.is_empty() {
            0.0
        } else {
            m / rt.len() as f64
        };
        ps.push(p);
        rs.push(rc);
        f1s.push(f(p, rc));
        for k in 1..=3 {
            let ck: BTreeSet<Tuple> = ct.iter().filter(|t| t.len() == k).cloned().collect();
            let rk: BTreeSet<Tuple> = rt.iter().filter(|t| t.len() == k).cloned().collect();
            by_type[k - 1][0] += match_count(&ck, &rk, synonyms, &mut cache);
            by_type[k - 1][1] += ck.len();
            by_type[k - 1][2] += rk.len();
        }
    }
    let tf = |k: usize| {
        let [mt, nc, nr] = by_type[k];
        let p = if nc > 0 { mt as f64 / nc as f64 } else { 0.0 };
        let r = if nr > 0 { mt as f64 / nr as f64 } else { 0.0 };
        f(p, r)
    };
    Ok(Spice {
        value: mean(&f1s),
        per_image: f1s,
        precision: ps,
        recall: rs,
        object_f: tf(0),
        attribute_f: tf(1),
        relation_f: tf(2),
    })
}
