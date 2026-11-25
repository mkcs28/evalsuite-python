//! LLM-as-a-judge and human-evaluation statistics: win rates, Bradley–Terry and Elo ratings, inter-rater
//! agreement (Krippendorff's alpha, Fleiss' kappa, judge–human agreement), judge bias checks (position,
//! verbosity, self-preference) and rubric scores, as EvalSuite's Python `evalsuite.text.judge`.
//!
//! Outcomes are numbers for the first model: 1 = win, 0.5 = tie, 0 = loss ([`outcome`] parses the usual
//! strings).
//!
//! ```
//! use evalsuite::judge::{win_rate, Ties};
//! let r = win_rate(&[1.0, 0.5, 0.0, 1.0], Ties::Half).unwrap();
//! assert_eq!(r.value, 0.625);
//! ```

use std::collections::HashMap;
use std::hash::Hash;

use crate::classification::{cohen_kappa, KappaWeights, ZeroDivision};
use crate::error::{invalid, Result};
use crate::special::binom_cdf;
use crate::util::{mean, np_sum};
use crate::Label;

/// Parses an outcome: `win`/`w`/`a`/`1` → 1, `loss`/`lose`/`l`/`b`/`0` → 0, `tie`/`draw`/`t`/`0.5` → 0.5.
pub fn outcome(s: &str) -> Result<f64> {
    match s.trim().to_lowercase().as_str() {
        "win" | "w" | "a" | "1" => Ok(1.0),
        "loss" | "lose" | "l" | "b" | "0" => Ok(0.0),
        "tie" | "draw" | "t" | "0.5" => Ok(0.5),
        _ => invalid(format!("Outcome must be 'win', 'loss', 'tie' (or 1, 0, 0.5); got {s:?}.")),
    }
}

fn check_outcome(o: f64) -> Result<f64> {
    if o == 0.0 || o == 0.5 || o == 1.0 {
        Ok(o)
    } else {
        invalid(format!("Outcome must be 'win', 'loss', 'tie' (or 1, 0, 0.5); got {o}."))
    }
}

fn wilson(k: f64, n: f64) -> (f64, f64) {
    const Z: f64 = 1.959963984540054;
    if n == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let p = k / n;
    let centre = (p + Z * Z / (2.0 * n)) / (1.0 + Z * Z / n);
    let half = Z * (p * (1.0 - p) / n + Z * Z / (4.0 * n * n)).sqrt() / (1.0 + Z * Z / n);
    (centre - half, centre + half)
}

/// How ties count in [`win_rate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Ties {
    /// A tie is half a win.
    #[default]
    Half,
    /// Ties are dropped.
    Exclude,
}

/// A win rate with its 95% Wilson interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WinRate {
    pub value: f64,
    pub n: usize,
    pub n_ties: usize,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Share of pairwise comparisons won.
pub fn win_rate(outcomes: &[f64], ties: Ties) -> Result<WinRate> {
    let mut v = outcomes.iter().map(|&o| check_outcome(o)).collect::<Result<Vec<f64>>>()?;
    if v.is_empty() {
        return invalid("outcomes is empty.");
    }
    if ties == Ties::Exclude {
        v.retain(|&x| x != 0.5);
        if v.is_empty() {
            return invalid("Every comparison is a tie; the win rate excluding ties is undefined.");
        }
    }
    let wins = np_sum(&v);
    let (ci_low, ci_high) = wilson(wins, v.len() as f64);
    Ok(WinRate { value: wins / v.len() as f64, n: v.len(), n_ties: v.iter().filter(|&&x| x == 0.5).count(), ci_low, ci_high })
}

/// One rating per model, in first-seen order.
#[derive(Debug, Clone, PartialEq)]
pub struct Ratings<M> {
    pub models: Vec<M>,
    pub values: Vec<f64>,
}

fn comparisons<M: Eq + Hash + Clone + std::fmt::Debug>(c: &[(M, M, f64)]) -> Result<(Vec<M>, Vec<(usize, usize, f64)>)> {
    if c.is_empty() {
        return invalid("comparisons is empty.");
    }
    let mut ids: HashMap<&M, usize> = HashMap::new();
    let mut names = Vec::new();
    let mut rows = Vec::with_capacity(c.len());
    for (i, (a, b, o)) in c.iter().enumerate() {
        if a == b {
            return invalid(format!("comparisons[{i}] compares {a:?} with itself."));
        }
        let mut ix = [0usize; 2];
        for (slot, m) in [a, b].into_iter().enumerate() {
            let n = ids.len();
            ix[slot] = *ids.entry(m).or_insert_with(|| {
                names.push(m.clone());
                n
            });
        }
        let (ia, ib) = (ix[0], ix[1]);
        rows.push((ia, ib, check_outcome(*o)?));
    }
    Ok((names, rows))
}

/// Scale of Bradley–Terry strengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BtScale {
    /// Natural-log strengths with mean 0 (as `choix`).
    #[default]
    Log,
    /// 1000 + 400·log10(strength).
    Elo,
}

/// Bradley–Terry maximum-likelihood strengths (MM algorithm, Hunter 2004) from `(model_a, model_b, outcome
/// for a)` comparisons. The comparison graph must be connected and no model may win or lose every comparison.
pub fn bradley_terry<M: Eq + Hash + Clone + std::fmt::Debug>(comparisons_: &[(M, M, f64)], scale: BtScale, max_iter: usize, tol: f64) -> Result<Ratings<M>> {
    let (names, rows) = comparisons(comparisons_)?;
    let m = names.len();
    let mut wins = vec![0.0; m * m];
    for &(a, b, o) in &rows {
        wins[a * m + b] += o;
        wins[b * m + a] += 1.0 - o;
    }
    let games: Vec<f64> = (0..m * m).map(|i| wins[i] + wins[(i % m) * m + i / m]).collect();
    let total_w: Vec<f64> = (0..m).map(|i| np_sum(&wins[i * m..(i + 1) * m])).collect();
    let total_g: Vec<f64> = (0..m).map(|i| np_sum(&games[i * m..(i + 1) * m])).collect();
    if total_w.iter().zip(&total_g).any(|(&w, &g)| w == 0.0 || w == g) {
        return invalid("Every model must win and lose at least one comparison (else its strength is infinite).");
    }
    let mut seen = vec![false; m];
    seen[0] = true;
    let mut frontier = vec![0];
    while let Some(i) = frontier.pop() {
        for j in 0..m {
            if games[i * m + j] > 0.0 && !seen[j] {
                seen[j] = true;
                frontier.push(j);
            }
        }
    }
    if seen.iter().any(|s| !s) {
        return invalid("The comparison graph is not connected; strengths are not comparable across components.");
    }
    let mut p = vec![1.0; m];
    for _ in 0..max_iter {
        let mut new: Vec<f64> = (0..m)
            .map(|i| {
                let row: Vec<f64> = (0..m).map(|j| games[i * m + j] / (p[i] + p[j])).collect();
                total_w[i] / np_sum(&row)
            })
            .collect();
        let logs: Vec<f64> = new.iter().map(|x| x.ln()).collect();
        let g = (np_sum(&logs) / m as f64).exp();
        new.iter_mut().for_each(|x| *x /= g);
        let diff = new.iter().zip(&p).map(|(a, b)| (a.ln() - b.ln()).abs()).fold(0.0, f64::max);
        p = new;
        if diff < tol {
            break;
        }
    }
    let logs: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let mu = np_sum(&logs) / m as f64;
    let values = logs
        .iter()
        .map(|l| {
            let v = l - mu;
            match scale {
                BtScale::Log => v,
                BtScale::Elo => 1000.0 + 400.0 * v / std::f64::consts::LN_10,
            }
        })
        .collect();
    Ok(Ratings { models: names, values })
}

/// Sequential Elo ratings (order-dependent): each comparison moves both ratings by `k·(outcome − expected)`.
pub fn elo_ratings<M: Eq + Hash + Clone + std::fmt::Debug>(comparisons_: &[(M, M, f64)], k: f64, initial: f64, scale: f64, base: f64) -> Result<Ratings<M>> {
    let (names, rows) = comparisons(comparisons_)?;
    if !(k > 0.0 && scale > 0.0 && base > 1.0) {
        return invalid("k and scale must be positive and base > 1.");
    }
    let mut r = vec![initial; names.len()];
    for (a, b, o) in rows {
        let ea = 1.0 / (1.0 + base.powf((r[b] - r[a]) / scale));
        r[a] += k * (o - ea);
        r[b] += k * ((1.0 - o) - (1.0 - ea));
    }
    Ok(Ratings { models: names, values: r })
}

/// Measurement level for Krippendorff's alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Level {
    #[default]
    Nominal,
    Ordinal,
    Interval,
    Ratio,
}

/// Krippendorff's alpha from a raters × items matrix (NaN = missing rating).
pub fn krippendorff_alpha(ratings: &[Vec<f64>], level: Level) -> Result<f64> {
    let n_items = ratings.first().map_or(0, Vec::len);
    if ratings.len() < 2 || n_items < 1 || ratings.iter().any(|r| r.len() != n_items) {
        return invalid("ratings must be a 2-D array: one row per rater (>= 2), one column per item (NaN = missing).");
    }
    let mut values: Vec<f64> = ratings.iter().flatten().copied().filter(|x| !x.is_nan()).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.dedup();
    let v = values.len();
    if v < 2 {
        return invalid("Krippendorff's alpha needs at least two distinct rating values.");
    }
    let code = |x: f64| values.partition_point(|&y| y < x);
    let mut counts = vec![0.0; n_items * v];
    for r in ratings {
        for (u, &x) in r.iter().enumerate() {
            if !x.is_nan() {
                counts[u * v + code(x)] += 1.0;
            }
        }
    }
    let mut coinc = vec![0.0; v * v];
    for u in 0..n_items {
        let row = &counts[u * v..(u + 1) * v];
        let m_u: f64 = row.iter().sum();
        if m_u < 2.0 {
            continue;
        }
        for c in 0..v {
            let s = row[c] / (m_u - 1.0);
            for k in 0..v {
                coinc[c * v + k] += s * row[k];
            }
            coinc[c * v + c] -= s;
        }
    }
    let n_c: Vec<f64> = (0..v).map(|c| coinc[c * v..(c + 1) * v].iter().sum()).collect();
    let n: f64 = n_c.iter().sum();
    if n == 0.0 {
        return invalid("No item has ratings from two or more raters.");
    }
    let mut cum = n_c.clone();
    for i in 1..v {
        cum[i] += cum[i - 1];
    }
    let delta = |c: usize, k: usize| -> f64 {
        let (a, b) = (values[c], values[k]);
        match level {
            Level::Nominal => f64::from(u8::from(c != k)),
            Level::Interval => (a - b) * (a - b),
            Level::Ratio => {
                let s = a + b;
                if s == 0.0 {
                    0.0
                } else {
                    ((a - b) / s).powi(2)
                }
            }
            Level::Ordinal => {
                let (lo, hi) = (c.min(k), c.max(k));
                let x = cum[hi] - if lo > 0 { cum[lo - 1] } else { 0.0 } - (n_c[c] + n_c[k]) / 2.0;
                x * x
            }
        }
    };
    let (mut d_o, mut d_e) = (0.0, 0.0);
    for c in 0..v {
        for k in 0..v {
            let d = delta(c, k);
            d_o += coinc[c * v + k] * d;
            d_e += n_c[c] * n_c[k] * d;
        }
    }
    let (d_o, d_e) = (d_o / n, d_e / (n * (n - 1.0)));
    if d_e == 0.0 {
        return invalid("Expected disagreement is zero; alpha is undefined.");
    }
    Ok(1.0 - d_o / d_e)
}

/// Fleiss' kappa from an items × raters matrix of category labels (a fixed number of raters per item).
pub fn fleiss_kappa<T: Eq + Hash>(ratings: &[Vec<T>]) -> Result<f64> {
    let n_r = ratings.first().map_or(0, Vec::len);
    if ratings.is_empty() || n_r < 2 || ratings.iter().any(|r| r.len() != n_r) {
        return invalid("ratings must be a 2-D array: one row per item, one column per rater (>= 2).");
    }
    let mut ids: HashMap<&T, usize> = HashMap::new();
    for x in ratings.iter().flatten() {
        let n = ids.len();
        ids.entry(x).or_insert(n);
    }
    let k = ids.len();
    let mut table = vec![0.0; ratings.len() * k];
    for (i, r) in ratings.iter().enumerate() {
        for x in r {
            table[i * k + ids[x]] += 1.0;
        }
    }
    let total = np_sum(&table);
    let p_j: Vec<f64> = (0..k).map(|j| (0..ratings.len()).map(|i| table[i * k + j]).sum::<f64>() / total).collect();
    let nr = n_r as f64;
    let p_i: Vec<f64> = (0..ratings.len()).map(|i| (table[i * k..(i + 1) * k].iter().map(|x| x * x).sum::<f64>() - nr) / (nr * (nr - 1.0))).collect();
    let p_bar = mean(&p_i);
    let p_e: f64 = np_sum(&p_j.iter().map(|x| x * x).collect::<Vec<_>>());
    if p_e == 1.0 {
        return invalid("All ratings fall in one category; Fleiss' kappa is undefined.");
    }
    Ok((p_bar - p_e) / (1.0 - p_e))
}

/// Pearson correlation (as `scipy.stats.pearsonr`; NaN for a constant input).
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let xm: Vec<f64> = x.iter().map(|a| a - mx).collect();
    let ym: Vec<f64> = y.iter().map(|a| a - my).collect();
    let nx = xm.iter().map(|a| a * a).sum::<f64>().sqrt();
    let ny = ym.iter().map(|a| a * a).sum::<f64>().sqrt();
    if nx == 0.0 || ny == 0.0 {
        return f64::NAN;
    }
    let r: f64 = xm.iter().zip(&ym).map(|(a, b)| (a / nx) * (b / ny)).sum();
    r.clamp(-1.0, 1.0)
}

/// Spearman rank correlation (Pearson on mid-ranks).
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&evalsuite_core::ranks::midranks(x), &evalsuite_core::ranks::midranks(y))
}

/// Judge–human agreement: the headline statistic plus raw agreement and Spearman's rho where defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Agreement {
    /// Cohen's kappa (categorical), quadratic-weighted kappa (ordinal) or Pearson's r (continuous).
    pub value: f64,
    pub raw_agreement: Option<f64>,
    pub spearman: Option<f64>,
    pub n: usize,
}

fn pair_len(j: usize, h: usize) -> Result<()> {
    if j == 0 || j != h {
        return invalid("judge and human must be non-empty and the same length.");
    }
    Ok(())
}

fn need_two(n: usize) -> Result<()> {
    if n < 2 {
        return invalid("Correlations need at least two ratings.");
    }
    Ok(())
}

/// Categorical judge–human agreement (Cohen's kappa).
pub fn judge_agreement<L: Label>(judge: &[L], human: &[L]) -> Result<Agreement> {
    pair_len(judge.len(), human.len())?;
    let k = cohen_kappa(human, judge, KappaWeights::Unweighted, None, None, ZeroDivision::Warn)?;
    let raw = judge.iter().zip(human).filter(|(a, b)| a == b).count() as f64 / judge.len() as f64;
    Ok(Agreement { value: k, raw_agreement: Some(raw), spearman: None, n: judge.len() })
}

/// Ordinal judge–human agreement (quadratic-weighted kappa, with Spearman's rho).
pub fn judge_agreement_ordinal(judge: &[i64], human: &[i64]) -> Result<Agreement> {
    pair_len(judge.len(), human.len())?;
    need_two(judge.len())?;
    let k = cohen_kappa(human, judge, KappaWeights::Quadratic, None, None, ZeroDivision::Warn)?;
    let (jf, hf): (Vec<f64>, Vec<f64>) = (judge.iter().map(|&x| x as f64).collect(), human.iter().map(|&x| x as f64).collect());
    let raw = judge.iter().zip(human).filter(|(a, b)| a == b).count() as f64 / judge.len() as f64;
    Ok(Agreement { value: k, raw_agreement: Some(raw), spearman: Some(spearman(&jf, &hf)), n: judge.len() })
}

/// Continuous judge–human agreement (Pearson's r, with Spearman's rho).
pub fn judge_agreement_continuous(judge: &[f64], human: &[f64]) -> Result<Agreement> {
    pair_len(judge.len(), human.len())?;
    need_two(judge.len())?;
    Ok(Agreement { value: pearson(judge, human), raw_agreement: None, spearman: Some(spearman(judge, human)), n: judge.len() })
}

/// A pairwise verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verdict {
    A,
    B,
    Tie,
}

impl Verdict {
    /// Parses `a`/`1`/`first`, `b`/`2`/`second`, `tie`/`draw`/`t`/`0.5` (case-insensitive).
    pub fn parse(s: &str) -> Result<Verdict> {
        match s.trim().to_lowercase().as_str() {
            "a" | "1" | "first" => Ok(Verdict::A),
            "b" | "2" | "second" => Ok(Verdict::B),
            "tie" | "draw" | "t" | "0.5" => Ok(Verdict::Tie),
            _ => invalid(format!("Verdict must be 'A', 'B' or 'tie'; got {s:?}.")),
        }
    }
}

/// Position consistency with the share of inconsistent pairs favouring the first / second position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionConsistency {
    pub value: f64,
    pub n_inconsistent: usize,
    pub first_position_rate: f64,
    pub second_position_rate: f64,
}

/// Share of comparisons where the judge gives the same verdict with the responses swapped. `swapped[i]`
/// refers to the swapped order (A = the response shown first in the swapped prompt).
pub fn position_consistency(original: &[Verdict], swapped: &[Verdict]) -> Result<PositionConsistency> {
    if original.is_empty() || original.len() != swapped.len() {
        return invalid("verdicts_original and verdicts_swapped must be non-empty and the same length.");
    }
    let same = original.iter().zip(swapped).filter(|(a, b)| a == b).count();
    let inc = original.len() - same;
    let first = original.iter().zip(swapped).filter(|p| *p == (&Verdict::A, &Verdict::B)).count();
    let second = original.iter().zip(swapped).filter(|p| *p == (&Verdict::B, &Verdict::A)).count();
    let r = |x: usize| if inc > 0 { x as f64 / inc as f64 } else { f64::NAN };
    Ok(PositionConsistency { value: same as f64 / original.len() as f64, n_inconsistent: inc, first_position_rate: r(first), second_position_rate: r(second) })
}

fn binom_two_sided_half(k: usize, n: usize) -> f64 {
    // scipy.stats.binomtest(k, n, 0.5).pvalue: symmetric, so twice the smaller tail.
    if 2 * k == n {
        return 1.0;
    }
    (2.0 * binom_cdf(k.min(n - k) as f64, n as f64, 0.5)).min(1.0)
}

/// Longer-response win rate among decisive comparisons of responses of different length, with the two-sided
/// exact binomial p-value against 0.5.
pub fn verbosity_bias(winners: &[Verdict], length_a: &[f64], length_b: &[f64]) -> Result<(f64, usize, f64)> {
    if winners.is_empty() || length_a.len() != winners.len() || length_b.len() != winners.len() {
        return invalid("winners, length_a and length_b must be non-empty and the same length.");
    }
    let flags: Vec<bool> = winners
        .iter()
        .zip(length_a.iter().zip(length_b))
        .filter(|(w, (a, b))| **w != Verdict::Tie && a != b)
        .map(|(w, (a, b))| (*w == Verdict::A) == (a > b))
        .collect();
    if flags.is_empty() {
        return invalid("No decisive comparison between responses of different length.");
    }
    let k = flags.iter().filter(|&&f| f).count();
    Ok((k as f64 / flags.len() as f64, flags.len(), binom_two_sided_half(k, flags.len())))
}

/// Self-preference bias: judge rate − human rate of preferring the judge's own model's response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelfPreference {
    pub value: f64,
    pub judge_rate: f64,
    pub human_rate: f64,
    /// Exact McNemar (sign) test p-value on the discordant pairs.
    pub p_value: f64,
}

/// Self-preference bias of a judge against human verdicts on the same comparisons.
pub fn self_preference_bias(judge_prefers_own: &[bool], human_prefers_own: &[bool]) -> Result<SelfPreference> {
    if judge_prefers_own.is_empty() || judge_prefers_own.len() != human_prefers_own.len() {
        return invalid("judge_prefers_own and human_prefers_own must be non-empty and the same length.");
    }
    let n = judge_prefers_own.len() as f64;
    let jr = judge_prefers_own.iter().filter(|&&x| x).count() as f64 / n;
    let hr = human_prefers_own.iter().filter(|&&x| x).count() as f64 / n;
    let b = judge_prefers_own.iter().zip(human_prefers_own).filter(|(j, h)| **j && !**h).count();
    let c = judge_prefers_own.iter().zip(human_prefers_own).filter(|(j, h)| !**j && **h).count();
    let p = if b + c > 0 { binom_two_sided_half(b, b + c) } else { 1.0 };
    Ok(SelfPreference { value: jr - hr, judge_rate: jr, human_rate: hr, p_value: p })
}

/// Rubric scores normalised to [0, 1]: overall mean and per-criterion means.
#[derive(Debug, Clone, PartialEq)]
pub struct Rubric {
    pub value: f64,
    pub per_criterion: Vec<f64>,
    pub per_response: Vec<f64>,
}

/// Mean rubric score, normalised by the scale `(lo, hi)`; `scores[i]` = one response's criterion scores.
pub fn rubric_score(scores: &[Vec<f64>], scale: (f64, f64)) -> Result<Rubric> {
    let c = scores.first().map_or(0, Vec::len);
    if c == 0 || scores.iter().any(|r| r.len() != c) {
        return invalid("scores must be a 2-D array: one row per response, one column per criterion.");
    }
    let (lo, hi) = scale;
    if hi <= lo {
        return invalid("scale must be (min, max) with max > min.");
    }
    if scores.iter().flatten().any(|&x| x.is_nan() || x < lo || x > hi) {
        return invalid(format!("scores must lie within the scale {scale:?} without missing values."));
    }
    let norm: Vec<f64> = scores.iter().flatten().map(|x| (x - lo) / (hi - lo)).collect();
    let per_criterion = (0..c).map(|j| mean(&norm.iter().skip(j).step_by(c).copied().collect::<Vec<_>>())).collect();
    let per_response = norm.chunks(c).map(mean).collect();
    Ok(Rubric { value: mean(&norm), per_criterion, per_response })
}
