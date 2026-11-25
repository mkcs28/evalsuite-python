//! Code-generation metrics: unit-test pass rates, static-analysis violations, execution success, coverage,
//! patch acceptance, SWE-bench resolved rate, security findings, runtime efficiency, syntax validity and
//! CodeBLEU, as EvalSuite's Python `evalsuite.llmsys.code`. Nothing here executes code: outcomes come from
//! your sandboxed runner.
//!
//! Python's own parser is not available from Rust, so [`syntax_validity_rate`] takes your checker, and
//! [`codebleu`] computes the n-gram and keyword-weighted n-gram components itself and takes the syntax
//! (AST sub-tree) and data-flow match scores from your parser (see [`CodeBleuParts`]). Cyclomatic complexity
//! and the maintainability index (`code_complexity` in Python, which uses CPython's `ast` and `tokenize`) are
//! not ported.
//!
//! ```
//! use evalsuite::code::unit_test_pass_rate;
//! let r = unit_test_pass_rate(&[vec![true, true], vec![true, false]]).unwrap();
//! assert_eq!((r.value, r.strict_pass_rate), (0.75, 0.5));
//! ```

use crate::error::{invalid, Result};
use crate::generation::{py_split, Vocab};
use crate::summary::{floats, nonempty, rate, same, Rate};
use crate::util::{mean, percentile, sorted};

/// Unit-test pass rate: mean share of tests passed per problem, and the share of problems passing all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestPass {
    pub value: f64,
    pub strict_pass_rate: f64,
    pub n_tests: usize,
}

/// `test_results[i]` = outcomes of problem i's tests.
pub fn unit_test_pass_rate(test_results: &[Vec<bool>]) -> Result<TestPass> {
    nonempty(test_results, "test_results")?;
    if test_results.iter().any(Vec::is_empty) {
        return invalid("Every problem needs at least one test result.");
    }
    let frac: Vec<f64> = test_results.iter().map(|r| r.iter().filter(|&&x| x).count() as f64 / r.len() as f64).collect();
    let strict: Vec<f64> = test_results.iter().map(|r| f64::from(u8::from(r.iter().all(|&x| x)))).collect();
    Ok(TestPass { value: mean(&frac), strict_pass_rate: mean(&strict), n_tests: test_results.iter().map(Vec::len).sum() })
}

/// Share of programs your `checker` accepts as syntactically valid (e.g. a call to a parser).
pub fn syntax_validity_rate<S: AsRef<str>>(code: &[S], checker: &dyn Fn(&str) -> bool) -> Result<Rate> {
    nonempty(code, "code")?;
    Ok(rate(&code.iter().map(|c| checker(c.as_ref())).collect::<Vec<_>>()))
}

/// Static-analysis violations per 1,000 lines, with the share of programs with any violation and the mean
/// per program.
pub fn static_analysis_violation_rate(violations: &[f64], lines: &[f64]) -> Result<(f64, f64, f64)> {
    floats(violations, "violations", Some(0.0), None)?;
    floats(lines, "lines", Some(0.0), None)?;
    same(violations.len(), lines.len(), "violations", "lines")?;
    let total: f64 = lines.iter().sum();
    if total == 0.0 {
        return invalid("lines sums to zero.");
    }
    let v: f64 = violations.iter().sum();
    let with = violations.iter().filter(|&&x| x > 0.0).count() as f64 / violations.len() as f64;
    Ok((1000.0 * v / total, with, mean(violations)))
}

/// Share of executions whose outcome is in `success` (e.g. `["passed", "ok", "True"]`), with the share of
/// each failure outcome (sorted).
pub fn execution_success_rate<S: AsRef<str>>(outcomes: &[S], success: &[&str]) -> Result<(Rate, Vec<(String, f64)>)> {
    nonempty(outcomes, "outcomes")?;
    let flags: Vec<bool> = outcomes.iter().map(|o| success.contains(&o.as_ref())).collect();
    let mut fails: Vec<&str> = outcomes.iter().map(AsRef::as_ref).filter(|o| !success.contains(o)).collect();
    fails.sort_unstable();
    let n = outcomes.len() as f64;
    let mut by: Vec<(String, f64)> = Vec::new();
    for f in fails {
        match by.last_mut() {
            Some(last) if last.0 == f => last.1 += 1.0,
            _ => by.push((f.to_owned(), 1.0)),
        }
    }
    by.iter_mut().for_each(|b| b.1 /= n);
    Ok((rate(&flags), by))
}

/// Pooled test coverage (covered / total units) and the mean per program (1 for programs with nothing to
/// cover).
pub fn coverage_rate(covered: &[f64], total: &[f64]) -> Result<(f64, f64)> {
    floats(covered, "covered", Some(0.0), None)?;
    floats(total, "total", Some(0.0), None)?;
    same(covered.len(), total.len(), "covered", "total")?;
    if covered.iter().zip(total).any(|(c, t)| c > t) {
        return invalid("covered cannot exceed total.");
    }
    let t: f64 = crate::util::np_sum(total);
    if t == 0.0 {
        return invalid("total sums to zero.");
    }
    let per: Vec<f64> = covered.iter().zip(total).map(|(c, t)| if *t > 0.0 { c / t } else { 1.0 }).collect();
    Ok((crate::util::np_sum(covered) / t, mean(&per)))
}

/// Patch acceptance rate (`accepted`), or the bug-reproduction rate: a reproduction test that fails before
/// the fix and passes after.
pub fn patch_acceptance_rate(accepted: Option<&[bool]>, fails_before: Option<&[bool]>, passes_after: Option<&[bool]>) -> Result<Rate> {
    if let Some(a) = accepted {
        nonempty(a, "accepted")?;
        return Ok(rate(a));
    }
    let (Some(fb), Some(pa)) = (fails_before, passes_after) else {
        return invalid("Pass accepted=, or both fails_before= and passes_after=.");
    };
    nonempty(fb, "fails_before")?;
    same(fb.len(), pa.len(), "fails_before", "passes_after")?;
    Ok(rate(&fb.iter().zip(pa).map(|(a, b)| *a && *b).collect::<Vec<_>>()))
}

/// SWE-bench resolved rate: every FAIL_TO_PASS and PASS_TO_PASS test passes (and the patch applied, when
/// `applied` is given). Returns (rate, share passing FAIL_TO_PASS alone, apply rate).
pub fn resolved_rate(fail_to_pass: &[Vec<bool>], pass_to_pass: &[Vec<bool>], applied: Option<&[bool]>) -> Result<(Rate, f64, Option<f64>)> {
    nonempty(fail_to_pass, "fail_to_pass")?;
    nonempty(pass_to_pass, "pass_to_pass")?;
    same(fail_to_pass.len(), pass_to_pass.len(), "fail_to_pass", "pass_to_pass")?;
    if fail_to_pass.iter().any(Vec::is_empty) {
        return invalid("Every instance needs at least one FAIL_TO_PASS test.");
    }
    let mut ok: Vec<bool> = fail_to_pass.iter().zip(pass_to_pass).map(|(f, p)| f.iter().all(|&x| x) && p.iter().all(|&x| x)).collect();
    let f2p = mean(&fail_to_pass.iter().map(|f| f64::from(u8::from(f.iter().all(|&x| x)))).collect::<Vec<_>>());
    let mut apply = None;
    if let Some(a) = applied {
        same(fail_to_pass.len(), a.len(), "fail_to_pass", "applied")?;
        ok.iter_mut().zip(a).for_each(|(o, a)| *o &= *a);
        apply = Some(a.iter().filter(|&&x| x).count() as f64 / a.len() as f64);
    }
    Ok((rate(&ok), f2p, apply))
}

/// Finding severities, lowest first.
pub const SEVERITIES: [&str; 4] = ["low", "medium", "high", "critical"];

/// Share of programs with a security finding at or above `min_severity`, with the findings per program for
/// each severity. `findings[i]` lists program i's finding severities (case-insensitive).
pub fn security_vulnerability_rate<S: AsRef<str>>(findings: &[Vec<S>], min_severity: &str) -> Result<(Rate, [f64; 4])> {
    let Some(min) = SEVERITIES.iter().position(|s| *s == min_severity) else {
        return invalid(format!("min_severity must be one of {SEVERITIES:?}."));
    };
    nonempty(findings, "findings")?;
    let mut counts = [0usize; 4];
    let mut flags = Vec::new();
    for (i, r) in findings.iter().enumerate() {
        let mut any = false;
        for s in r {
            let low = s.as_ref().to_lowercase();
            let Some(k) = SEVERITIES.iter().position(|x| *x == low) else {
                return invalid(format!("findings[{i}] has unknown severities [{low:?}]."));
            };
            counts[k] += 1;
            any |= k >= min;
        }
        flags.push(any);
    }
    let n = findings.len() as f64;
    Ok((rate(&flags), counts.map(|c| c as f64 / n)))
}

/// Normalised execution time (mean time / reference time), with its median, the share more than 2× slower,
/// and the normalised memory usage when memory is given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Runtime {
    pub value: f64,
    pub median_time_ratio: f64,
    pub share_slower_than_2x: f64,
    pub nmu: Option<f64>,
}

/// Runtime efficiency against reference solutions (EffiBench NET / NMU).
pub fn runtime_efficiency(times: &[f64], reference_times: &[f64], memory: Option<(&[f64], &[f64])>) -> Result<Runtime> {
    floats(times, "times", Some(0.0), None)?;
    floats(reference_times, "reference_times", Some(0.0), None)?;
    same(times.len(), reference_times.len(), "times", "reference_times")?;
    if reference_times.iter().any(|&r| r <= 0.0) {
        return invalid("reference_times must be positive.");
    }
    let ratio: Vec<f64> = times.iter().zip(reference_times).map(|(t, r)| t / r).collect();
    let nmu = match memory {
        Some((m, rm)) => {
            floats(m, "memory", Some(0.0), None)?;
            floats(rm, "reference_memory", Some(0.0), None)?;
            same(times.len(), m.len(), "times", "memory")?;
            same(times.len(), rm.len(), "times", "reference_memory")?;
            if rm.iter().any(|&x| x <= 0.0) {
                return invalid("reference_memory must be positive.");
            }
            Some(mean(&m.iter().zip(rm).map(|(a, b)| a / b).collect::<Vec<_>>()))
        }
        None => None,
    };
    Ok(Runtime {
        value: mean(&ratio),
        median_time_ratio: percentile(&sorted(&ratio), 50.0),
        share_slower_than_2x: ratio.iter().filter(|&&r| r > 2.0).count() as f64 / ratio.len() as f64,
        nmu,
    })
}

/// Python's keywords, the default keyword list of CodeBLEU's weighted n-gram match.
pub const PYTHON_KEYWORDS: [&str; 38] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield", "match", "case", "type",
];

fn corpus_bleu(refs: &[Vec<Vec<&str>>], hyps: &[Vec<&str>], keywords: Option<&[&str]>) -> f64 {
    let mut v = Vocab::default();
    let h_ids: Vec<Vec<u32>> = hyps.iter().map(|h| v.encode(h)).collect();
    let r_ids: Vec<Vec<Vec<u32>>> = refs.iter().map(|rs| rs.iter().map(|r| v.encode(r)).collect()).collect();
    let kw: Option<Vec<bool>> = keywords.map(|k| {
        let mut flags = vec![false; v.len()];
        for (t, &id) in v.iter() {
            flags[id as usize] = k.contains(t);
        }
        flags
    });
    // Python 3.12+ `sum()` is compensated; the fixtures and the Python package on current versions use it.
    let (mut num, mut den) = ([0.0f64; 4], [0.0f64; 4]);
    let (mut hl, mut rl) = (0i64, 0i64);
    for (h, rs) in h_ids.iter().zip(&r_ids) {
        let rr: Vec<&[u32]> = rs.iter().map(Vec::as_slice).collect();
        let (terms, a, b) = evalsuite_core::text::codebleu_example(h, &rr, kw.as_deref(), true);
        for (n, ts) in terms.iter().enumerate() {
            for &(x, y) in ts {
                num[n] += x;
                den[n] += y;
            }
        }
        hl += a;
        rl += b;
    }
    if num[0] == 0.0 {
        return 0.0;
    }
    let (r, c) = (rl as f64, hl as f64);
    let bp = if c > r {
        1.0
    } else if c == 0.0 {
        0.0
    } else {
        (1.0 - r / c).exp()
    };
    let logs: Vec<f64> = (0..4).map(|n| 0.25 * ((if num[n] != 0.0 { num[n] } else { 0.1 }) / den[n]).ln()).collect();
    bp * fsum(&logs).exp()
}

/// Exactly rounded sum (Python's `math.fsum`, Shewchuk's algorithm).
fn fsum(x: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &v in x {
        let mut v = v;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if v.abs() < y.abs() {
                std::mem::swap(&mut v, &mut y);
            }
            let hi = v + y;
            let lo = y - (hi - v);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            v = hi;
        }
        partials.truncate(i);
        partials.push(v);
    }
    partials.iter().rev().fold(0.0, |acc, p| acc + p)
}

/// CodeBLEU components and the weighted score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeBleuParts {
    pub ngram_match: f64,
    pub weighted_ngram_match: f64,
    /// AST sub-tree match (from your parser).
    pub syntax_match: f64,
    /// Data-flow match (from your parser); 0 counts as 1 in the score, as in the reference implementation
    /// when no reference has data flow.
    pub dataflow_match: f64,
    pub value: f64,
}

/// CodeBLEU (Ren et al. 2020): weights · (n-gram BLEU, keyword-weighted n-gram BLEU, syntax match, data-flow
/// match). The two n-gram terms are computed here from whitespace tokens (or `tokenizer`); `syntax_match`
/// and `dataflow_match` come from your parser of the language. `keywords` defaults to [`PYTHON_KEYWORDS`].
pub fn codebleu<R: AsRef<str>, P: AsRef<str>>(
    references: &[Vec<R>],
    predictions: &[P],
    syntax_match: f64,
    dataflow_match: f64,
    weights: [f64; 4],
    keywords: Option<&[&str]>,
    tokenizer: Option<&dyn Fn(&str) -> Vec<String>>,
) -> Result<CodeBleuParts> {
    nonempty(predictions, "predictions")?;
    same(references.len(), predictions.len(), "references", "predictions")?;
    if references.iter().any(Vec::is_empty) {
        return invalid("references[i] must be a string or a list of strings.");
    }
    if weights.iter().any(|&w| w < 0.0) || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
        return invalid("weights must be four non-negative numbers summing to 1.");
    }
    let tok = |s: &str| -> Vec<String> { tokenizer.map_or_else(|| py_split(s.trim_matches(crate::generation::py_space)).into_iter().map(str::to_owned).collect(), |f| f(s.trim_matches(crate::generation::py_space))) };
    let t_refs: Vec<Vec<Vec<String>>> = references.iter().map(|rs| rs.iter().map(|r| tok(r.as_ref())).collect()).collect();
    let t_hyps: Vec<Vec<String>> = predictions.iter().map(|p| tok(p.as_ref())).collect();
    let r: Vec<Vec<Vec<&str>>> = t_refs.iter().map(|rs| rs.iter().map(|x| x.iter().map(String::as_str).collect()).collect()).collect();
    let h: Vec<Vec<&str>> = t_hyps.iter().map(|x| x.iter().map(String::as_str).collect()).collect();
    let ngram = corpus_bleu(&r, &h, None);
    let weighted = corpus_bleu(&r, &h, Some(keywords.unwrap_or(&PYTHON_KEYWORDS)));
    let [a, b, g, d] = weights;
    let df = if dataflow_match == 0.0 { 1.0 } else { dataflow_match };
    Ok(CodeBleuParts { ngram_match: ngram, weighted_ngram_match: weighted, syntax_match, dataflow_match, value: a * ngram + b * weighted + g * syntax_match + d * df })
}
