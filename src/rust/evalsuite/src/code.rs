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
    let frac: Vec<f64> = test_results
        .iter()
        .map(|r| r.iter().filter(|&&x| x).count() as f64 / r.len() as f64)
        .collect();
    let strict: Vec<f64> = test_results
        .iter()
        .map(|r| f64::from(u8::from(r.iter().all(|&x| x))))
        .collect();
    Ok(TestPass {
        value: mean(&frac),
        strict_pass_rate: mean(&strict),
        n_tests: test_results.iter().map(Vec::len).sum(),
    })
}

/// Share of programs your `checker` accepts as syntactically valid (e.g. a call to a parser).
pub fn syntax_validity_rate<S: AsRef<str>>(
    code: &[S],
    checker: &dyn Fn(&str) -> bool,
) -> Result<Rate> {
    nonempty(code, "code")?;
    Ok(rate(
        &code.iter().map(|c| checker(c.as_ref())).collect::<Vec<_>>(),
    ))
}

/// Static-analysis violations per 1,000 lines, with the share of programs with any violation and the mean
/// per program.
pub fn static_analysis_violation_rate(
    violations: &[f64],
    lines: &[f64],
) -> Result<(f64, f64, f64)> {
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
pub fn execution_success_rate<S: AsRef<str>>(
    outcomes: &[S],
    success: &[&str],
) -> Result<(Rate, Vec<(String, f64)>)> {
    nonempty(outcomes, "outcomes")?;
    let flags: Vec<bool> = outcomes
        .iter()
        .map(|o| success.contains(&o.as_ref()))
        .collect();
    let mut fails: Vec<&str> = outcomes
        .iter()
        .map(AsRef::as_ref)
        .filter(|o| !success.contains(o))
        .collect();
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
    let per: Vec<f64> = covered
        .iter()
        .zip(total)
        .map(|(c, t)| if *t > 0.0 { c / t } else { 1.0 })
        .collect();
    Ok((crate::util::np_sum(covered) / t, mean(&per)))
}

/// Patch acceptance rate (`accepted`), or the bug-reproduction rate: a reproduction test that fails before
/// the fix and passes after.
pub fn patch_acceptance_rate(
    accepted: Option<&[bool]>,
    fails_before: Option<&[bool]>,
    passes_after: Option<&[bool]>,
) -> Result<Rate> {
    if let Some(a) = accepted {
        nonempty(a, "accepted")?;
        return Ok(rate(a));
    }
    let (Some(fb), Some(pa)) = (fails_before, passes_after) else {
        return invalid("Pass accepted=, or both fails_before= and passes_after=.");
    };
    nonempty(fb, "fails_before")?;
    same(fb.len(), pa.len(), "fails_before", "passes_after")?;
    Ok(rate(
        &fb.iter().zip(pa).map(|(a, b)| *a && *b).collect::<Vec<_>>(),
    ))
}

/// SWE-bench resolved rate: every FAIL_TO_PASS and PASS_TO_PASS test passes (and the patch applied, when
/// `applied` is given). Returns (rate, share passing FAIL_TO_PASS alone, apply rate).
pub fn resolved_rate(
    fail_to_pass: &[Vec<bool>],
    pass_to_pass: &[Vec<bool>],
    applied: Option<&[bool]>,
) -> Result<(Rate, f64, Option<f64>)> {
    nonempty(fail_to_pass, "fail_to_pass")?;
    nonempty(pass_to_pass, "pass_to_pass")?;
    same(
        fail_to_pass.len(),
        pass_to_pass.len(),
        "fail_to_pass",
        "pass_to_pass",
    )?;
    if fail_to_pass.iter().any(Vec::is_empty) {
        return invalid("Every instance needs at least one FAIL_TO_PASS test.");
    }
    let mut ok: Vec<bool> = fail_to_pass
        .iter()
        .zip(pass_to_pass)
        .map(|(f, p)| f.iter().all(|&x| x) && p.iter().all(|&x| x))
        .collect();
    let f2p = mean(
        &fail_to_pass
            .iter()
            .map(|f| f64::from(u8::from(f.iter().all(|&x| x))))
            .collect::<Vec<_>>(),
    );
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
pub fn security_vulnerability_rate<S: AsRef<str>>(
    findings: &[Vec<S>],
    min_severity: &str,
) -> Result<(Rate, [f64; 4])> {
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
pub fn runtime_efficiency(
    times: &[f64],
    reference_times: &[f64],
    memory: Option<(&[f64], &[f64])>,
) -> Result<Runtime> {
    floats(times, "times", Some(0.0), None)?;
    floats(reference_times, "reference_times", Some(0.0), None)?;
    same(
        times.len(),
        reference_times.len(),
        "times",
        "reference_times",
    )?;
    if reference_times.iter().any(|&r| r <= 0.0) {
        return invalid("reference_times must be positive.");
    }
    let ratio: Vec<f64> = times
        .iter()
        .zip(reference_times)
        .map(|(t, r)| t / r)
        .collect();
    let nmu = match memory {
        Some((m, rm)) => {
            floats(m, "memory", Some(0.0), None)?;
            floats(rm, "reference_memory", Some(0.0), None)?;
            same(times.len(), m.len(), "times", "memory")?;
            same(times.len(), rm.len(), "times", "reference_memory")?;
            if rm.iter().any(|&x| x <= 0.0) {
                return invalid("reference_memory must be positive.");
            }
            Some(mean(
                &m.iter().zip(rm).map(|(a, b)| a / b).collect::<Vec<_>>(),
            ))
        }
        None => None,
    };
    Ok(Runtime {
        value: mean(&ratio),
        median_time_ratio: percentile(&sorted(&ratio), 50.0),
        share_slower_than_2x: ratio.iter().filter(|&&r| r > 2.0).count() as f64
            / ratio.len() as f64,
        nmu,
    })
}

/// Python's keywords, the default keyword list of CodeBLEU's weighted n-gram match.
pub const PYTHON_KEYWORDS: [&str; 38] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield", "match", "case", "type",
];

fn corpus_bleu(refs: &[Vec<Vec<&str>>], hyps: &[Vec<&str>], keywords: Option<&[&str]>) -> f64 {
    let mut v = Vocab::default();
    let h_ids: Vec<Vec<u32>> = hyps.iter().map(|h| v.encode(h)).collect();
    let r_ids: Vec<Vec<Vec<u32>>> = refs
        .iter()
        .map(|rs| rs.iter().map(|r| v.encode(r)).collect())
        .collect();
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
    let logs: Vec<f64> = (0..4)
        .map(|n| 0.25 * ((if num[n] != 0.0 { num[n] } else { 0.1 }) / den[n]).ln())
        .collect();
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
    same(
        references.len(),
        predictions.len(),
        "references",
        "predictions",
    )?;
    if references.iter().any(Vec::is_empty) {
        return invalid("references[i] must be a string or a list of strings.");
    }
    if weights.iter().any(|&w| w < 0.0) || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
        return invalid("weights must be four non-negative numbers summing to 1.");
    }
    let tok = |s: &str| -> Vec<String> {
        tokenizer.map_or_else(
            || {
                py_split(s.trim_matches(crate::generation::py_space))
                    .into_iter()
                    .map(str::to_owned)
                    .collect()
            },
            |f| f(s.trim_matches(crate::generation::py_space)),
        )
    };
    let t_refs: Vec<Vec<Vec<String>>> = references
        .iter()
        .map(|rs| rs.iter().map(|r| tok(r.as_ref())).collect())
        .collect();
    let t_hyps: Vec<Vec<String>> = predictions.iter().map(|p| tok(p.as_ref())).collect();
    let r: Vec<Vec<Vec<&str>>> = t_refs
        .iter()
        .map(|rs| {
            rs.iter()
                .map(|x| x.iter().map(String::as_str).collect())
                .collect()
        })
        .collect();
    let h: Vec<Vec<&str>> = t_hyps
        .iter()
        .map(|x| x.iter().map(String::as_str).collect())
        .collect();
    let ngram = corpus_bleu(&r, &h, None);
    let weighted = corpus_bleu(&r, &h, Some(keywords.unwrap_or(&PYTHON_KEYWORDS)));
    let [a, b, g, d] = weights;
    let df = if dataflow_match == 0.0 {
        1.0
    } else {
        dataflow_match
    };
    Ok(CodeBleuParts {
        ngram_match: ngram,
        weighted_ngram_match: weighted,
        syntax_match,
        dataflow_match,
        value: a * ngram + b * weighted + g * syntax_match + d * df,
    })
}

// ------------------------------------------------------------------------------- Python source analysis

#[cfg(feature = "python")]
use crate::pyast::{self, Ctx, Node, Tok, Token};
#[cfg(feature = "python")]
use std::collections::{HashMap, HashSet};

/// Whether Python source is syntactically valid: it parses, and passes the common compile-time checks
/// (`return` / `yield` / `await` outside a function, `break` / `continue` outside a loop, `nonlocal` at module
/// level). A built-in checker for [`syntax_validity_rate`].
#[cfg(feature = "python")]
pub fn python_syntax_ok(src: &str) -> bool {
    if src.contains('\0') {
        return false;
    }
    fn check(n: &Node, func: bool, asyncf: bool, lp: bool) -> bool {
        match n.tag {
            "Return" if !func => return false,
            "Yield" | "YieldFrom" if !func => return false,
            "Await" if !asyncf => return false,
            "Break" | "Continue" if !lp => return false,
            "Nonlocal" if !func => return false,
            _ => {}
        }
        match n.tag {
            "FunctionDef" | "AsyncFunctionDef" => {
                let a = n.tag == "AsyncFunctionDef";
                n.fields.iter().all(|&(f, s, e)| {
                    let inner = f == "body";
                    n.kids[s..e].iter().all(|k| {
                        if inner {
                            check(k, true, a, false)
                        } else {
                            check(k, func, asyncf, lp)
                        }
                    })
                })
            }
            "Lambda" => n.fields.iter().all(|&(f, s, e)| {
                n.kids[s..e].iter().all(|k| {
                    if f == "body" {
                        check(k, true, false, false)
                    } else {
                        check(k, func, asyncf, lp)
                    }
                })
            }),
            "ClassDef" => n.fields.iter().all(|&(f, s, e)| {
                n.kids[s..e].iter().all(|k| {
                    if f == "body" {
                        check(k, false, false, false)
                    } else {
                        check(k, func, asyncf, lp)
                    }
                })
            }),
            "For" | "AsyncFor" | "While" => n.fields.iter().all(|&(f, s, e)| {
                n.kids[s..e]
                    .iter()
                    .all(|k| check(k, func, asyncf, if f == "body" { true } else { lp }))
            }),
            _ => n.kids.iter().all(|k| check(k, func, asyncf, lp)),
        }
    }
    pyast::parse(src).is_some_and(|t| check(&t, false, false, false))
}

#[cfg(feature = "python")]
struct Cx {
    off: bool,
    complexity: i64,
    functions: Vec<(String, i64)>,
    classes: Vec<i64>,
    methods: Vec<(String, i64)>,
}

#[cfg(feature = "python")]
impl Cx {
    fn run(n: &Node, off: bool) -> Cx {
        let mut v = Cx {
            off,
            complexity: i64::from(off),
            functions: vec![],
            classes: vec![],
            methods: vec![],
        };
        v.visit(n);
        v
    }

    fn visit(&mut self, n: &Node) {
        match n.tag {
            "Assert" => self.complexity += 1,
            "FunctionDef" | "AsyncFunctionDef" => {
                let body = 1 + n
                    .field("body")
                    .iter()
                    .map(|c| Cx::run(c, false).complexity)
                    .sum::<i64>();
                self.functions
                    .push((n.name.clone().unwrap_or_default(), body));
            }
            "ClassDef" => {
                let name = n.name.clone().unwrap_or_default();
                let mut body = 1;
                for c in n.field("body") {
                    let v = Cx::run(c, false);
                    body += v.complexity + v.functions.iter().map(|f| f.1).sum::<i64>();
                    self.methods.extend(
                        v.functions
                            .into_iter()
                            .map(|(f, c)| (format!("{name}.{f}"), c)),
                    );
                }
                self.classes.push(body);
            }
            _ => {
                let len = |f: &str| n.field(f).len() as i64;
                self.complexity += match n.tag {
                    "Try" | "TryStar" => len("handlers") + i64::from(len("orelse") > 0),
                    "BoolOp" => len("values") - 1,
                    "If" | "IfExp" => 1,
                    "Match" => {
                        let cases = n.field("cases");
                        let underscore = cases.iter().any(|c| {
                            c.field("pattern")
                                .first()
                                .is_some_and(|p| p.has("pattern") && p.field("pattern").is_empty())
                        });
                        (cases.len() as i64 - i64::from(underscore)).max(0)
                    }
                    "For" | "While" | "AsyncFor" => i64::from(len("orelse") > 0) + 1,
                    "comprehension" => len("ifs") + 1,
                    _ => 0,
                };
                for k in &n.kids {
                    self.visit(k);
                }
            }
        }
    }

    fn total(&self) -> i64 {
        let f: i64 = self.functions.iter().map(|x| x.1).sum::<i64>() - self.functions.len() as i64;
        let k: i64 = self.classes.iter().sum::<i64>() - self.classes.len() as i64;
        self.complexity + f + k + i64::from(!self.off)
    }
}

#[cfg(feature = "python")]
fn halstead_volume(tree: &Node) -> f64 {
    let mut ops: HashSet<&'static str> = HashSet::new();
    let mut opnds: HashSet<(Option<String>, String)> = HashSet::new();
    let (mut n1, mut n2) = (0usize, 0usize);
    let mut unique = 0usize;
    fn key(n: &Node, ctx: &Option<String>, unique: &mut usize) -> (Option<String>, String) {
        let k = match n.tag {
            "Name" | "Attribute" => format!("v:{}", n.name.as_deref().unwrap_or("")),
            // Python compares the raw values: a str constant equals a name with the same text.
            "Constant" => match n.constant.as_deref().unwrap_or("").strip_prefix("s:") {
                Some(text) => format!("v:{text}"),
                None => format!("k:{}", n.constant.as_deref().unwrap_or("")),
            },
            _ => {
                *unique += 1;
                format!("id:{unique}")
            }
        };
        (ctx.clone(), k)
    }
    #[allow(clippy::too_many_arguments)]
    fn visit(
        n: &Node,
        ctx: &Option<String>,
        ops: &mut HashSet<&'static str>,
        opnds: &mut HashSet<(Option<String>, String)>,
        n1: &mut usize,
        n2: &mut usize,
        unique: &mut usize,
    ) {
        if n.tag == "FunctionDef" || n.tag == "AsyncFunctionDef" {
            let c = n.name.clone();
            for b in n.field("body") {
                visit(b, &c, ops, opnds, n1, n2, unique);
            }
            return;
        }
        let rec: Option<(usize, Vec<&'static str>, Vec<&Node>)> = match n.tag {
            "BinOp" => Some((
                1,
                vec![n.op.unwrap()],
                vec![&n.field("left")[0], &n.field("right")[0]],
            )),
            "UnaryOp" => Some((1, vec![n.op.unwrap()], vec![&n.field("operand")[0]])),
            "BoolOp" => Some((1, vec![n.op.unwrap()], n.field("values").iter().collect())),
            "AugAssign" => Some((
                1,
                vec![n.op.unwrap()],
                vec![&n.field("target")[0], &n.field("value")[0]],
            )),
            "Compare" => {
                let mut o: Vec<&Node> = n.field("comparators").iter().collect();
                o.push(&n.field("left")[0]);
                Some((n.ops.len(), n.ops.clone(), o))
            }
            _ => None,
        };
        if let Some((k, names, operands)) = rec {
            *n1 += k;
            *n2 += operands.len();
            ops.extend(names);
            for o in operands {
                let kk = key(o, ctx, unique);
                opnds.insert(kk);
            }
        }
        for k in &n.kids {
            visit(k, ctx, ops, opnds, n1, n2, unique);
        }
    }
    for k in &tree.kids {
        visit(
            k,
            &None,
            &mut ops,
            &mut opnds,
            &mut n1,
            &mut n2,
            &mut unique,
        );
    }
    let (n, eta) = (n1 + n2, ops.len() + opnds.len());
    if eta > 0 {
        n as f64 * (eta as f64).log2()
    } else {
        0.0
    }
}

#[cfg(feature = "python")]
fn logical_lines(tokens: &[Token]) -> i64 {
    let mut groups: Vec<Vec<&Token>> = vec![vec![]];
    for t in tokens {
        if t.kind == Tok::Op && t.text == ";" && t.start != t.end {
            groups.push(vec![]);
        } else {
            groups.last_mut().unwrap().push(t);
        }
    }
    let mut total = 0;
    for g in groups {
        let kept: Vec<&&Token> = g
            .iter()
            .filter(|t| !matches!(t.kind, Tok::Comment | Tok::Nl | Tok::Newline))
            .collect();
        let colon: Vec<usize> = kept
            .iter()
            .enumerate()
            .filter(|(_, t)| t.kind == Tok::Op && t.text == ":")
            .map(|(i, _)| i)
            .collect();
        if let Some(&last) = colon.last() {
            total += 2 - i64::from(kept.len() >= 2 && last == kept.len() - 2);
        } else if kept
            .iter()
            .any(|t| !matches!(t.kind, Tok::Nl | Tok::Newline | Tok::EndMarker))
        {
            total += 1;
        }
    }
    total
}

/// radon's raw metrics: (logical lines, source lines, comment + multi-line string lines).
#[cfg(feature = "python")]
fn raw(src: &str) -> (i64, i64, i64) {
    let ws = crate::generation::py_space;
    let lines: Vec<&str> = crate::qa::py_splitlines(src)
        .into_iter()
        .map(|l| l.trim_matches(ws))
        .collect();
    let (mut lloc, mut comments, mut multi, mut sloc) = (0i64, 0i64, 0i64, 0i64);
    let mut i = 0;
    'outer: while i < lines.len() {
        let mut used = vec![lines[i]];
        i += 1;
        let tokens = loop {
            if let Ok(t) = pyast::tokenize(&used.join("\n")) {
                break t;
            }
            if i >= lines.len() {
                break 'outer;
            }
            used.push(lines[i]);
            i += 1;
        };
        comments += tokens.iter().filter(|t| t.kind == Tok::Comment).count() as i64;
        let only = tokens[1..]
            .iter()
            .all(|t| matches!(t.kind, Tok::EndMarker | Tok::Nl | Tok::Newline));
        let nonempty = used.iter().filter(|l| !l.is_empty()).count() as i64;
        if tokens[0].kind == Tok::Comment && only {
        } else if tokens[0].kind == Tok::String && only {
            if tokens[0].end.0 != tokens[0].start.0 {
                multi += nonempty;
            }
        } else {
            sloc += nonempty;
        }
        lloc += logical_lines(&tokens);
    }
    (lloc, sloc, comments + multi)
}

/// Complexity of one program.
#[cfg(feature = "python")]
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramComplexity {
    /// Cyclomatic complexity of each function and method (`Class.method`).
    pub functions: Vec<(String, i64)>,
    pub total_complexity: i64,
    pub maintainability_index: f64,
}

/// Code complexity: mean cyclomatic complexity per function / method, its maximum, the mean
/// maintainability index and per-program results.
#[cfg(feature = "python")]
#[derive(Debug, Clone, PartialEq)]
pub struct Complexity {
    pub value: f64,
    pub max_complexity: i64,
    pub maintainability_index: f64,
    pub per_program: Vec<ProgramComplexity>,
}

/// Cyclomatic complexity and maintainability index of Python programs, with radon's counting rules (as the
/// Python package). Programs that do not parse are rejected.
#[cfg(feature = "python")]
pub fn code_complexity<S: AsRef<str>>(code: &[S]) -> Result<Complexity> {
    nonempty(code, "code")?;
    let (mut funcs, mut mi, mut per) = (Vec::new(), Vec::new(), Vec::new());
    for (i, src) in code.iter().enumerate() {
        let src = src.as_ref();
        let Some(tree) = pyast::parse(src) else {
            return invalid(format!("code[{i}] is not valid Python."));
        };
        let v = Cx::run(&tree, true);
        let vol = halstead_volume(&tree);
        let blocks: Vec<(String, i64)> = v.functions.iter().chain(&v.methods).cloned().collect();
        funcs.extend(blocks.iter().map(|b| b.1));
        let g = v.total();
        let (lloc, sloc, comments) = raw(src);
        let c_pct = if sloc > 0 {
            comments as f64 / sloc as f64 * 100.0
        } else {
            0.0
        };
        let m = if vol <= 0.0 || lloc <= 0 {
            100.0
        } else {
            let nn = 171.0 - 5.2 * vol.ln() - 0.23 * g as f64 - 16.2 * (lloc as f64).ln()
                + 50.0 * (2.46 * c_pct.to_radians()).sqrt().sin();
            (nn * 100.0 / 171.0).clamp(0.0, 100.0)
        };
        mi.push(m);
        per.push(ProgramComplexity {
            functions: blocks,
            total_complexity: g,
            maintainability_index: m,
        });
    }
    Ok(Complexity {
        value: if funcs.is_empty() {
            1.0
        } else {
            funcs.iter().sum::<i64>() as f64 / funcs.len() as f64
        },
        max_complexity: funcs.iter().copied().max().unwrap_or(1),
        maintainability_index: mean(&mi),
        per_program: per,
    })
}

/// Python source without comments and docstrings, blank lines removed (CodeBLEU's pre-processing); the
/// source unchanged when it does not tokenize.
#[cfg(feature = "python")]
pub fn strip_python_comments(src: &str) -> String {
    let Ok(tokens) = pyast::tokenize(src) else {
        return src.to_owned();
    };
    let mut out = String::new();
    let (mut prev, mut last_line, mut last_col) = (Tok::Indent, 0usize, 0usize);
    for t in &tokens {
        if t.kind == Tok::Op && t.start == t.end {
            continue; // f-string colon marker, not text
        }
        let ((sl, sc), (el, ec)) = (t.start, t.end);
        if sl > last_line {
            last_col = 0;
        }
        if sc > last_col {
            out.extend(std::iter::repeat_n(' ', sc - last_col));
        }
        if !(t.kind == Tok::Comment
            || (t.kind == Tok::String && matches!(prev, Tok::Indent | Tok::Newline)))
        {
            out.push_str(&t.text);
        }
        prev = t.kind;
        last_col = ec;
        last_line = el;
    }
    out.split('\n')
        .filter(|l| !l.trim_matches(crate::generation::py_space).is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(feature = "python")]
fn subtrees(tree: &Node, table: &mut HashMap<(String, Vec<usize>), usize>) -> Vec<usize> {
    fn go(
        n: &Node,
        root: bool,
        table: &mut HashMap<(String, Vec<usize>), usize>,
        out: &mut Vec<usize>,
    ) -> usize {
        let ids: Vec<usize> = n.kids.iter().map(|k| go(k, false, table, out)).collect();
        let tag = match n.tag {
            "BinOp" | "UnaryOp" | "BoolOp" | "AugAssign" => {
                format!("{}:{}", n.tag, n.op.unwrap_or(""))
            }
            t => t.to_owned(),
        };
        let len = table.len();
        let has = !ids.is_empty();
        let sid = *table.entry((tag, ids)).or_insert(len);
        if has || root {
            out.push(sid);
        }
        sid
    }
    let mut out = Vec::new();
    go(tree, true, table, &mut out);
    out
}

#[cfg(feature = "python")]
type Edge = (String, &'static str, Vec<String>);

#[cfg(feature = "python")]
fn dataflow(tree: &Node) -> Vec<Edge> {
    fn names(n: Option<&Node>) -> Vec<String> {
        n.map_or(vec![], |n| {
            n.walk()
                .into_iter()
                .filter(|x| x.tag == "Name")
                .map(|x| x.name.clone().unwrap_or_default())
                .collect()
        })
    }
    fn sorted_set(v: Vec<String>) -> Vec<String> {
        let mut v = v;
        v.sort();
        v.dedup();
        v
    }
    fn targets(n: &Node) -> Vec<String> {
        n.walk()
            .into_iter()
            .filter(|x| x.tag == "Name" && x.ctx == Ctx::Store)
            .map(|x| x.name.clone().unwrap_or_default())
            .collect()
    }
    fn uses(n: Option<&Node>, defined: &HashSet<String>, edges: &mut Vec<Edge>) {
        if let Some(n) = n {
            for x in n.walk() {
                if x.tag == "Name" && x.ctx == Ctx::Load {
                    let id = x.name.clone().unwrap_or_default();
                    if defined.contains(&id) {
                        edges.push((id.clone(), "comesFrom", vec![id]));
                    }
                }
            }
        }
    }
    fn stmt(s: &Node, defined: &mut HashSet<String>, edges: &mut Vec<Edge>) {
        match s.tag {
            "Assign" | "AnnAssign" | "AugAssign" => {
                let value = s.field("value").first();
                uses(value, defined, edges);
                let src = sorted_set(names(value));
                let tg = if s.tag == "Assign" {
                    s.field("targets")
                } else {
                    s.field("target")
                };
                for t in tg {
                    for v in targets(t) {
                        let mut from = src.clone();
                        if s.tag == "AugAssign" {
                            from.push(v.clone());
                        }
                        edges.push((v.clone(), "computedFrom", from));
                        defined.insert(v);
                    }
                }
            }
            "For" | "AsyncFor" => {
                let iter = s.field("iter").first();
                uses(iter, defined, edges);
                for v in targets(&s.field("target")[0]) {
                    edges.push((v.clone(), "computedFrom", sorted_set(names(iter))));
                    defined.insert(v);
                }
                for b in s.field("body").iter().chain(s.field("orelse")) {
                    stmt(b, defined, edges);
                }
            }
            "FunctionDef" | "AsyncFunctionDef" => {
                let a = &s.field("args")[0];
                for x in a
                    .field("posonlyargs")
                    .iter()
                    .chain(a.field("args"))
                    .chain(a.field("kwonlyargs"))
                {
                    defined.insert(x.name.clone().unwrap_or_default());
                }
                for b in s.field("body") {
                    stmt(b, defined, edges);
                }
            }
            _ => {
                let body_fields: Vec<&str> = ["body", "orelse", "finalbody", "handlers"]
                    .into_iter()
                    .filter(|f| s.has(f))
                    .collect();
                for &(f, a, b) in &s.fields {
                    if !body_fields.contains(&f) {
                        for x in &s.kids[a..b] {
                            uses(Some(x), defined, edges);
                        }
                    }
                }
                for f in body_fields {
                    for b in s.field(f) {
                        if b.tag == "ExceptHandler" {
                            if let Some(n) = &b.name {
                                defined.insert(n.clone());
                            }
                            for x in b.field("body") {
                                stmt(x, defined, edges);
                            }
                        } else {
                            stmt(b, defined, edges);
                        }
                    }
                }
            }
        }
    }
    let (mut defined, mut edges) = (HashSet::new(), Vec::new());
    for s in tree.field("body") {
        stmt(s, &mut defined, &mut edges);
    }
    edges
}

#[cfg(feature = "python")]
fn normalize_flow(flow: Vec<Edge>) -> Vec<Edge> {
    let mut names: HashMap<String, String> = HashMap::new();
    let name = |s: &str, names: &mut HashMap<String, String>| -> String {
        let n = names.len();
        names
            .entry(s.to_owned())
            .or_insert_with(|| format!("var_{n}"))
            .clone()
    };
    flow.into_iter()
        .map(|(v, rel, src)| {
            for s in &src {
                name(s, &mut names);
            }
            let vv = name(&v, &mut names);
            (vv, rel, src.iter().map(|s| name(s, &mut names)).collect())
        })
        .collect()
}

/// CodeBLEU of Python code with all four components computed here: n-gram and keyword-weighted n-gram
/// BLEU, AST sub-tree match and data-flow match (as the Python package's `codebleu`).
#[cfg(feature = "python")]
pub fn codebleu_python<R: AsRef<str>, P: AsRef<str>>(
    references: &[Vec<R>],
    predictions: &[P],
    weights: [f64; 4],
    keywords: Option<&[&str]>,
    tokenizer: Option<&dyn Fn(&str) -> Vec<String>>,
) -> Result<CodeBleuParts> {
    let base = codebleu(
        references,
        predictions,
        0.0,
        0.0,
        weights,
        keywords,
        tokenizer,
    )?;
    let ws = crate::generation::py_space;
    let parse = |s: &str| pyast::parse(&strip_python_comments(s));
    let mut table = HashMap::new();
    let (mut matched, mut total, mut df_match, mut df_total) = (0usize, 0usize, 0usize, 0usize);
    for (rs, p) in references.iter().zip(predictions) {
        let cand = parse(p.as_ref().trim_matches(ws));
        let cand_sub: HashSet<usize> = cand.as_ref().map_or_else(HashSet::new, |c| {
            subtrees(c, &mut table).into_iter().collect()
        });
        let cand_flow = cand
            .as_ref()
            .map_or_else(Vec::new, |c| normalize_flow(dataflow(c)));
        for r in rs {
            let Some(rt) = parse(r.as_ref().trim_matches(ws)) else {
                continue;
            };
            let ref_sub = subtrees(&rt, &mut table);
            matched += ref_sub.iter().filter(|s| cand_sub.contains(s)).count();
            total += ref_sub.len();
            let ref_flow = normalize_flow(dataflow(&rt));
            let mut pool = cand_flow.clone();
            df_total += ref_flow.len();
            for e in ref_flow {
                if let Some(k) = pool.iter().position(|x| *x == e) {
                    df_match += 1;
                    pool.remove(k);
                }
            }
        }
    }
    let syntax = if total > 0 {
        matched as f64 / total as f64
    } else {
        0.0
    };
    let dataflow_m = if df_total > 0 {
        df_match as f64 / df_total as f64
    } else {
        0.0
    };
    let [a, b, g, d] = weights;
    let df = if dataflow_m == 0.0 { 1.0 } else { dataflow_m };
    Ok(CodeBleuParts {
        syntax_match: syntax,
        dataflow_match: dataflow_m,
        value: a * base.ngram_match + b * base.weighted_ngram_match + g * syntax + d * df,
        ..base
    })
}

/// Whether the comment-stripped source parses (CodeBLEU's parse step); for the parity tests.
#[cfg(feature = "python")]
#[doc(hidden)]
pub fn __python_parses_stripped(src: &str) -> bool {
    pyast::parse(&strip_python_comments(src)).is_some()
}
