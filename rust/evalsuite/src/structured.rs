//! Structured-output and tool-use metrics (JSON / XML validity, JSON Schema compliance, required fields, tool
//! selection and arguments, tool-call F1, API-call success, format and instruction compliance, extra content),
//! as EvalSuite's Python `evalsuite.text.structured`.
//!
//! Regular expressions use the `regex` crate (linear time; no look-around or back-references), where Python
//! uses its backtracking `re`; patterns are matched with dot-matches-newline on, as Python's default `re.S`.
//!
//! ```
//! use evalsuite::structured::{json_validity, JsonMode};
//! let r = json_validity(&["{\"a\": 1}", "not json"], JsonMode::Strict).unwrap();
//! assert_eq!(r.value, 0.5);
//! ```

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

pub use crate::agents::ToolCall;
use crate::error::{invalid, Result};
use crate::generation::py_space;
use crate::json_schema::{json_equal, validate};
use crate::summary::nonempty;

/// Where to look for JSON in a model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsonMode {
    /// The whole output (surrounding whitespace allowed).
    #[default]
    Strict,
    /// The first ```` ```json ```` code block if present, else the whole output.
    Fenced,
    /// As `Fenced`, then also the span from the first `{` to the last `}` (or `[` … `]`).
    Lenient,
}

/// Share of passing examples and the per-example flags.
#[derive(Debug, Clone, PartialEq)]
pub struct Flags {
    pub value: f64,
    pub per_example: Vec<bool>,
}

fn flags(per_example: Vec<bool>) -> Flags {
    let value = per_example.iter().filter(|&&x| x).count() as f64 / per_example.len() as f64;
    Flags { value, per_example }
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_space)
}

fn fence() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?s)```(?:json|JSON)?\s*\n?(.*?)```").unwrap())
}

/// Parses JSON from a model output (see [`JsonMode`]); `None` when nothing parses.
pub fn extract_json(text: &str, mode: JsonMode) -> Option<Value> {
    let mut candidates: Vec<&str> = vec![py_strip(text)];
    if mode != JsonMode::Strict {
        if let Some(c) = fence().captures(text) {
            candidates.insert(0, py_strip(c.get(1).unwrap().as_str()));
        }
    }
    if mode == JsonMode::Lenient {
        for (o, c) in [('{', '}'), ('[', ']')] {
            if let (Some(s), Some(e)) = (text.find(o), text.rfind(c)) {
                if s < e {
                    candidates.push(&text[s..=e]);
                }
            }
        }
    }
    candidates.into_iter().find_map(|c| serde_json::from_str(c).ok())
}

fn outputs<S: AsRef<str>>(o: &[S]) -> Result<()> {
    if o.is_empty() {
        return invalid("outputs is empty.");
    }
    Ok(())
}

/// Share of outputs that parse as JSON.
pub fn json_validity<S: AsRef<str>>(outputs_: &[S], mode: JsonMode) -> Result<Flags> {
    outputs(outputs_)?;
    Ok(flags(outputs_.iter().map(|o| extract_json(o.as_ref(), mode).is_some()).collect()))
}

/// JSON Schema compliance with the first error of each failing output.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaCompliance {
    pub flags: Flags,
    /// (output index, first error) of the failing outputs.
    pub errors: Vec<(usize, String)>,
}

/// Share of outputs that parse as JSON and validate against `schema` (see [`crate::json_schema`]).
pub fn json_schema_compliance<S: AsRef<str>>(outputs_: &[S], schema: &Value, mode: JsonMode) -> Result<SchemaCompliance> {
    if !schema.is_object() {
        return invalid("schema must be a JSON Schema object (a dict).");
    }
    outputs(outputs_)?;
    let (mut ok, mut errors) = (Vec::new(), Vec::new());
    for (i, o) in outputs_.iter().enumerate() {
        match extract_json(o.as_ref(), mode) {
            None => {
                ok.push(false);
                errors.push((i, "not valid JSON".to_owned()));
            }
            Some(v) => {
                let e = validate(&v, schema)?;
                ok.push(e.is_empty());
                if let Some(first) = e.into_iter().next() {
                    errors.push((i, first));
                }
            }
        }
    }
    Ok(SchemaCompliance { flags: flags(ok), errors })
}

/// Share of outputs that are well-formed XML (documents with a DOCTYPE or ENTITY declaration are rejected),
/// optionally with the given root tag (`{namespace}local` for a namespaced root).
pub fn xml_validity<S: AsRef<str>>(outputs_: &[S], root_tag: Option<&str>) -> Result<Flags> {
    outputs(outputs_)?;
    Ok(flags(
        outputs_
            .iter()
            .map(|o| {
                let text = py_strip(o.as_ref());
                let up = text.to_uppercase();
                if up.contains("<!DOCTYPE") || up.contains("<!ENTITY") {
                    return false;
                }
                match roxmltree::Document::parse(text) {
                    Err(_) => false,
                    Ok(doc) => root_tag.is_none_or(|want| {
                        let t = doc.root_element().tag_name();
                        let tag = match t.namespace() {
                            Some(ns) => format!("{{{ns}}}{}", t.name()),
                            None => t.name().to_owned(),
                        };
                        tag == want
                    }),
                }
            })
            .collect(),
    ))
}

fn get<'a>(obj: &'a Value, path: &str) -> Option<&'a Value> {
    let mut node = obj;
    for part in path.split('.') {
        node = match node {
            Value::Object(o) if o.contains_key(part) => &o[part],
            Value::Array(a) if !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()) => a.get(part.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(node)
}

/// Required-field accuracy: pooled (micro) and per-example (macro) share of expected fields found with the
/// expected value.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldAccuracy {
    pub micro: f64,
    pub macro_: f64,
    pub per_example: Vec<f64>,
    pub n_fields: usize,
}

/// A value-normalising function applied to both sides before comparison.
pub type Normalize<'a> = &'a dyn Fn(&Value) -> Value;

/// `expected[i]` maps dotted field paths (`"a.b.0"`) to values; `outputs[i]` is a model output — a JSON
/// string (parsed with `mode`) or an already parsed value.
pub fn required_field_accuracy(expected: &[Map<String, Value>], outputs_: &[Value], mode: JsonMode, normalize: Option<Normalize>) -> Result<FieldAccuracy> {
    if expected.is_empty() || expected.len() != outputs_.len() {
        return invalid("expected and outputs must be non-empty and the same length.");
    }
    let (mut hits, mut totals) = (Vec::new(), Vec::new());
    for (i, (e, o)) in expected.iter().zip(outputs_).enumerate() {
        if e.is_empty() {
            return invalid(format!("expected[{i}] must be a non-empty dict of field path -> value."));
        }
        let obj = match o {
            Value::String(s) => extract_json(s, mode),
            v => Some(v.clone()),
        };
        let mut n_ok = 0usize;
        if let Some(obj) = obj {
            for (path, want) in e {
                if let Some(got) = get(&obj, path) {
                    n_ok += usize::from(match normalize {
                        Some(f) => json_equal(&f(got), &f(want)),
                        None => json_equal(got, want),
                    });
                }
            }
        }
        hits.push(n_ok as f64);
        totals.push(e.len() as f64);
    }
    let per: Vec<f64> = hits.iter().zip(&totals).map(|(h, t)| h / t).collect();
    Ok(FieldAccuracy {
        micro: crate::util::np_sum(&hits) / crate::util::np_sum(&totals),
        macro_: crate::util::mean(&per),
        per_example: per,
        n_fields: totals.iter().sum::<f64>() as usize,
    })
}

fn pair(exp: &[Vec<ToolCall>], pred: &[Vec<ToolCall>]) -> Result<()> {
    if exp.is_empty() || exp.len() != pred.len() {
        return invalid("expected_calls and predicted_calls must be non-empty and the same length.");
    }
    Ok(())
}

/// Share of examples calling the right tools (as a multiset, or in order with `ordered`); arguments ignored.
pub fn tool_selection_accuracy(expected: &[Vec<ToolCall>], predicted: &[Vec<ToolCall>], ordered: bool) -> Result<Flags> {
    pair(expected, predicted)?;
    Ok(flags(
        expected
            .iter()
            .zip(predicted)
            .map(|(e, p)| {
                let mut en: Vec<&str> = e.iter().map(|c| c.name.as_str()).collect();
                let mut pn: Vec<&str> = p.iter().map(|c| c.name.as_str()).collect();
                if !ordered {
                    en.sort_unstable();
                    pn.sort_unstable();
                }
                en == pn
            })
            .collect(),
    ))
}

/// Share of expected calls matched by a predicted call to the same tool with equal arguments (each predicted
/// call used once).
pub fn tool_argument_accuracy(expected: &[Vec<ToolCall>], predicted: &[Vec<ToolCall>], normalize: Option<Normalize>) -> Result<f64> {
    pair(expected, predicted)?;
    let (mut hits, mut total) = (0usize, 0usize);
    for (e, p) in expected.iter().zip(predicted) {
        let mut pool: Vec<&ToolCall> = p.iter().collect();
        for call in e {
            total += 1;
            let k = pool.iter().position(|c| {
                c.name == call.name
                    && match normalize {
                        Some(f) => json_equal(&f(&c.arguments), &f(&call.arguments)),
                        None => json_equal(&c.arguments, &call.arguments),
                    }
            });
            if let Some(k) = k {
                hits += 1;
                pool.remove(k);
            }
        }
    }
    if total == 0 {
        return invalid("No expected tool calls to score.");
    }
    Ok(hits as f64 / total as f64)
}

/// How predicted calls are matched to expected ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CallMatch {
    /// Same tool and equal arguments.
    #[default]
    Arguments,
    /// Same tool.
    Name,
}

/// Tool-call F1 with its precision, recall and the share of predicted calls with unparseable arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallF1 {
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub invalid_call_rate: f64,
    pub n_expected: usize,
    pub n_predicted: usize,
}

/// Pooled precision / recall / F1 of predicted against expected tool calls.
pub fn tool_call_f1(expected: &[Vec<ToolCall>], predicted: &[Vec<ToolCall>], matching: CallMatch) -> Result<CallF1> {
    pair(expected, predicted)?;
    let (mut matched, mut n_pred, mut n_exp, mut bad) = (0usize, 0usize, 0usize, 0usize);
    for (e, p) in expected.iter().zip(predicted) {
        n_exp += e.len();
        n_pred += p.len();
        bad += p.iter().filter(|c| c.arguments.as_object().is_some_and(|o| o.contains_key("__unparseable__"))).count();
        let mut pool: Vec<&ToolCall> = p.iter().collect();
        for call in e {
            if let Some(k) = pool.iter().position(|c| c.name == call.name && (matching == CallMatch::Name || json_equal(&c.arguments, &call.arguments))) {
                matched += 1;
                pool.remove(k);
            }
        }
    }
    let precision = if n_pred > 0 { matched as f64 / n_pred as f64 } else { f64::from(u8::from(n_exp == 0)) };
    let recall = if n_exp > 0 { matched as f64 / n_exp as f64 } else { 1.0 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    Ok(CallF1 { f1, precision, recall, invalid_call_rate: if n_pred > 0 { bad as f64 / n_pred as f64 } else { 0.0 }, n_expected: n_exp, n_predicted: n_pred })
}

/// The outcome of one API call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiResult {
    /// Explicit success or failure.
    Success(bool),
    /// An HTTP status: 100–399 is a success.
    Status(i64),
    /// No response or an exception: a failure.
    Failed,
}

/// Share of successful API calls.
pub fn api_call_success_rate(results: &[ApiResult]) -> Result<Flags> {
    if results.is_empty() {
        return invalid("results is empty.");
    }
    Ok(flags(
        results
            .iter()
            .map(|r| match *r {
                ApiResult::Success(b) => b,
                ApiResult::Status(s) => (100..400).contains(&s),
                ApiResult::Failed => false,
            })
            .collect(),
    ))
}

fn check_matrix(checks: &[Vec<bool>], name: &str) -> Result<()> {
    if checks.is_empty() {
        return invalid(format!("{name} is empty."));
    }
    if let Some(i) = checks.iter().position(Vec::is_empty) {
        return invalid(format!("{name}[{i}] has no checks."));
    }
    Ok(())
}

/// Share of outputs passing all of their instruction checks (`checks[i]` = the checks of output i).
pub fn instruction_compliance_rate(checks: &[Vec<bool>]) -> Result<Flags> {
    check_matrix(checks, "checks")?;
    Ok(flags(checks.iter().map(|r| r.iter().all(|&x| x)).collect()))
}

/// Share of all individual constraint checks satisfied, pooled over outputs.
pub fn constraint_satisfaction_rate(checks: &[Vec<bool>]) -> Result<f64> {
    check_matrix(checks, "checks")?;
    let flat: Vec<bool> = checks.iter().flatten().copied().collect();
    Ok(flags(flat).value)
}

fn compile(pattern: &str) -> Result<Regex> {
    Regex::new(&format!("(?s){pattern}")).or_else(|e| invalid(format!("Invalid pattern: {e}")))
}

fn anchored(pattern: &str) -> Result<Regex> {
    compile(&format!(r"\A(?:{pattern})\z"))
}

/// Share of outputs (stripped) that fully match `pattern`.
pub fn format_compliance<S: AsRef<str>>(outputs_: &[S], pattern: &str) -> Result<Flags> {
    outputs(outputs_)?;
    compile(pattern)?;
    let re = anchored(pattern)?;
    Ok(flags(outputs_.iter().map(|o| re.is_match(py_strip(o.as_ref()))).collect()))
}

/// Share of later-turn checks still satisfied: `turn_checks[c][t]` = the checks of turn t of conversation c.
pub fn instruction_retention(turn_checks: &[Vec<Vec<bool>>]) -> Result<f64> {
    nonempty(turn_checks, "turn_checks")?;
    let flat: Vec<bool> = turn_checks.iter().flatten().flatten().copied().collect();
    if flat.is_empty() {
        return invalid("No later-turn checks to score.");
    }
    Ok(flags(flat).value)
}

/// What the requested payload is, for [`extra_content_rate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Payload<'a> {
    /// JSON: extra content means the output is not strict JSON but contains some (fenced or embedded).
    Json,
    /// A pattern: extra content means it matches a part of the output but not the whole (stripped) output.
    Regex(&'a str),
}

/// Share of outputs with text beyond the requested payload.
pub fn extra_content_rate<S: AsRef<str>>(outputs_: &[S], payload: Payload) -> Result<Flags> {
    outputs(outputs_)?;
    Ok(flags(match payload {
        Payload::Json => outputs_
            .iter()
            .map(|o| extract_json(o.as_ref(), JsonMode::Strict).is_none() && extract_json(o.as_ref(), JsonMode::Lenient).is_some())
            .collect(),
        Payload::Regex(p) => {
            if p.is_empty() {
                return invalid("payload must be 'json', or 'regex' together with a pattern.");
            }
            let (re, full) = (compile(p)?, anchored(p)?);
            outputs_.iter().map(|o| re.is_match(o.as_ref()) && !full.is_match(py_strip(o.as_ref()))).collect()
        }
    }))
}
