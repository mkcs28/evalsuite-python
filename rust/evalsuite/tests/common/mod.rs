//! Shared helpers for the parity tests: load fixtures written by `scripts/rust_fixtures.py` and compare.
#![allow(dead_code)]

use serde_json::Value;

pub fn load(group: &str) -> Vec<Value> {
    let path = format!("{}/tests/fixtures/{group}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e} (run scripts/rust_fixtures.py)"));
    serde_json::from_str::<Vec<Value>>(&text).unwrap()
}

pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) if s == "nan" => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        Value::Bool(b) => f64::from(u8::from(*b)),
        other => panic!("not a number: {other}"),
    }
}

pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

pub fn opt_nums(v: &Value) -> Option<Vec<f64>> {
    if v.is_null() {
        None
    } else {
        Some(nums(v))
    }
}

pub fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(nums).collect()
}

pub fn bool_rows(v: &Value) -> Vec<Vec<bool>> {
    v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| num(x) != 0.0).collect()).collect()
}

pub fn ints(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

pub fn strs(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_owned()).collect()
}

/// A computed value: a number, or a (possibly nested) list of numbers flattened.
#[derive(Debug)]
pub enum Got {
    Num(f64),
    List(Vec<f64>),
    Err(String),
    /// The case is not covered by the Rust API (skipped, counted).
    Skip,
}

fn flatten(v: &Value, out: &mut Vec<f64>) {
    match v {
        Value::Array(a) => a.iter().for_each(|x| flatten(x, out)),
        x => out.push(num(x)),
    }
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= rtol * a.abs().max(b.abs()) + 1e-12
}

/// Replays every case of `group` through `run`; panics listing mismatches. Returns (checked, skipped).
pub fn check(group: &str, rtol: f64, run: impl Fn(&str, &Value) -> Got) -> (usize, usize) {
    let cases = load(group);
    let (mut checked, mut skipped) = (0, 0);
    let mut bad = Vec::new();
    for c in &cases {
        let f = c["fn"].as_str().unwrap();
        let got = run(f, &c["args"]);
        let exp = &c["expected"];
        let ok = match (&got, exp) {
            (Got::Skip, _) => {
                skipped += 1;
                continue;
            }
            (Got::Err(_), Value::Object(o)) if o.contains_key("error") => true,
            (_, Value::Object(o)) if o.contains_key("error") => false,
            (Got::Err(_), _) => false,
            (Got::Num(x), e) if !e.is_array() => close(*x, num(e), rtol),
            (Got::List(x), e) if e.is_array() => {
                let mut flat = Vec::new();
                flatten(e, &mut flat);
                flat.len() == x.len() && x.iter().zip(&flat).all(|(a, b)| close(*a, *b, rtol))
            }
            _ => false,
        };
        checked += 1;
        if !ok && bad.len() < 15 {
            let args = c["args"].to_string();
            bad.push(format!("{f}: got {got:?}, expected {exp}  args {}", &args[..args.len().min(300)]));
        }
    }
    eprintln!("{group}: {checked} checked, {skipped} skipped");
    assert!(bad.is_empty(), "{} mismatches in {group}:\n{}", bad.len(), bad.join("\n"));
    (checked, skipped)
}
