#![cfg(feature = "python")]
mod common;

use common::{check, strs, Got};
use evalsuite::code::*;
use serde_json::Value;

fn refs(v: &Value) -> Vec<Vec<String>> {
    v.as_array().unwrap().iter().map(strs).collect()
}

#[test]
fn pycode_parity() {
    let (n, _) = check("pycode", 1e-12, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        let src = || a["src"].as_str().unwrap();
        match name {
            "syntax_ok" => Got::Num(f64::from(u8::from(python_syntax_ok(src())))),
            "parses" => Got::Num(f64::from(u8::from(__python_parses_stripped(src())))),
            "strip" => {
                let s = strip_python_comments(src());
                Got::List(if s.is_empty() {
                    vec![-1.0]
                } else {
                    s.chars().map(|c| c as u32 as f64).collect()
                })
            }
            "code_complexity" => match code_complexity(&strs(&a["code"])) {
                Err(e) => Got::Err(e.to_string()),
                Ok(r) => Got::Num(match key {
                    "max_complexity" => r.max_complexity as f64,
                    "maintainability_index" => r.maintainability_index,
                    "total" => r.per_program[0].total_complexity as f64,
                    _ => r.value,
                }),
            },
            "codebleu_python" => {
                match codebleu_python(&refs(&a["refs"]), &strs(&a["preds"]), [0.25; 4], None, None)
                {
                    Err(e) => Got::Err(e.to_string()),
                    Ok(r) => Got::Num(match key {
                        "ngram_match" => r.ngram_match,
                        "weighted_ngram_match" => r.weighted_ngram_match,
                        "syntax_match" => r.syntax_match,
                        "dataflow_match" => r.dataflow_match,
                        _ => r.value,
                    }),
                }
            }
            _ => Got::Skip,
        }
    });
    assert!(n > 400);
}
