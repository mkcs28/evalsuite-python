mod common;

use common::{check, strs, Got};
use evalsuite::qa::*;
use serde_json::Value;

fn refs(v: &Value) -> Vec<Vec<String>> {
    v.as_array().unwrap().iter().map(strs).collect()
}

fn codes(s: Option<String>) -> Got {
    Got::List(match s {
        None => vec![-1.0],
        Some(s) => s.chars().map(|c| c as u32 as f64).collect(),
    })
}

fn style(v: &Value) -> AnswerStyle {
    match v.as_str().unwrap() {
        "boxed" => AnswerStyle::Boxed,
        "choice" => AnswerStyle::Choice,
        "last_line" => AnswerStyle::LastLine,
        _ => AnswerStyle::Gsm8k,
    }
}

fn list(r: evalsuite::Result<Vec<f64>>) -> Got {
    match r {
        Ok(v) => Got::List(v),
        Err(e) => Got::Err(e.to_string()),
    }
}

#[test]
fn qa_parity() {
    let (n, _) = check("qa", 1e-12, |f, a| {
        let norm = || {
            if a["normalize"] == true {
                Norm::Squad
            } else {
                Norm::None
            }
        };
        match f {
            "normalize_answer" => codes(Some(normalize_answer(a["text"].as_str().unwrap()))),
            "extract_answer" => codes(extract_answer(
                a["text"].as_str().unwrap(),
                style(&a["style"]),
            )),
            "exact_match" => list(exact_match(
                &refs(&a["references"]),
                &strs(&a["predictions"]),
                norm(),
            )),
            "token_f1" => list(token_f1(
                &refs(&a["references"]),
                &strs(&a["predictions"]),
                norm(),
            )),
            "majority_vote_accuracy" => list(majority_vote_accuracy(
                &refs(&a["references"]),
                &refs(&a["samples"]),
                norm(),
            )),
            "pass_at_k" => {
                let u = |v: &Value| -> Vec<u64> {
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap())
                        .collect()
                };
                list(pass_at_k(
                    &u(&a["n_samples"]),
                    &u(&a["n_correct"]),
                    a["k"].as_u64().unwrap(),
                ))
            }
            "benchmark_accuracy" => list(
                benchmark_accuracy(
                    &strs(&a["references"]),
                    &strs(&a["outputs"]),
                    style(&a["style"]),
                    None,
                )
                .map(|b| {
                    b.per_example
                        .iter()
                        .map(|&h| f64::from(u8::from(h)))
                        .collect()
                }),
            ),
            _ => Got::Skip,
        }
    });
    assert!(n > 1000);
}
