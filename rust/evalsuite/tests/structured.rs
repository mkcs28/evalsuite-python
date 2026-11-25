mod common;

use common::{check, strs, Got};
use evalsuite::structured::*;
use serde_json::Value;

fn calls(v: &Value) -> Vec<Vec<ToolCall>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|t| {
            t.as_array()
                .unwrap()
                .iter()
                .map(|c| ToolCall::from_json(c).unwrap())
                .collect()
        })
        .collect()
}

fn bools(v: &Value) -> Vec<Vec<bool>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_bool().unwrap())
                .collect()
        })
        .collect()
}

fn mode(v: &Value) -> JsonMode {
    match v.as_str() {
        Some("fenced") => JsonMode::Fenced,
        Some("lenient") => JsonMode::Lenient,
        _ => JsonMode::Strict,
    }
}

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> f64) -> Got {
    match r {
        Ok(x) => Got::Num(f(x)),
        Err(e) => Got::Err(e.to_string()),
    }
}

#[test]
fn structured_parity() {
    let (n, _) = check("structured", 1e-12, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        let outs = || strs(&a["outputs"]);
        match name {
            "json_validity" => g(json_validity(&outs(), mode(&a["mode"])), |r| r.value),
            "json_schema_compliance" => g(
                json_schema_compliance(&outs(), &a["schema"], mode(&a["mode"])),
                |r| r.flags.value,
            ),
            "xml_validity" => g(xml_validity(&outs(), a["root_tag"].as_str()), |r| r.value),
            "extra_content_rate" => {
                let p = if key == "regex" {
                    Payload::Regex(a["pattern"].as_str().unwrap())
                } else {
                    Payload::Json
                };
                g(extra_content_rate(&outs(), p), |r| r.value)
            }
            "format_compliance" => g(
                format_compliance(&outs(), a["pattern"].as_str().unwrap()),
                |r| r.value,
            ),
            "required_field_accuracy" => {
                let e: Vec<_> = a["expected"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_object().unwrap().clone())
                    .collect();
                let o: Vec<Value> = a["outputs"].as_array().unwrap().clone();
                g(
                    required_field_accuracy(&e, &o, mode(&a["mode"]), None),
                    |r| {
                        if a["average"] == "micro" {
                            r.micro
                        } else {
                            r.macro_
                        }
                    },
                )
            }
            "tool_selection_accuracy" => g(
                tool_selection_accuracy(
                    &calls(&a["expected"]),
                    &calls(&a["predicted"]),
                    a["ordered"].as_bool().unwrap(),
                ),
                |r| r.value,
            ),
            "tool_argument_accuracy" => g(
                tool_argument_accuracy(&calls(&a["expected"]), &calls(&a["predicted"]), None),
                |r| r,
            ),
            "tool_call_f1" => {
                let m = if a["match"] == "name" {
                    CallMatch::Name
                } else {
                    CallMatch::Arguments
                };
                g(
                    tool_call_f1(&calls(&a["expected"]), &calls(&a["predicted"]), m),
                    |r| match key {
                        "precision" => r.precision,
                        "recall" => r.recall,
                        "invalid_call_rate" => r.invalid_call_rate,
                        _ => r.f1,
                    },
                )
            }
            "api_call_success_rate" => {
                let r: Vec<ApiResult> = a["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| match x {
                        Value::Bool(b) => ApiResult::Success(*b),
                        Value::Number(n) => ApiResult::Status(n.as_i64().unwrap()),
                        _ => ApiResult::Failed,
                    })
                    .collect();
                g(api_call_success_rate(&r), |r| r.value)
            }
            "instruction_compliance_rate" => {
                g(instruction_compliance_rate(&bools(&a["checks"])), |r| {
                    r.value
                })
            }
            "constraint_satisfaction_rate" => {
                g(constraint_satisfaction_rate(&bools(&a["checks"])), |r| r)
            }
            "instruction_retention" => {
                let t: Vec<Vec<Vec<bool>>> = a["turn_checks"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(bools)
                    .collect();
                g(instruction_retention(&t), |r| r)
            }
            _ => Got::Skip,
        }
    });
    assert!(n > 2500);
}
