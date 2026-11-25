mod common;

use common::{check, nums, strs, Got};
use evalsuite::code::*;
use serde_json::Value;

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> Got) -> Got {
    match r {
        Ok(x) => f(x),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect()
}

fn brows(v: &Value) -> Vec<Vec<bool>> {
    v.as_array().unwrap().iter().map(bools).collect()
}

#[test]
fn code_parity() {
    let (n, _) = check("code", 1e-10, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        let num = Got::Num;
        match name {
            "unit_test_pass_rate" => g(unit_test_pass_rate(&brows(&a["tr"])), |r| {
                num(if key == "strict_pass_rate" {
                    r.strict_pass_rate
                } else {
                    r.value
                })
            }),
            "static_analysis_violation_rate" => g(
                static_analysis_violation_rate(&nums(&a["v"]), &nums(&a["l"])),
                |r| num(r.0),
            ),
            "execution_success_rate" => g(
                execution_success_rate(&strs(&a["o"]), &["passed", "ok", "True"]),
                |r| {
                    if key == "by" {
                        Got::List(r.1.iter().map(|x| x.1).collect())
                    } else {
                        num(r.0.value)
                    }
                },
            ),
            "coverage_rate" => g(coverage_rate(&nums(&a["c"]), &nums(&a["t"])), |r| {
                num(if key == "mean_per_program" { r.1 } else { r.0 })
            }),
            "patch_acceptance_rate" => g(
                patch_acceptance_rate(None, Some(&bools(&a["fb"])), Some(&bools(&a["pa"]))),
                |r| num(r.value),
            ),
            "resolved_rate" => g(
                resolved_rate(&brows(&a["f"]), &brows(&a["p"]), Some(&bools(&a["a"]))),
                |r| {
                    num(match key {
                        "f2p_only_rate" => r.1,
                        "apply_rate" => r.2.unwrap(),
                        _ => r.0.value,
                    })
                },
            ),
            "security_vulnerability_rate" => {
                let fi: Vec<Vec<String>> = a["f"].as_array().unwrap().iter().map(strs).collect();
                g(
                    security_vulnerability_rate(&fi, a["m"].as_str().unwrap()),
                    |r| num(r.0.value),
                )
            }
            "runtime_efficiency" => {
                let (m, rm) = (nums(&a["m"]), nums(&a["rm"]));
                g(
                    runtime_efficiency(&nums(&a["t"]), &nums(&a["r"]), Some((&m, &rm))),
                    |r| {
                        num(match key {
                            "median_time_ratio" => r.median_time_ratio,
                            "share_slower_than_2x" => r.share_slower_than_2x,
                            "nmu" => r.nmu.unwrap(),
                            _ => r.value,
                        })
                    },
                )
            }
            "codebleu" => {
                let refs: Vec<Vec<String>> =
                    a["refs"].as_array().unwrap().iter().map(strs).collect();
                g(
                    codebleu(&refs, &strs(&a["preds"]), 0.0, 0.0, [0.25; 4], None, None),
                    |r| {
                        num(if key == "weighted_ngram_match" {
                            r.weighted_ngram_match
                        } else {
                            r.ngram_match
                        })
                    },
                )
            }
            _ => Got::Skip,
        }
    });
    assert!(n > 1500);
}
