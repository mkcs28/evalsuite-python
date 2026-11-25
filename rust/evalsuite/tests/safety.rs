mod common;

use common::{check, nums, rows, strs, Got};
use evalsuite::safety::*;
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

fn vals(v: &[(String, f64)]) -> Got {
    Got::List(v.iter().map(|x| x.1).collect())
}

#[test]
fn safety_parity() {
    let (n, _) = check("safety", 1e-10, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        let num = Got::Num;
        match name {
            "detect_pii" => g(detect_pii(a["text"].as_str().unwrap(), None), |r| {
                let s: Vec<String> = r
                    .iter()
                    .map(|(k, v)| format!("{k}:{}", v.join("\u{0}")))
                    .collect();
                let s = s.join("|");
                Got::List(if s.is_empty() {
                    vec![-1.0]
                } else {
                    s.chars().map(|c| c as u32 as f64).collect()
                })
            }),
            "harmful_response_rate" => {
                let c = strs(&a["c"]);
                g(
                    harmful_response_rate(&bools(&a["h"]), Some(&bools(&a["hp"])), Some(&c)),
                    |r| match key {
                        "ci_high" => num(r.0.ci_high),
                        "by_category" => vals(&r.1),
                        _ => num(r.0.value),
                    },
                )
            }
            "refusal_rate" => g(refusal_rate(&bools(&a["r"]), Some(&bools(&a["s"]))), |r| {
                num(match key {
                    "precision" => r.precision.unwrap(),
                    "f1" => r.f1.unwrap(),
                    "overall_refusal_rate" => r.overall_refusal_rate.unwrap(),
                    _ => r.rate.value,
                })
            }),
            "over_refusal_rate" => g(over_refusal_rate(&bools(&a["r"]), &bools(&a["s"])), |r| {
                num(r.value)
            }),
            "attack_success_rate" => {
                let c = strs(&a["c"]);
                g(attack_success_rate(&bools(&a["h"]), Some(&c)), |r| {
                    vals(&r.1)
                })
            }
            "red_team_success_rate" => {
                let att: Vec<Vec<bool>> = a["att"].as_array().unwrap().iter().map(bools).collect();
                let k = a["k"].as_u64().map(|k| k as usize);
                g(red_team_success_rate::<String>(&att, k, None), |r| {
                    num(if key == "per_attempt_rate" {
                        r.1
                    } else {
                        r.0.value
                    })
                })
            }
            "toxicity_score" => {
                let hate = rows(&a["hate"]);
                g(
                    toxicity_score(&rows(&a["tox"]), 0.5, &[("hate", &hate)]),
                    |r| {
                        num(match key {
                            "toxicity_probability" => r.toxicity_probability,
                            "by_attribute_hate" => r.by_attribute[0].1,
                            _ => r.value,
                        })
                    },
                )
            }
            "stereotype_preference" => {
                let c = strs(&a["c"]);
                g(
                    stereotype_preference(&nums(&a["s"]), &nums(&a["a"]), Some(&c)),
                    |r| {
                        num(match key {
                            "ci_low" => r.ci_low,
                            "distance_from_unbiased" => r.distance_from_unbiased,
                            _ => r.value,
                        })
                    },
                )
            }
            "weat_effect_size" => g(
                weat_effect_size(
                    &rows(&a["X"]),
                    &rows(&a["Y"]),
                    &rows(&a["A"]),
                    &rows(&a["B"]),
                    100_000,
                    0,
                ),
                |r| {
                    num(match key {
                        "p_value" => r.1,
                        "test_statistic" => r.2,
                        _ => r.0,
                    })
                },
            ),
            "pii_leakage_rate" => {
                g(
                    pii_leakage_rate(&strs(&a["o"]), None, &["secret"]),
                    |r| match key {
                        "protected_leak_rate" => num(r.protected_leak_rate),
                        "by_kind" => vals(&r.by_kind),
                        _ => num(r.rate.value),
                    },
                )
            }
            "exposure" => g(
                exposure(&nums(&a["can"]), &rows(&a["cand"]), a["space"].as_f64()),
                |r| num(r.0),
            ),
            "policy_violation_rate" => {
                let v: Vec<Vec<String>> = a["v"].as_array().unwrap().iter().map(strs).collect();
                g(
                    policy_violation_rate(&v, Some(&["hate", "pii", "medical", "other"])),
                    |r| {
                        if key == "by" {
                            vals(&r.1)
                        } else {
                            num(r.0.value)
                        }
                    },
                )
            }
            _ => Got::Skip,
        }
    });
    assert!(n > 3000);
}
