mod common;

use common::{check, nums, rows, strs, Got};
use evalsuite::judge::*;
use serde_json::Value;

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> Got) -> Got {
    match r {
        Ok(x) => f(x),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn comps(v: &Value) -> Vec<(String, String, f64)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r[0].as_str().unwrap().into(),
                r[1].as_str().unwrap().into(),
                r[2].as_f64().unwrap(),
            )
        })
        .collect()
}

fn verdicts(v: &Value) -> Vec<Verdict> {
    strs(v).iter().map(|s| Verdict::parse(s).unwrap()).collect()
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect()
}

#[test]
fn judge_parity() {
    let (n, _) = check("judge", 1e-9, |f, a| {
        let key = f.rsplit('.').next().unwrap();
        let name = f.split('.').next().unwrap();
        match name {
            "win_rate" => {
                let t = if a["ties"] == "half" {
                    Ties::Half
                } else {
                    Ties::Exclude
                };
                g(win_rate(&nums(&a["outcomes"]), t), |r| {
                    Got::Num(match key {
                        "ci_low" => r.ci_low,
                        "ci_high" => r.ci_high,
                        _ => r.value,
                    })
                })
            }
            "bradley_terry" => {
                let s = if a["scale"] == "log" {
                    BtScale::Log
                } else {
                    BtScale::Elo
                };
                g(
                    bradley_terry(&comps(&a["comparisons"]), s, 10_000, 1e-10),
                    |r| Got::List(r.values),
                )
            }
            "elo_ratings" => g(
                elo_ratings(
                    &comps(&a["comparisons"]),
                    a["k"].as_f64().unwrap(),
                    1000.0,
                    400.0,
                    10.0,
                ),
                |r| Got::List(r.values),
            ),
            "krippendorff_alpha" => {
                let lv = match a["level"].as_str().unwrap() {
                    "ordinal" => Level::Ordinal,
                    "interval" => Level::Interval,
                    "ratio" => Level::Ratio,
                    _ => Level::Nominal,
                };
                g(krippendorff_alpha(&rows(&a["ratings"]), lv), Got::Num)
            }
            "fleiss_kappa" => {
                let r: Vec<Vec<i64>> = a["ratings"].as_array().unwrap().iter().map(ints).collect();
                g(fleiss_kappa(&r), Got::Num)
            }
            "judge_agreement" => {
                let r = if f.contains("categorical") {
                    judge_agreement(&ints(&a["judge"]), &ints(&a["human"]))
                } else if f.contains("ordinal") {
                    judge_agreement_ordinal(&ints(&a["judge"]), &ints(&a["human"]))
                } else {
                    judge_agreement_continuous(&nums(&a["judge"]), &nums(&a["human"]))
                };
                g(r, |r| {
                    Got::Num(if key == "spearman" {
                        r.spearman.unwrap()
                    } else {
                        r.value
                    })
                })
            }
            "position_consistency" => g(
                position_consistency(&verdicts(&a["original"]), &verdicts(&a["swapped"])),
                |r| {
                    Got::Num(match key {
                        "first_position_rate" => r.first_position_rate,
                        "second_position_rate" => r.second_position_rate,
                        _ => r.value,
                    })
                },
            ),
            "verbosity_bias" => g(
                verbosity_bias(
                    &verdicts(&a["winners"]),
                    &nums(&a["length_a"]),
                    &nums(&a["length_b"]),
                ),
                |r| Got::Num(if key == "p_value" { r.2 } else { r.0 }),
            ),
            "self_preference_bias" => g(
                self_preference_bias(&bools(&a["judge"]), &bools(&a["human"])),
                |r| Got::Num(if key == "p_value" { r.p_value } else { r.value }),
            ),
            "rubric_score" => g(rubric_score(&rows(&a["scores"]), (1.0, 5.0)), |r| {
                if key == "per_criterion" {
                    Got::List(r.per_criterion)
                } else {
                    Got::Num(r.value)
                }
            }),
            _ => Got::Skip,
        }
    });
    assert!(n > 1000);
}
