mod common;

use common::{check, nums, rows, strs, Got};
use evalsuite::multilingual::*;
use serde_json::Value;

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> Got) -> Got {
    match r {
        Ok(x) => f(x),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

#[test]
fn multilingual_parity() {
    let (n, _) = check("multilingual", 1e-10, |f, a| {
        let parts: Vec<&str> = f.split('.').collect();
        let key = *parts.last().unwrap();
        let num = Got::Num;
        match parts[0] {
            "language_id_accuracy" => g(language_id_accuracy(&strs(&a["t"]), &strs(&a["p"])), |r| match key {
                "macro_f1" => num(r.macro_f1),
                "recall" => Got::List(r.recall.iter().map(|x| x.1).collect()),
                _ => num(r.value),
            }),
            "bitext_mining_accuracy" => {
                let k = a["k"].as_u64().unwrap() as usize;
                let sc = if parts[1].starts_with("margin") { Scoring::Margin(k) } else { Scoring::Cosine };
                g(bitext_mining_accuracy(&rows(&a["S"]), &rows(&a["T"]), sc), |r| num(match key {
                    "source_to_target" => r.source_to_target,
                    "mean_pair_cosine" => r.mean_pair_cosine,
                    _ => r.value,
                }))
            }
            "language_parity" => {
                let items: Vec<(String, Vec<f64>)> = a["sbl"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().to_owned(), nums(&p[1]))).collect();
                let refs: Vec<(&str, &[f64])> = items.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
                g(language_parity(&refs), |r| num(match key {
                    "max_gap" => r.max_gap,
                    "std" => r.std,
                    _ => r.value,
                }))
            }
            "code_switching_robustness" => g(code_switching_robustness(&bools(&a["m"]), &bools(&a["c"])), |r| num(match key {
                "absolute_drop" => r.absolute_drop,
                "flip_rate" => r.flip_rate,
                _ => r.value,
            })),
            "direct_assessment" => {
                let sy = strs(&a["y"]);
                g(direct_assessment(&nums(&a["s"]), &strs(&a["r"]), Some(&sy)), |r| if key == "by_system_z" { Got::List(r.by_system_z.iter().map(|x| x.1).collect()) } else { num(r.value) })
            }
            "cultural_appropriateness" => {
                let gr = strs(&a["g"]);
                g(cultural_appropriateness(&nums(&a["r"]), (1.0, 5.0), Some(&gr)), |r| num(if key == "region_gap" { r.2.unwrap() } else { r.0 }))
            }
            "language_consistency" => g(language_consistency(&strs(&a["t"]), &strs(&a["p"])), |r| num(if key == "ci_low" { r.ci_low } else { r.value })),
            "cross_lingual_consistency" => {
                let ans: Vec<Vec<String>> = a["ans"].as_array().unwrap().iter().map(strs).collect();
                g(cross_lingual_consistency(&ans, None), |r| num(if key == "pairwise_agreement" { r.1 } else { r.0 }))
            }
            _ => Got::Skip,
        }
    });
    assert!(n > 2000);
}
