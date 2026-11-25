mod common;

use common::{check, nums, rows, strs, Got};
use evalsuite::robustness::*;
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

fn srows(v: &Value) -> Vec<Vec<String>> {
    v.as_array().unwrap().iter().map(strs).collect()
}

#[test]
fn robustness_parity() {
    let (n, _) = check("robustness", 1e-10, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        let num = Got::Num;
        let drop = |r: Drop| num(match key {
            "relative_drop" => r.relative_drop,
            "flip_rate" => r.flip_rate,
            _ => r.value,
        });
        match name {
            "adversarial_robustness" => g(adversarial_robustness(&bools(&a["c"]), &bools(&a["p"])), drop),
            "noise_robustness" => g(noise_robustness(&bools(&a["c"]), &bools(&a["p"])), drop),
            "paraphrase_consistency" => g(paraphrase_consistency(&srows(&a["ans"]), None), |r| num(if key == "pairwise_agreement" { r.1 } else { r.0 })),
            "response_stability" => g(response_stability(&srows(&a["ans"]), None), |r| num(match key {
                "unanimous_rate" => r.unanimous_rate,
                "modal_share" => r.modal_share,
                _ => r.value,
            })),
            "ood_accuracy" => g(ood_accuracy(&bools(&a["c"]), &bools(&a["o"])), |r| num(if key == "gap" { r.2 } else { r.0 })),
            "distribution_shift_drop" => g(distribution_shift_drop(&nums(&a["s"]), &nums(&a["t"])), |r| num(match key {
                "relative_drop" => r.relative_drop,
                "ci_low" => r.ci_low,
                "ci_high" => r.ci_high,
                _ => r.value,
            })),
            "invariance_violation_rate" => {
                let gr = strs(&a["g"]);
                g(invariance_violation_rate(&strs(&a["a"]), &strs(&a["b"]), None, Some(&gr)), |r| if key == "by_group" { Got::List(r.1.iter().map(|x| x.1).collect()) } else { num(r.0.value) })
            }
            "contradiction_rate" => g(contradiction_rate(&srows(&a["pl"])), |r| num(r.value)),
            "error_rate" => g(error_rate(&strs(&a["st"]), &["ok", "success", "200", "True"]), |r| match key {
                "timeout_rate" => num(r.timeout_rate),
                "by_status" => Got::List(r.by_status.iter().map(|x| x.1).collect()),
                _ => num(r.rate.value),
            }),
            "recovery_success_rate" => g(recovery_success_rate(&bools(&a["f"]), &bools(&a["r"])), |r| num(if key == "failure_rate" { r.1 } else { r.0.value })),
            "prompt_sensitivity" => g(prompt_sensitivity(&rows(&a["tm"])), |r| num(match key {
                "std" => r.std,
                "example_disagreement_rate" => r.example_disagreement_rate,
                "mean" => r.mean,
                _ => r.value,
            })),
            "truncation_sensitivity" => g(truncation_sensitivity(&nums(&a["c"]), &nums(&a["l"])), |r| num(if key == "drop_shortest_to_longest" { r.2 } else { r.0 })),
            _ => Got::Skip,
        }
    });
    assert!(n > 2500);
}

#[test]
fn typos_are_reproducible() {
    let t = ["The quick brown fox jumps over the lazy dog"];
    assert_eq!(add_typos(&t, 0.0, 1).unwrap()[0], t[0]);
    let a = add_typos(&t, 0.3, 7).unwrap();
    assert_eq!(a, add_typos(&t, 0.3, 7).unwrap());
    assert_ne!(a[0], t[0]);
    assert!(add_typos(&t, 1.5, 0).is_err());
}
