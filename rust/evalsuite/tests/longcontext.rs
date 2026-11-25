mod common;

use common::{check, nums, rows, strs, Got};
use evalsuite::longcontext::*;
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

fn brows(v: &Value) -> Vec<Vec<bool>> {
    v.as_array().unwrap().iter().map(bools).collect()
}

fn srows(v: &Value) -> Vec<Vec<String>> {
    v.as_array().unwrap().iter().map(strs).collect()
}

#[test]
fn longcontext_parity() {
    let (n, _) = check("longcontext", 1e-12, |f, a| {
        let key = f.split_once('.').map_or("value", |p| p.1);
        let num = Got::Num;
        match f.split('.').next().unwrap() {
            "retrieval_accuracy_by_length" => g(retrieval_accuracy_by_length(&bools(&a["c"]), &nums(&a["L"]), 0.6), |r| num(match key {
                "effective_length" => r.effective_length.unwrap_or(-1.0),
                "n_passing_lengths" => r.n_passing_lengths as f64,
                _ => r.value,
            })),
            "needle_in_haystack" => g(needle_in_haystack(&nums(&a["c"]), &nums(&a["L"]), &nums(&a["d"])), |r| num(if key == "worst_cell" { r.worst_cell } else { r.value })),
            "position_accuracy" => g(position_accuracy(&nums(&a["c"]), &nums(&a["pos"]), a["nb"].as_u64().unwrap() as usize), |r| num(r.0)),
            "lost_in_the_middle" => g(lost_in_the_middle(&nums(&a["c"]), &nums(&a["pos"]), 0.2), |r| num(match key {
                "start_accuracy" => r.start_accuracy,
                "middle_accuracy" => r.middle_accuracy,
                _ => r.value,
            })),
            "context_utilization" => g(context_utilization(&srows(&a["used"]), &srows(&a["prov"])), |r| num(if key == "mean_per_example" { r.1 } else { r.0 })),
            "summary_coverage" => g(summary_coverage(&brows(&a["kp"]), Some(&rows(&a["w"])), Some(&brows(&a["cl"]))), |r| num(match key {
                "factual_consistency" => r.factual_consistency.unwrap(),
                "fully_consistent_rate" => r.fully_consistent_rate.unwrap(),
                _ => r.value,
            })),
            "compression_ratio" => g(compression_ratio(&strs(&a["src"]), &strs(&a["summ"]), None), |r| num(if key == "median" { r.1 } else { r.0 })),
            "citation_coverage" => g(citation_coverage(&rows(&a["cps"])), |r| num(match key {
                "mean_per_output" => r.1,
                "ci_high" => r.0.ci_high,
                _ => r.0.value,
            })),
            "cross_document_consistency" => g(cross_document_consistency(&srows(&a["ans"]), None), |r| num(if key == "pairwise_agreement" { r.1 } else { r.0 })),
            _ => Got::Skip,
        }
    });
    assert!(n > 1500);
}
