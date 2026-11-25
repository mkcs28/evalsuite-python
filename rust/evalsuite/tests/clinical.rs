mod common;

use common::*;
use evalsuite::classification::binary::Strategy;
use evalsuite::{calibration as cal, clinical as cl, uncertainty as u};

fn r(x: evalsuite::Result<f64>) -> Got {
    x.map_or_else(|e| Got::Err(e.to_string()), Got::Num)
}

fn bools(v: &serde_json::Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

#[test]
fn clinical_matches_python() {
    let (n, s) = check("clinical", 1e-9, |f, a| {
        let y = ints(&a["y_true"]);
        let w = opt_nums(&a["sample_weight"]);
        let w = w.as_deref();
        let z = Default::default();
        if f == "net_benefit" {
            return r(cl::net_benefit(&y, &nums(&a["y_prob"]), num(&a["threshold"]), None, w));
        }
        if f == "decision_curve" {
            return match cl::decision_curve(&y, &nums(&a["y_prob"]), &nums(&a["thresholds"]), None, w) {
                Ok((m, all)) => Got::List([m, all].concat()),
                Err(e) => Got::Err(e.to_string()),
            };
        }
        let p = ints(&a["y_pred"]);
        r(match f {
            "sensitivity" => cl::sensitivity(&y, &p, None, w, z),
            "ppv" => cl::ppv(&y, &p, None, w, z),
            "lr_positive" => cl::lr_positive(&y, &p, None, w),
            "lr_negative" => cl::lr_negative(&y, &p, None, w),
            "youden_j" => cl::youden_j(&y, &p, None, w),
            "diagnostic_odds_ratio" => cl::diagnostic_odds_ratio(&y, &p, None, w, num(&a["correction"])),
            _ => return Got::Skip,
        })
    });
    assert!(n > 400 && s == 0);
}

#[test]
fn calibration_matches_python() {
    let (n, s) = check("calibration", 1e-8, |f, a| {
        let y = ints(&a["y_true"]);
        let p = nums(&a["y_prob"]);
        let w = opt_nums(&a["sample_weight"]);
        let w = w.as_deref();
        match f {
            "maximum_calibration_error" => {
                let st = if a["strategy"].as_str() == Some("quantile") { Strategy::Quantile } else { Strategy::Uniform };
                r(cal::maximum_calibration_error(&y, &p, 8, st, None, w))
            }
            "calibration_slope" => cal::calibration_slope(&y, &p, None, w).map_or_else(|e| Got::Err(e.to_string()), |(a, b)| Got::List(vec![a, b])),
            "calibration_intercept" => r(cal::calibration_intercept(&y, &p, None, w)),
            "hosmer_lemeshow" => cal::hosmer_lemeshow(&y, &p, 8, None).map_or_else(|e| Got::Err(e.to_string()), |(a, b)| Got::List(vec![a, b])),
            _ => Got::Skip,
        }
    });
    assert!(n > 100 && s == 0);
}

#[test]
fn uncertainty_matches_python() {
    let (n, s) = check("uncertainty", 1e-10, |f, a| {
        let c = bools(&a["correct"]);
        let p = nums(&a["confidence"]);
        match f {
            "adaptive_calibration_error" => r(u::adaptive_calibration_error(&c, &p, 10)),
            "selective_risk" => r(u::selective_risk(&c, &p, num(&a["coverage"]))),
            "risk_at_coverage" => r(u::risk_at_coverage(&c, &p, num(&a["coverage"]))),
            "coverage_at_risk" => r(u::coverage_at_risk(&c, &p, num(&a["risk"]))),
            "aurc" => u::aurc(&c, &p).map_or_else(|e| Got::Err(e.to_string()), |(a, b)| Got::List(vec![a, b])),
            "confidence_accuracy_correlation" => r(u::confidence_accuracy_correlation(&c, &p)),
            "risk_coverage_curve" => u::risk_coverage_curve(&c, &p).map_or_else(|e| Got::Err(e.to_string()), |(a, b, d)| Got::List([a, b, d].concat())),
            _ => Got::Skip,
        }
    });
    assert!(n > 500 && s == 0);
}
