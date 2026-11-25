mod common;

use common::*;
use evalsuite::classification::Score;
use evalsuite::regression::{self as r, multioutput::{self as m, Multioutput}};

#[test]
fn regression_matches_python() {
    let (checked, skipped) = check("regression", 1e-9, |f, a| {
        let to = |x: evalsuite::Result<f64>| match x {
            Ok(v) => Got::Num(v),
            Err(e) => Got::Err(e.to_string()),
        };
        if let Some(rest) = f.strip_prefix("multioutput.") {
            let (t, p) = (rows(&a["y_true"]), rows(&a["y_pred"]));
            let how = match &a["multioutput"] {
                serde_json::Value::String(s) if s == "raw_values" => Multioutput::RawValues,
                serde_json::Value::String(_) => Multioutput::UniformAverage,
                v => Multioutput::Weights(nums(v)),
            };
            let s = match rest {
                "mae" => m::mae(&t, &p, None, &how),
                "rmse" => m::rmse(&t, &p, None, &how),
                "r2" => m::r2(&t, &p, None, &how),
                "msle" => m::msle(&t, &p, None, &how),
                _ => return Got::Skip,
            };
            return match s {
                Ok(Score::Scalar(v)) => Got::Num(v),
                Ok(Score::PerClass(v)) => Got::List(v),
                Err(e) => Got::Err(e.to_string()),
            };
        }
        let (t, p) = (nums(&a["y_true"]), nums(&a["y_pred"]));
        let w = opt_nums(&a["sample_weight"]);
        let w = w.as_deref();
        to(match f {
            "mae" => r::mae(&t, &p, w),
            "mse" => r::mse(&t, &p, w),
            "rmse" => r::rmse(&t, &p, w),
            "r2" => r::r2(&t, &p, w),
            "mape" => r::mape(&t, &p, w),
            "smape" => r::smape(&t, &p, w),
            "msle" => r::msle(&t, &p, w),
            "rmsle" => r::rmsle(&t, &p, w),
            "median_absolute_error" => r::median_absolute_error(&t, &p, w),
            "explained_variance" => r::explained_variance(&t, &p, w),
            "mean_bias_error" => r::mean_bias_error(&t, &p, w),
            "rae" => r::rae(&t, &p, w),
            "rse" => r::rse(&t, &p, w),
            "max_error" => r::max_error(&t, &p),
            "adjusted_r2" => r::adjusted_r2(&t, &p, 3, w),
            "quantile_loss" => r::quantile_loss(&t, &p, num(&a["alpha"]), w),
            "huber_loss" => r::huber_loss(&t, &p, num(&a["delta"]), w),
            _ => return Got::Skip,
        })
    });
    assert!(checked > 700 && skipped == 0);
}
