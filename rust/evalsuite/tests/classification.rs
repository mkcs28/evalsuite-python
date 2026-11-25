mod common;

use common::*;
use evalsuite::classification::{
    self as c, binary, multilabel, Average, KappaWeights, MultiClass, Normalize, Opts, Score,
};
use serde_json::Value;

fn avg(a: &Value) -> Average {
    match a.as_str().unwrap_or("none") {
        "binary" => Average::Binary,
        "micro" => Average::Micro,
        "macro" => Average::Macro,
        "weighted" => Average::Weighted,
        "samples" => Average::Samples,
        _ => Average::None,
    }
}

fn score(r: evalsuite::Result<Score>) -> Got {
    match r {
        Ok(Score::Scalar(v)) => Got::Num(v),
        Ok(Score::PerClass(v)) => Got::List(v),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn num_r(r: evalsuite::Result<f64>) -> Got {
    match r {
        Ok(v) => Got::Num(v),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn lists(r: evalsuite::Result<(Vec<f64>, Vec<f64>, Vec<f64>)>) -> Got {
    match r {
        Ok((a, b, d)) => Got::List([a, b, d].concat()),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn run_labels<L: evalsuite::Label>(
    f: &str,
    a: &Value,
    yt: Vec<L>,
    yp: Option<Vec<L>>,
    pos: Option<L>,
) -> Got {
    let w = opt_nums(&a["sample_weight"]);
    let wr = w.as_deref();
    let mut o: Opts<L> = Opts::default().average(avg(&a["average"]));
    if a["average"].is_null() {
        o = o.average(Average::Auto);
    }
    if let Some(w) = &w {
        o = o.sample_weight(w.clone());
    }
    if let Some(p) = pos {
        o = o.pos_label(p);
    }
    let yp_ = yp.as_deref().unwrap_or(&[]);
    match f {
        "precision" => score(c::precision(&yt, yp_, &o)),
        "recall" => score(c::recall(&yt, yp_, &o)),
        "f1" => score(c::f1(&yt, yp_, &o)),
        "specificity" => score(c::specificity(&yt, yp_, &o)),
        "npv" => score(c::npv(&yt, yp_, &o)),
        "jaccard" => score(c::jaccard(&yt, yp_, &o)),
        "fbeta" => score(c::fbeta(&yt, yp_, num(&a["beta"]), &o)),
        "accuracy" => num_r(c::accuracy(&yt, yp_, wr)),
        "hamming_loss" => num_r(c::hamming_loss(&yt, yp_, wr)),
        "mcc" => num_r(c::mcc(&yt, yp_, wr, Default::default())),
        "balanced_accuracy" => num_r(c::balanced_accuracy(
            &yt,
            yp_,
            a["adjusted"].as_bool().unwrap(),
            wr,
        )),
        "cohen_kappa" => {
            let k = match a["weights"].as_str().unwrap() {
                "linear" => KappaWeights::Linear,
                "quadratic" => KappaWeights::Quadratic,
                _ => KappaWeights::Unweighted,
            };
            num_r(c::cohen_kappa(&yt, yp_, k, None, wr, Default::default()))
        }
        "confusion_matrix" => match c::confusion_matrix(&yt, yp_, None, wr, Normalize::True) {
            Ok((m, _)) => Got::List(m.concat()),
            Err(e) => Got::Err(e.to_string()),
        },
        _ => Got::Skip,
    }
}

#[test]
fn classification_matches_python() {
    let (checked, skipped) = check("classification", 1e-9, |f, a| {
        let w = opt_nums(&a["sample_weight"]);
        let wr = w.as_deref();
        if let Some(rest) = f.strip_prefix("multilabel.") {
            let t = bool_rows(&a["y_true"]);
            return match rest {
                "roc_auc" => score(multilabel::roc_auc(
                    &t,
                    &rows(&a["y_prob"]),
                    avg(&a["average"]),
                    wr,
                )),
                "average_precision" => score(multilabel::average_precision(
                    &t,
                    &rows(&a["y_prob"]),
                    avg(&a["average"]),
                    wr,
                )),
                _ => {
                    let p = bool_rows(&a["y_pred"]);
                    let (av, z) = (avg(&a["average"]), Default::default());
                    match rest {
                        "precision" => score(multilabel::precision(&t, &p, av, wr, z)),
                        "recall" => score(multilabel::recall(&t, &p, av, wr, z)),
                        "f1" => score(multilabel::f1(&t, &p, av, wr, z)),
                        "jaccard" => score(multilabel::jaccard(&t, &p, av, wr, z)),
                        "accuracy" => num_r(multilabel::accuracy(&t, &p, wr)),
                        "hamming_loss" => num_r(multilabel::hamming_loss(&t, &p, wr)),
                        _ => Got::Skip,
                    }
                }
            };
        }
        if a["y_true"]
            .as_array()
            .unwrap()
            .first()
            .is_some_and(Value::is_string)
        {
            let pos = a["pos_label"].as_str().map(str::to_owned);
            return run_labels(f, a, strs(&a["y_true"]), Some(strs(&a["y_pred"])), pos);
        }
        let yt = ints(&a["y_true"]);
        if let Some(rest) = f.strip_prefix("binary.") {
            let s = nums(&a["y_prob"]);
            let strategy = if a["strategy"].as_str() == Some("quantile") {
                binary::Strategy::Quantile
            } else {
                binary::Strategy::Uniform
            };
            return match rest {
                "roc_auc" => num_r(binary::roc_auc(&yt, &s, None, wr)),
                "average_precision" => num_r(binary::average_precision(&yt, &s, None, wr)),
                "log_loss" => num_r(binary::log_loss(&yt, &s, None, wr)),
                "brier_score" => num_r(binary::brier_score(&yt, &s, None, wr)),
                "roc_curve" => lists(binary::roc_curve(&yt, &s, None, wr)),
                "pr_curve" => lists(binary::pr_curve(&yt, &s, None, wr)),
                "expected_calibration_error" => num_r(binary::expected_calibration_error(
                    &yt, &s, 7, strategy, None, wr,
                )),
                "calibration_curve" => {
                    lists(binary::calibration_curve(&yt, &s, 7, strategy, None, wr))
                }
                _ => Got::Skip,
            };
        }
        if !a["y_prob"].is_null() {
            let p = rows(&a["y_prob"]);
            let mc = if a["multi_class"].as_str() == Some("ovo") {
                MultiClass::Ovo
            } else {
                MultiClass::Ovr
            };
            return match f {
                "roc_auc" => score(c::roc_auc(&yt, &p, avg(&a["average"]), mc, None, wr)),
                "average_precision" => {
                    score(c::average_precision(&yt, &p, avg(&a["average"]), None, wr))
                }
                "log_loss" => num_r(c::log_loss(&yt, &p, None, wr)),
                "brier_score" => num_r(c::brier_score(&yt, &p, None, wr)),
                "top_k_accuracy" => num_r(c::top_k_accuracy(&yt, &p, 2, None, wr)),
                _ => Got::Skip,
            };
        }
        run_labels(f, a, yt, Some(ints(&a["y_pred"])), None)
    });
    assert!(
        checked > 2000 && skipped == 0,
        "checked {checked}, skipped {skipped}"
    );
}
