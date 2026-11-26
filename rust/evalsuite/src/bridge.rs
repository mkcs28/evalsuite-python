//! A JSON-in / JSON-out entry point used by the WebAssembly (npm), C, R and Julia bindings, so every
//! language calls exactly the same engine:
//!
//! ```
//! let out = evalsuite::bridge::call("f1", r#"{"y_true": [0, 1, 1, 0], "y_pred": [0, 1, 0, 0]}"#).unwrap();
//! assert_eq!(out, r#"{"metric":"f1","value":0.6666666666666666}"#);
//! ```
//!
//! Arguments (by metric): `y_true`, `y_pred`, `y_prob` (binary: positive-class probabilities), `average`
//! (`binary`, `micro`, `macro`, `weighted`, `none`), `sample_weight`; `references` (one string or a list of
//! strings per prediction) and `predictions` for text metrics; `reference` and `current` for drift. The result is
//! `{"metric": name, "value": number}` (or `"values": [...]` for per-class / per-example results); errors are
//! returned as `Err` with EvalSuite's message.

use serde_json::{json, Map, Value};

use crate::classification::{self as cls, Average, Opts, Score};
use crate::error::{invalid, Result};
use crate::generation as gen;
use crate::labels::Label;
use crate::stats::{self, Interval};
use crate::{calibration as calib, clinical as clin, retrieval as ret, segmentation as seg};
use crate::{monitor, regression};

/// Metric names `call` understands.
pub const METRICS: &[&str] = &[
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "mcc",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "mae",
    "mse",
    "rmse",
    "r2",
    "mape",
    "median_absolute_error",
    "max_error",
    "bleu",
    "chrf",
    "ter",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "sensitivity",
    "ppv",
    "npv",
    "lr_positive",
    "lr_negative",
    "diagnostic_odds_ratio",
    "youden_j",
    "net_benefit",
    "calibration_slope",
    "calibration_intercept",
    "expected_calibration_error",
    "maximum_calibration_error",
    "hosmer_lemeshow",
    "ndcg",
    "mrr",
    "map",
    "precision_at_k",
    "recall_at_k",
    "hit_rate",
    "dice",
    "iou",
    "psi",
    "ks",
];

const CLASSIFICATION: &[&str] = &[
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "mcc",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
    "sensitivity",
    "ppv",
    "npv",
    "lr_positive",
    "lr_negative",
    "diagnostic_odds_ratio",
    "youden_j",
    "net_benefit",
    "calibration_slope",
    "calibration_intercept",
    "expected_calibration_error",
    "maximum_calibration_error",
    "hosmer_lemeshow",
];

/// Per-observation arguments: resampled together by the bootstrap.
const PER_OBS: &[&str] = &[
    "y_true",
    "y_pred",
    "y_prob",
    "sample_weight",
    "references",
    "predictions",
    "relevant",
    "retrieved",
];

fn field<'a>(a: &'a Map<String, Value>, k: &str) -> Result<&'a Value> {
    match a.get(k) {
        Some(v) if !v.is_null() => Ok(v),
        _ => invalid(format!("missing argument '{k}'.")),
    }
}

fn floats(a: &Map<String, Value>, k: &str) -> Result<Vec<f64>> {
    let Value::Array(v) = field(a, k)? else {
        return invalid(format!("'{k}' must be an array of numbers."));
    };
    v.iter()
        .map(|x| match x.as_f64() {
            Some(f) => Ok(f),
            None => invalid(format!("'{k}' must be an array of numbers.")),
        })
        .collect()
}

fn opt_floats(a: &Map<String, Value>, k: &str) -> Result<Option<Vec<f64>>> {
    if a.get(k).is_none_or(Value::is_null) {
        Ok(None)
    } else {
        floats(a, k).map(Some)
    }
}

fn strings(a: &Map<String, Value>, k: &str) -> Result<Vec<String>> {
    let Value::Array(v) = field(a, k)? else {
        return invalid(format!("'{k}' must be an array of strings."));
    };
    v.iter()
        .map(|x| match x {
            Value::String(s) => Ok(s.clone()),
            _ => invalid(format!("'{k}' must be an array of strings.")),
        })
        .collect()
}

fn references(a: &Map<String, Value>) -> Result<Vec<Vec<String>>> {
    let Value::Array(v) = field(a, "references")? else {
        return invalid("'references' must be an array.");
    };
    v.iter()
        .map(|x| match x {
            Value::String(s) => Ok(vec![s.clone()]),
            Value::Array(r) => r
                .iter()
                .map(|s| match s {
                    Value::String(s) => Ok(s.clone()),
                    _ => invalid("'references' must hold strings or lists of strings."),
                })
                .collect(),
            _ => invalid("'references' must hold strings or lists of strings."),
        })
        .collect()
}

fn average(a: &Map<String, Value>) -> Result<Average> {
    Ok(match a.get("average").and_then(Value::as_str) {
        None | Some("auto") => Average::Auto,
        Some("binary") => Average::Binary,
        Some("micro") => Average::Micro,
        Some("macro") => Average::Macro,
        Some("weighted") => Average::Weighted,
        Some("none") => Average::None,
        Some(o) => return invalid(format!("unknown average '{o}'.")),
    })
}

fn score(s: Score) -> Value {
    match s {
        Score::Scalar(v) => json!(v),
        Score::PerClass(v) => json!(v),
    }
}

fn labels_of(a: &Map<String, Value>, k: &str) -> Result<Option<Vec<Value>>> {
    match a.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(v)) => Ok(Some(v.clone())),
        _ => invalid(format!("'{k}' must be an array of labels.")),
    }
}

fn classify<L: Label>(
    name: &str,
    a: &Map<String, Value>,
    t: &[L],
    p: Option<&[L]>,
    pos: Option<L>,
    analytic: Option<f64>,
) -> Result<(Value, Option<Interval>)> {
    let w = opt_floats(a, "sample_weight")?;
    let w = w.as_deref();
    let need = || -> Result<&[L]> {
        match p {
            Some(p) => Ok(p),
            None => invalid(format!("{name} needs 'y_pred'.")),
        }
    };
    let pl = pos.as_ref();
    // Wilson interval for a proportion k / m of binary counts (unweighted only).
    let wilson = |k: f64, m: f64| -> Option<Interval> {
        let level = analytic?;
        if w.is_some() || m < 1.0 {
            return None;
        }
        stats::proportion_ci(k as u64, m as u64, level, stats::ProportionMethod::Wilson).ok()
    };
    let o = Opts {
        average: average(a)?,
        pos_label: pos.clone(),
        sample_weight: w.map(<[f64]>::to_vec),
        ..Opts::default()
    };
    let zd = cls::ZeroDivision::Warn;
    let counts = || clin::binary_counts(t, need()?, pl, None);
    Ok(match name {
        "accuracy" => {
            let pr = need()?;
            let k = t.iter().zip(pr).filter(|(a, b)| a == b).count();
            (
                json!(cls::accuracy(t, pr, w)?),
                wilson(k as f64, t.len() as f64),
            )
        }
        "balanced_accuracy" => (json!(cls::balanced_accuracy(t, need()?, false, w)?), None),
        "precision" => (score(cls::precision(t, need()?, &o)?), None),
        "recall" => (score(cls::recall(t, need()?, &o)?), None),
        "f1" => (score(cls::f1(t, need()?, &o)?), None),
        "specificity" | "sensitivity" | "ppv" | "npv" => {
            let (tp, fp, fn_, tn) = counts()?;
            let (k, m) = match name {
                "sensitivity" => (tp, tp + fn_),
                "specificity" => (tn, tn + fp),
                "ppv" => (tp, tp + fp),
                _ => (tn, tn + fn_),
            };
            let v = match name {
                "sensitivity" => clin::sensitivity(t, need()?, pl, w, zd)?,
                "specificity" => clin::specificity(t, need()?, pl, w, zd)?,
                "ppv" => clin::ppv(t, need()?, pl, w, zd)?,
                _ => clin::npv(t, need()?, pl, w, zd)?,
            };
            (json!(v), wilson(k, m))
        }
        "mcc" => (json!(cls::mcc(t, need()?, w, zd)?), None),
        "lr_positive" => (json!(clin::lr_positive(t, need()?, pl, w)?), None),
        "lr_negative" => (json!(clin::lr_negative(t, need()?, pl, w)?), None),
        "diagnostic_odds_ratio" => (
            json!(clin::diagnostic_odds_ratio(t, need()?, pl, w, 0.5)?),
            None,
        ),
        "youden_j" => (json!(clin::youden_j(t, need()?, pl, w)?), None),
        _ => {
            let q = floats(a, "y_prob")?;
            let bins = a.get("n_bins").and_then(Value::as_u64).unwrap_or(10) as usize;
            let strategy = match a.get("strategy").and_then(Value::as_str) {
                Some("quantile") => cls::binary::Strategy::Quantile,
                _ => cls::binary::Strategy::Uniform,
            };
            match name {
                "roc_auc" => {
                    let v = cls::binary::roc_auc(t, &q, pl, w)?;
                    let ci = match (analytic, w) {
                        (Some(level), None) => {
                            let pos = pos.clone().or_else(|| L::default_positive(t));
                            let y: Vec<u8> = t
                                .iter()
                                .map(|x| u8::from(Some(x) == pos.as_ref()))
                                .collect();
                            stats::roc_auc_ci(&y, &q, level).ok()
                        }
                        _ => None,
                    };
                    (json!(v), ci)
                }
                "average_precision" => (json!(cls::binary::average_precision(t, &q, pl, w)?), None),
                "log_loss" => (json!(cls::binary::log_loss(t, &q, pl, w)?), None),
                "brier_score" => (json!(cls::binary::brier_score(t, &q, pl, w)?), None),
                "net_benefit" => {
                    let thr = a.get("threshold").and_then(Value::as_f64).unwrap_or(0.5);
                    (json!(clin::net_benefit(t, &q, thr, pl, w)?), None)
                }
                "calibration_slope" => (json!(calib::calibration_slope(t, &q, pl, w)?.0), None),
                "calibration_intercept" => {
                    (json!(calib::calibration_intercept(t, &q, pl, w)?), None)
                }
                "expected_calibration_error" => (
                    json!(cls::binary::expected_calibration_error(
                        t, &q, bins, strategy, pl, w
                    )?),
                    None,
                ),
                "maximum_calibration_error" => (
                    json!(calib::maximum_calibration_error(
                        t, &q, bins, strategy, pl, w
                    )?),
                    None,
                ),
                _ => {
                    let (stat, p) = calib::hosmer_lemeshow(t, &q, bins, pl)?;
                    (json!({"statistic": stat, "p_value": p}), None)
                }
            }
        }
    })
}

fn classification(
    name: &str,
    a: &Map<String, Value>,
    analytic: Option<f64>,
) -> Result<(Value, Option<Interval>)> {
    let t = labels_of(a, "y_true")?.map_or_else(|| invalid("missing argument 'y_true'."), Ok)?;
    let p = labels_of(a, "y_pred")?;
    let pos = a.get("pos_label").filter(|v| !v.is_null()).cloned();
    let all = t.iter().chain(p.iter().flatten()).chain(pos.iter());
    if all.clone().all(|v| v.as_i64().is_some()) {
        let ints = |v: &[Value]| {
            v.iter()
                .map(|x| x.as_i64().unwrap_or(0))
                .collect::<Vec<i64>>()
        };
        let pv = p.as_deref().map(ints);
        let pl = pos.and_then(|v| v.as_i64());
        classify(name, a, &ints(&t), pv.as_deref(), pl, analytic)
    } else if all.clone().all(Value::is_string) {
        let strs = |v: &[Value]| {
            v.iter()
                .map(|x| x.as_str().unwrap_or("").to_owned())
                .collect::<Vec<String>>()
        };
        let pv = p.as_deref().map(strs);
        let pl = pos.and_then(|v| v.as_str().map(str::to_owned));
        classify(name, a, &strs(&t), pv.as_deref(), pl, analytic)
    } else {
        invalid("labels must be all integers or all strings.")
    }
}

fn id_of(v: &Value) -> Result<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        _ => invalid("document ids must be strings or numbers."),
    }
}

fn retrieval_args(
    a: &Map<String, Value>,
) -> Result<(Vec<ret::Judgements<String>>, Vec<Vec<String>>)> {
    let Value::Array(rel) = field(a, "relevant")? else {
        return invalid("'relevant' must be an array (one entry per query).");
    };
    let Value::Array(got) = field(a, "retrieved")? else {
        return invalid("'retrieved' must be an array (one ranked list per query).");
    };
    let relevant = rel
        .iter()
        .map(|q| match q {
            Value::Array(ids) => Ok(ret::judgements(
                ids.iter().map(id_of).collect::<Result<Vec<_>>>()?,
            )),
            Value::Object(g) => ret::graded(
                g.iter()
                    .map(|(k, v)| match v.as_f64() {
                        Some(x) => Ok((k.clone(), x)),
                        None => invalid("graded relevance must map ids to numbers."),
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            _ => invalid("each 'relevant' entry must be a list of ids or an {id: grade} object."),
        })
        .collect::<Result<Vec<_>>>()?;
    let retrieved = got
        .iter()
        .map(|q| match q {
            Value::Array(ids) => ids.iter().map(id_of).collect::<Result<Vec<_>>>(),
            _ => invalid("each 'retrieved' entry must be a ranked list of ids."),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((relevant, retrieved))
}

fn masks(a: &Map<String, Value>, k: &str, shape: &[usize]) -> Result<Vec<seg::Mask>> {
    let flat = floats(a, k)?;
    let size: usize = shape.iter().product();
    if size == 0 || flat.len() % size != 0 {
        return invalid(format!("'{k}' must hold whole masks of the given 'shape'."));
    }
    flat.chunks(size)
        .map(|c| seg::Mask::new(shape.to_vec(), c.iter().map(|&x| x as i64).collect()))
        .collect()
}

/// The metric itself (no interval).
fn compute(
    name: &str,
    a: &Map<String, Value>,
    analytic: Option<f64>,
) -> Result<(Value, Option<Interval>)> {
    if CLASSIFICATION.contains(&name) {
        return classification(name, a, analytic);
    }
    let value = match name {
        "mae" | "mse" | "rmse" | "r2" | "mape" | "median_absolute_error" | "max_error" => {
            let (t, p) = (floats(a, "y_true")?, floats(a, "y_pred")?);
            let w = opt_floats(a, "sample_weight")?;
            let w = w.as_deref();
            json!(match name {
                "mae" => regression::mae(&t, &p, w)?,
                "mse" => regression::mse(&t, &p, w)?,
                "rmse" => regression::rmse(&t, &p, w)?,
                "r2" => regression::r2(&t, &p, w)?,
                "mape" => regression::mape(&t, &p, w)?,
                "median_absolute_error" => regression::median_absolute_error(&t, &p, w)?,
                _ => regression::max_error(&t, &p)?,
            })
        }
        "bleu" | "chrf" | "ter" | "rouge_1" | "rouge_2" | "rouge_l" => {
            let (r, p) = (references(a)?, strings(a, "predictions")?);
            match name {
                "bleu" => json!(gen::bleu(&r, &p, &gen::BleuOpts::default())?),
                "chrf" => json!(gen::chrf(&r, &p, &gen::ChrfOpts::default())?),
                "ter" => json!(gen::ter(&r, &p, false)?),
                "rouge_1" => json!(gen::rouge_1(&r, &p)?),
                "rouge_2" => json!(gen::rouge_2(&r, &p)?),
                _ => json!(gen::rouge_l(&r, &p)?),
            }
        }
        "ndcg" | "mrr" | "map" | "precision_at_k" | "recall_at_k" | "hit_rate" => {
            let (rel, got) = retrieval_args(a)?;
            let k = a.get("k").and_then(Value::as_u64).map(|k| k as usize);
            let need_k = || match k {
                Some(k) => Ok(k),
                None => invalid(format!("{name} needs 'k'.")),
            };
            let q = match name {
                "ndcg" => {
                    let gains = match a.get("gains").and_then(Value::as_str) {
                        Some("exponential") => ret::Gains::Exponential,
                        _ => ret::Gains::Linear,
                    };
                    ret::ndcg_at_k(&rel, &got, k, gains)?
                }
                "mrr" => ret::mrr(&rel, &got, k)?,
                "map" => ret::mean_average_precision_at_k(&rel, &got, k)?,
                "precision_at_k" => ret::precision_at_k(&rel, &got, need_k()?)?,
                "recall_at_k" => ret::recall_at_k(&rel, &got, need_k()?)?,
                _ => ret::hit_rate_at_k(&rel, &got, need_k()?)?,
            };
            json!(q.mean)
        }
        "dice" | "iou" => {
            let shape: Vec<usize> = floats(a, "shape")?.iter().map(|&x| x as usize).collect();
            let (t, p) = (masks(a, "y_true", &shape)?, masks(a, "y_pred", &shape)?);
            let o = seg::Opts {
                num_classes: a
                    .get("num_classes")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize),
                average: match a.get("average").and_then(Value::as_str) {
                    Some("none") => seg::Average::None,
                    _ => seg::Average::Macro,
                },
                include_background: a
                    .get("include_background")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                ..seg::Opts::default()
            };
            score(if name == "dice" {
                seg::dice(&t, &p, &o)?
            } else {
                seg::iou(&t, &p, &o)?
            })
        }
        "psi" => json!(monitor::population_stability_index(
            &floats(a, "reference")?,
            &floats(a, "current")?,
            a.get("bins").and_then(Value::as_u64).unwrap_or(10) as usize,
            1e-4,
        )?),
        "ks" => {
            let (d, p) = monitor::ks_drift(&floats(a, "reference")?, &floats(a, "current")?)?;
            json!({"statistic": d, "p_value": p})
        }
        _ => {
            return invalid(format!(
                "unknown metric '{name}'. Available: {}.",
                METRICS.join(", ")
            ))
        }
    };
    Ok((value, None))
}

/// The arguments restricted to observations `idx` (every per-observation array resampled together).
fn subset(a: &Map<String, Value>, idx: &[usize]) -> Map<String, Value> {
    let mut out = a.clone();
    for k in PER_OBS {
        if let Some(Value::Array(v)) = a.get(*k) {
            out.insert(
                (*k).to_owned(),
                Value::Array(idx.iter().map(|&i| v[i].clone()).collect()),
            );
        }
    }
    out
}

fn n_obs(a: &Map<String, Value>) -> Result<usize> {
    let mut n = None;
    for k in PER_OBS {
        if let Some(Value::Array(v)) = a.get(*k) {
            match n {
                None => n = Some(v.len()),
                Some(m) if m != v.len() => {
                    return invalid(
                        "per-observation arguments must have the same length for an interval.",
                    )
                }
                _ => {}
            }
        }
    }
    match n {
        Some(n) if n >= 2 => Ok(n),
        _ => invalid("an interval needs at least two observations."),
    }
}

/// Computes metric `name` on the JSON-encoded arguments and returns the JSON-encoded result.
///
/// With `"ci": "analytic"` (Wilson for proportions, DeLong for ROC AUC; otherwise bootstrap) or
/// `"ci": "bootstrap"` (BCa, `n_resamples` default 1000, `seed` default 0), the result also has `ci_low`,
/// `ci_high`, `ci_level` (`level`, default 0.95) and `ci_method`.
pub fn call(name: &str, args: &str) -> Result<String> {
    let parsed: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(e) => return invalid(format!("arguments are not valid JSON: {e}")),
    };
    let Value::Object(a) = parsed else {
        return invalid("arguments must be a JSON object.");
    };
    let ci = match a.get("ci") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => None,
        Some(Value::String(s)) if s == "analytic" || s == "bootstrap" => Some(s.clone()),
        Some(Value::Bool(true)) => Some("analytic".to_owned()),
        _ => return invalid("'ci' must be \"analytic\", \"bootstrap\" or false."),
    };
    let level = a.get("level").and_then(Value::as_f64).unwrap_or(0.95);
    let analytic = (ci.as_deref() == Some("analytic")).then_some(level);
    let (value, mut interval) = compute(name, &a, analytic)?;
    let mut method = interval.as_ref().map(|_| {
        if name == "roc_auc" {
            "delong"
        } else {
            "wilson"
        }
    });
    if ci.is_some() && interval.is_none() {
        let Some(est) = value.as_f64() else {
            return invalid(format!(
                "an interval is available only for a single-number {name}."
            ));
        };
        let n = n_obs(&a)?;
        let opts = stats::Bootstrap {
            n_resamples: a.get("n_resamples").and_then(Value::as_u64).unwrap_or(1000) as usize,
            level,
            seed: a.get("seed").and_then(Value::as_u64).unwrap_or(0),
            ..stats::Bootstrap::default()
        };
        let b = stats::bootstrap_ci(
            n,
            |idx| {
                let (v, _) = compute(name, &subset(&a, idx), None)?;
                Ok(v.as_f64().unwrap_or(f64::NAN))
            },
            &opts,
        )?;
        interval = Some(Interval {
            estimate: est,
            low: b.low,
            high: b.high,
        });
        method = Some("bootstrap-bca");
    }
    let key = if value.is_array() { "values" } else { "value" };
    let mut out = Map::new();
    out.insert("metric".into(), json!(name));
    out.insert(key.into(), value);
    if let (Some(iv), Some(m)) = (interval, method) {
        out.insert("ci_low".into(), json!(iv.low));
        out.insert("ci_high".into(), json!(iv.high));
        out.insert("ci_level".into(), json!(level));
        out.insert("ci_method".into(), json!(m));
    }
    Ok(Value::Object(out).to_string())
}
