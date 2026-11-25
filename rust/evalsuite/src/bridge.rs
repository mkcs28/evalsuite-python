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
    "psi",
    "ks",
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
) -> Result<Value> {
    let w = opt_floats(a, "sample_weight")?;
    let w = w.as_deref();
    let need = || -> Result<&[L]> {
        match p {
            Some(p) => Ok(p),
            None => invalid(format!("{name} needs 'y_pred'.")),
        }
    };
    let o = Opts {
        average: average(a)?,
        pos_label: pos.clone(),
        sample_weight: w.map(<[f64]>::to_vec),
        ..Opts::default()
    };
    Ok(match name {
        "accuracy" => json!(cls::accuracy(t, need()?, w)?),
        "balanced_accuracy" => json!(cls::balanced_accuracy(t, need()?, false, w)?),
        "precision" => score(cls::precision(t, need()?, &o)?),
        "recall" => score(cls::recall(t, need()?, &o)?),
        "f1" => score(cls::f1(t, need()?, &o)?),
        "specificity" => score(cls::specificity(t, need()?, &o)?),
        "mcc" => json!(cls::mcc(t, need()?, w, cls::ZeroDivision::Warn)?),
        _ => {
            let q = floats(a, "y_prob")?;
            let pl = pos.as_ref();
            json!(match name {
                "roc_auc" => cls::binary::roc_auc(t, &q, pl, w)?,
                "average_precision" => cls::binary::average_precision(t, &q, pl, w)?,
                "log_loss" => cls::binary::log_loss(t, &q, pl, w)?,
                _ => cls::binary::brier_score(t, &q, pl, w)?,
            })
        }
    })
}

fn classification(name: &str, a: &Map<String, Value>) -> Result<Value> {
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
        classify(
            name,
            a,
            &ints(&t),
            pv.as_deref(),
            pos.and_then(|v| v.as_i64()),
        )
    } else if all.clone().all(Value::is_string) {
        let strs = |v: &[Value]| {
            v.iter()
                .map(|x| x.as_str().unwrap_or("").to_owned())
                .collect::<Vec<String>>()
        };
        let pv = p.as_deref().map(strs);
        let pl = pos.and_then(|v| v.as_str().map(str::to_owned));
        classify(name, a, &strs(&t), pv.as_deref(), pl)
    } else {
        invalid("labels must be all integers or all strings.")
    }
}

/// Computes metric `name` on the JSON-encoded arguments and returns the JSON-encoded result.
pub fn call(name: &str, args: &str) -> Result<String> {
    let parsed: Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(e) => return invalid(format!("arguments are not valid JSON: {e}")),
    };
    let Value::Object(a) = parsed else {
        return invalid("arguments must be a JSON object.");
    };
    let value = match name {
        "accuracy" | "balanced_accuracy" | "precision" | "recall" | "f1" | "specificity"
        | "mcc" | "roc_auc" | "average_precision" | "log_loss" | "brier_score" => {
            classification(name, &a)?
        }
        "mae" | "mse" | "rmse" | "r2" | "mape" | "median_absolute_error" | "max_error" => {
            let (t, p) = (floats(&a, "y_true")?, floats(&a, "y_pred")?);
            let w = opt_floats(&a, "sample_weight")?;
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
            let (r, p) = (references(&a)?, strings(&a, "predictions")?);
            match name {
                "bleu" => json!(gen::bleu(&r, &p, &gen::BleuOpts::default())?),
                "chrf" => json!(gen::chrf(&r, &p, &gen::ChrfOpts::default())?),
                "ter" => json!(gen::ter(&r, &p, false)?),
                "rouge_1" => json!(gen::rouge_1(&r, &p)?),
                "rouge_2" => json!(gen::rouge_2(&r, &p)?),
                _ => json!(gen::rouge_l(&r, &p)?),
            }
        }
        "psi" => json!(monitor::population_stability_index(
            &floats(&a, "reference")?,
            &floats(&a, "current")?,
            a.get("bins").and_then(Value::as_u64).unwrap_or(10) as usize,
            1e-4,
        )?),
        "ks" => {
            let (d, p) = monitor::ks_drift(&floats(&a, "reference")?, &floats(&a, "current")?)?;
            json!({"statistic": d, "p_value": p})
        }
        _ => {
            return invalid(format!(
                "unknown metric '{name}'. Available: {}.",
                METRICS.join(", ")
            ))
        }
    };
    let key = if value.is_array() { "values" } else { "value" };
    let mut out = Map::new();
    out.insert("metric".into(), json!(name));
    out.insert(key.into(), value);
    Ok(Value::Object(out).to_string())
}
