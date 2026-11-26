//! `evalsuite`: evaluate predictions stored in CSV, TSV or JSON files from the command line.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use evalsuite::classification::{self as cls, Average, MultiClass, Opts, ZeroDivision};
use evalsuite::{calibration, clinical, generation, qa, regression, stats, Label};

#[derive(Parser)]
#[command(
    name = "evalsuite",
    version,
    about = "EvalSuite: unified, reproducible evaluation for machine learning and research."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Format {
    Text,
    Json,
    Csv,
    Markdown,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum Task {
    Classification,
    Regression,
}

#[derive(clap::Args)]
struct Out {
    /// Output format (default: from --output extension, else text).
    #[arg(long, short = 'f', value_enum)]
    format: Option<Format>,
    /// Write to this file instead of standard output.
    #[arg(long, short = 'o')]
    output: Option<String>,
    /// Decimal places.
    #[arg(long, default_value_t = 4)]
    digits: usize,
}

#[derive(Subcommand)]
enum Command {
    /// Evaluate predictions in a file with the standard (or chosen) metrics.
    Evaluate {
        /// CSV / TSV / JSON / JSON-lines file with one row per observation ('-' reads CSV from stdin).
        file: String,
        /// Column with the true labels or values.
        #[arg(long)]
        y_true: String,
        /// Column with predicted labels or values.
        #[arg(long)]
        y_pred: Option<String>,
        /// Probability column(s): one for P(positive), or one per class in label order.
        #[arg(long, num_args = 1..)]
        y_prob: Vec<String>,
        /// Column with sample weights.
        #[arg(long)]
        weight: Option<String>,
        /// Default: inferred from the data.
        #[arg(long, value_enum)]
        task: Option<Task>,
        /// Comma-separated metric names, e.g. accuracy,f1,mcc.
        #[arg(long)]
        metrics: Option<String>,
        #[command(flatten)]
        out: Out,
    },
    /// Diagnostic accuracy of a binary test: sensitivity, specificity, PPV, NPV, LR+, LR−, DOR with CIs.
    Diagnostic {
        file: String,
        #[arg(long)]
        y_true: String,
        #[arg(long)]
        y_pred: String,
        /// Confidence level.
        #[arg(long, default_value_t = 0.95)]
        level: f64,
        #[command(flatten)]
        out: Out,
    },
    /// Calibration of predicted risks: Brier, ECE, MCE, intercept, slope, Hosmer–Lemeshow.
    Calibration {
        file: String,
        #[arg(long)]
        y_true: String,
        #[arg(long)]
        y_prob: String,
        /// Bins for ECE / MCE.
        #[arg(long, default_value_t = 10)]
        n_bins: usize,
        #[command(flatten)]
        out: Out,
    },
    /// Text generation: BLEU, chrF, chrF++, TER, ROUGE, exact match, token F1.
    Text {
        file: String,
        /// Column with the model output.
        #[arg(long)]
        prediction: String,
        /// Column with a reference (repeat for several).
        #[arg(long, required = true)]
        reference: Vec<String>,
        /// Comma-separated subset, e.g. bleu,chrf,rouge_l.
        #[arg(long)]
        metrics: Option<String>,
        #[command(flatten)]
        out: Out,
    },
    /// List the metrics each command can compute.
    Metrics,
    /// Install the latest evalsuite-cli from crates.io (runs `cargo install evalsuite-cli`, which replaces an
    /// older version and leaves the current one in place).
    Upgrade {
        /// Show the command instead of running it.
        #[arg(long)]
        dry_run: bool,
    },
}

/// A table read from a file: column name → cells.
struct Table {
    columns: HashMap<String, Vec<String>>,
    n: usize,
}

impl Table {
    fn col(&self, name: &str) -> Result<&[String], String> {
        self.columns.get(name).map(Vec::as_slice).ok_or_else(|| {
            let mut have: Vec<&String> = self.columns.keys().collect();
            have.sort();
            format!("column {name:?} not found; the file has {have:?}")
        })
    }

    fn nums(&self, name: &str) -> Result<Vec<f64>, String> {
        self.col(name)?
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.trim()
                    .parse::<f64>()
                    .map_err(|_| format!("column {name:?}, row {}: {v:?} is not a number", i + 1))
            })
            .collect()
    }
}

fn read_table(path: &str) -> Result<Table, String> {
    let mut text = String::new();
    if path == "-" {
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(|e| e.to_string())?;
    } else {
        text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    }
    let lower = path.to_lowercase();
    let trimmed = text.trim_start();
    if lower.ends_with(".json")
        || lower.ends_with(".jsonl")
        || lower.ends_with(".ndjson")
        || trimmed.starts_with('{')
        || trimmed.starts_with('[')
    {
        return read_json(&text);
    }
    let delim = if lower.ends_with(".tsv") || lower.ends_with(".tab") {
        b'\t'
    } else {
        b','
    };
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delim)
        .from_reader(text.as_bytes());
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .map(str::to_owned)
        .collect();
    let mut columns: HashMap<String, Vec<String>> =
        headers.iter().map(|h| (h.clone(), Vec::new())).collect();
    let mut n = 0;
    for rec in rdr.records() {
        let rec = rec.map_err(|e| e.to_string())?;
        for (h, v) in headers.iter().zip(rec.iter()) {
            columns.get_mut(h).unwrap().push(v.to_owned());
        }
        n += 1;
    }
    Ok(Table { columns, n })
}

fn read_json(text: &str) -> Result<Table, String> {
    let rows: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(text) {
        Ok(serde_json::Value::Array(a)) => a,
        Ok(serde_json::Value::Object(o)) => {
            // Column-oriented: {"col": [..], ...}
            let mut columns = HashMap::new();
            let mut n = 0;
            for (k, v) in o {
                let cells: Vec<String> = v
                    .as_array()
                    .ok_or("a JSON object must map columns to arrays")?
                    .iter()
                    .map(cell)
                    .collect();
                n = cells.len();
                columns.insert(k, cells);
            }
            return Ok(Table { columns, n });
        }
        _ => text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?,
    };
    let mut columns: HashMap<String, Vec<String>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        let o = r
            .as_object()
            .ok_or(format!("row {} is not a JSON object", i + 1))?;
        for (k, v) in o {
            let col = columns
                .entry(k.clone())
                .or_insert_with(|| vec![String::new(); i]);
            col.push(cell(v));
        }
        for c in columns.values_mut() {
            c.resize(i + 1, String::new());
        }
    }
    Ok(Table {
        n: rows.len(),
        columns,
    })
}

fn cell(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => i32::from(*b).to_string(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// One output row: key, display name, value, optional interval and note.
struct Row {
    key: String,
    name: String,
    value: f64,
    ci: Option<(f64, f64)>,
    note: String,
}

fn row(key: &str, name: &str, value: f64) -> Row {
    Row {
        key: key.into(),
        name: name.into(),
        value,
        ci: None,
        note: String::new(),
    }
}

fn render(
    title: &str,
    rows: &[Row],
    meta: &[(&str, serde_json::Value)],
    out: &Out,
) -> Result<(), String> {
    let fmt = out.format.unwrap_or_else(|| {
        match out
            .output
            .as_deref()
            .map(|o| o.rsplit('.').next().unwrap_or("").to_lowercase())
        {
            Some(e) if e == "json" => Format::Json,
            Some(e) if e == "csv" => Format::Csv,
            Some(e) if e == "md" || e == "markdown" => Format::Markdown,
            _ => Format::Text,
        }
    });
    let d = out.digits;
    let num = |v: f64| {
        if v.is_nan() {
            "nan".to_owned()
        } else if v.is_infinite() {
            if v > 0.0 {
                "inf".into()
            } else {
                "-inf".into()
            }
        } else {
            format!("{v:.d$}")
        }
    };
    let jnum = |v: f64| {
        if v.is_finite() {
            serde_json::json!(v)
        } else {
            serde_json::json!(num(v))
        }
    };
    let mut s = String::new();
    match fmt {
        Format::Text => {
            s.push_str(title);
            s.push('\n');
            let w = rows
                .iter()
                .map(|r| r.name.chars().count())
                .max()
                .unwrap_or(0);
            for r in rows {
                let pad = " ".repeat(w - r.name.chars().count());
                s.push_str(&format!("  {}{pad}  {:>10}", r.name, num(r.value)));
                if let Some((lo, hi)) = r.ci {
                    s.push_str(&format!("  ({}–{})", num(lo), num(hi)));
                }
                if !r.note.is_empty() {
                    s.push_str(&format!("  {}", r.note));
                }
                s.push('\n');
            }
        }
        Format::Json => {
            let mut m = serde_json::Map::new();
            for (k, v) in meta {
                m.insert((*k).into(), v.clone());
            }
            let mut metrics = serde_json::Map::new();
            for r in rows {
                let mut o = serde_json::Map::new();
                o.insert("name".into(), serde_json::json!(r.name));
                o.insert("value".into(), jnum(r.value));
                if let Some((lo, hi)) = r.ci {
                    o.insert("ci_low".into(), jnum(lo));
                    o.insert("ci_high".into(), jnum(hi));
                }
                metrics.insert(r.key.clone(), serde_json::Value::Object(o));
            }
            m.insert("metrics".into(), serde_json::Value::Object(metrics));
            s = serde_json::to_string_pretty(&serde_json::Value::Object(m)).unwrap();
            s.push('\n');
        }
        Format::Csv => {
            s.push_str("metric,name,value,ci_low,ci_high\n");
            for r in rows {
                let (lo, hi) =
                    r.ci.map_or((String::new(), String::new()), |(a, b)| (num(a), num(b)));
                s.push_str(&format!(
                    "{},\"{}\",{},{lo},{hi}\n",
                    r.key,
                    r.name.replace('"', "\"\""),
                    num(r.value)
                ));
            }
        }
        Format::Markdown => {
            s.push_str(&format!(
                "**{title}**\n\n| Metric | Value | 95% CI |\n| --- | ---: | --- |\n"
            ));
            for r in rows {
                let ci =
                    r.ci.map_or(String::new(), |(a, b)| format!("{}–{}", num(a), num(b)));
                s.push_str(&format!("| {} | {} | {ci} |\n", r.name, num(r.value)));
            }
        }
    }
    match &out.output {
        Some(p) => std::fs::write(p, s).map_err(|e| format!("{p}: {e}")),
        None => std::io::stdout()
            .write_all(s.as_bytes())
            .map_err(|e| e.to_string()),
    }
}

const CLASSIFICATION: [(&str, &str); 12] = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("specificity", "Specificity"),
    ("mcc", "MCC"),
    ("cohen_kappa", "Cohen's kappa"),
    ("roc_auc", "ROC AUC"),
    ("average_precision", "Average precision"),
    ("log_loss", "Log loss"),
    ("brier_score", "Brier score"),
];

const REGRESSION: [(&str, &str); 16] = [
    ("mae", "MAE"),
    ("mse", "MSE"),
    ("rmse", "RMSE"),
    ("r2", "R²"),
    ("explained_variance", "Explained variance"),
    ("median_absolute_error", "Median absolute error"),
    ("max_error", "Max error"),
    ("mean_bias_error", "Mean bias error"),
    ("mape", "MAPE"),
    ("smape", "sMAPE"),
    ("msle", "MSLE"),
    ("rmsle", "RMSLE"),
    ("rae", "RAE"),
    ("rse", "RSE"),
    ("huber_loss", "Huber loss"),
    ("quantile_loss", "Quantile loss"),
];

const TEXT: [(&str, &str, &str); 9] = [
    ("bleu", "BLEU", "0–100, sacreBLEU"),
    ("chrf", "chrF", "0–100"),
    ("chrf_pp", "chrF++", "0–100"),
    ("ter", "TER", "0–∞, lower is better"),
    ("rouge_1", "ROUGE-1", "F1, 0–1"),
    ("rouge_2", "ROUGE-2", "F1, 0–1"),
    ("rouge_l", "ROUGE-L", "F1, 0–1"),
    ("exact_match", "Exact match", "0–1, SQuAD normalisation"),
    ("token_f1", "Token F1", "0–1, SQuAD normalisation"),
];

fn chosen<'a>(
    all: &'a [(&'a str, &'a str)],
    pick: &Option<String>,
    default: &[&str],
) -> Result<Vec<(&'a str, &'a str)>, String> {
    let names: Vec<String> = match pick {
        Some(p) => p
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect(),
        None => default.iter().map(|s| (*s).to_owned()).collect(),
    };
    names
        .iter()
        .map(|n| {
            all.iter().find(|(k, _)| k == n).copied().ok_or_else(|| {
                format!(
                    "unknown metric {n:?}; available: {}",
                    all.iter().map(|x| x.0).collect::<Vec<_>>().join(", ")
                )
            })
        })
        .collect()
}

fn err(e: evalsuite::Error) -> String {
    e.to_string()
}

fn classification<L: Label>(
    y: &[L],
    p: Option<&[L]>,
    prob: &[Vec<f64>],
    w: Option<&[f64]>,
    pick: &Option<String>,
) -> Result<(Vec<Row>, String), String> {
    let mut labels: Vec<&L> = y.iter().chain(p.unwrap_or(&[])).collect();
    labels.sort();
    labels.dedup();
    let binary = labels.len() <= 2 && (prob.is_empty() || prob.len() == 1);
    let mut default: Vec<&str> = vec![];
    if p.is_some() {
        default.extend([
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "mcc",
            "cohen_kappa",
        ]);
    }
    if !prob.is_empty() {
        default.extend(["roc_auc", "average_precision", "log_loss", "brier_score"]);
    }
    let picked = chosen(&CLASSIFICATION, pick, &default)?;
    let opts = Opts {
        average: if binary {
            Average::Binary
        } else {
            Average::Macro
        },
        sample_weight: w.map(<[f64]>::to_vec),
        ..Opts::default()
    };
    let rows_prob: Vec<Vec<f64>> = if prob.len() > 1 {
        (0..y.len())
            .map(|i| prob.iter().map(|c| c[i]).collect())
            .collect()
    } else {
        vec![]
    };
    let mut rows = Vec::new();
    for (k, name) in picked {
        let need_pred = || p.ok_or_else(|| format!("{k} needs --y-pred"));
        let need_prob = || {
            if prob.is_empty() {
                Err(format!("{k} needs --y-prob"))
            } else {
                Ok(())
            }
        };
        let scalar = |s: cls::Score| s.value().unwrap_or(f64::NAN);
        let v = match k {
            "accuracy" => cls::accuracy(y, need_pred()?, w).map_err(err)?,
            "balanced_accuracy" => {
                cls::balanced_accuracy(y, need_pred()?, false, w).map_err(err)?
            }
            "precision" => scalar(cls::precision(y, need_pred()?, &opts).map_err(err)?),
            "recall" => scalar(cls::recall(y, need_pred()?, &opts).map_err(err)?),
            "f1" => scalar(cls::f1(y, need_pred()?, &opts).map_err(err)?),
            "specificity" => scalar(cls::specificity(y, need_pred()?, &opts).map_err(err)?),
            "mcc" => cls::mcc(y, need_pred()?, w, ZeroDivision::Warn).map_err(err)?,
            "cohen_kappa" => cls::cohen_kappa(
                y,
                need_pred()?,
                cls::KappaWeights::Unweighted,
                None,
                w,
                ZeroDivision::Warn,
            )
            .map_err(err)?,
            _ => {
                need_prob()?;
                if prob.len() == 1 {
                    let q = &prob[0];
                    match k {
                        "roc_auc" => cls::binary::roc_auc(y, q, None, w),
                        "average_precision" => cls::binary::average_precision(y, q, None, w),
                        "log_loss" => cls::binary::log_loss(y, q, None, w),
                        _ => cls::binary::brier_score(y, q, None, w),
                    }
                    .map_err(err)?
                } else {
                    match k {
                        "roc_auc" => scalar(
                            cls::roc_auc(y, &rows_prob, Average::Macro, MultiClass::Ovr, None, w)
                                .map_err(err)?,
                        ),
                        "average_precision" => scalar(
                            cls::average_precision(y, &rows_prob, Average::Macro, None, w)
                                .map_err(err)?,
                        ),
                        "log_loss" => cls::log_loss(y, &rows_prob, None, w).map_err(err)?,
                        _ => cls::brier_score(y, &rows_prob, None, w).map_err(err)?,
                    }
                }
            }
        };
        rows.push(row(k, name, v));
    }
    Ok((
        rows,
        if binary {
            "binary".into()
        } else {
            "multiclass".into()
        },
    ))
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Upgrade { dry_run } => {
            let args = ["install", "--locked", "evalsuite-cli"];
            println!(
                "evalsuite {}: cargo {}",
                env!("CARGO_PKG_VERSION"),
                args.join(" ")
            );
            if dry_run {
                return Ok(());
            }
            let status = std::process::Command::new(
                std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()),
            )
            .args(args)
            .status()
            .map_err(|e| {
                format!("could not run cargo ({e}); install Rust from https://rustup.rs")
            })?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("cargo install failed ({status})"))
            }
        }
        Command::Metrics => {
            println!(
                "evaluate (classification): {}",
                CLASSIFICATION
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!(
                "evaluate (regression):     {}",
                REGRESSION
                    .iter()
                    .map(|x| x.0)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            println!("diagnostic:  sensitivity, specificity, ppv, npv, lr_positive, lr_negative, diagnostic_odds_ratio, youden_j, accuracy, prevalence");
            println!("calibration: brier_score, ece, mce, calibration_intercept, calibration_slope, hosmer_lemeshow");
            println!(
                "text:        {}",
                TEXT.iter().map(|x| x.0).collect::<Vec<_>>().join(", ")
            );
            println!("\nThe Rust library has every EvalSuite metric: https://docs.rs/evalsuite");
            Ok(())
        }
        Command::Evaluate {
            file,
            y_true,
            y_pred,
            y_prob,
            weight,
            task,
            metrics,
            out,
        } => {
            let t = read_table(&file)?;
            let w = weight.as_deref().map(|c| t.nums(c)).transpose()?;
            let yt = t.col(&y_true)?;
            let numeric = yt.iter().all(|v| v.trim().parse::<f64>().is_ok());
            let integral = yt.iter().all(|v| v.trim().parse::<i64>().is_ok());
            let task = task.unwrap_or(
                if !y_prob.is_empty()
                    || !numeric
                    || (integral && {
                        let mut u: Vec<&String> = yt.iter().collect();
                        u.sort();
                        u.dedup();
                        u.len() <= 20
                    })
                {
                    Task::Classification
                } else {
                    Task::Regression
                },
            );
            if task == Task::Regression {
                let (a, b) = (
                    t.nums(&y_true)?,
                    t.nums(y_pred.as_deref().ok_or("regression needs --y-pred")?)?,
                );
                let picked = chosen(
                    &REGRESSION,
                    &metrics,
                    &REGRESSION.iter().take(8).map(|x| x.0).collect::<Vec<_>>(),
                )?;
                let mut rows = Vec::new();
                for (k, name) in picked {
                    let wk = if k == "max_error" { None } else { w.as_deref() };
                    let v = match k {
                        "mae" => regression::mae(&a, &b, wk),
                        "mse" => regression::mse(&a, &b, wk),
                        "rmse" => regression::rmse(&a, &b, wk),
                        "r2" => regression::r2(&a, &b, wk),
                        "explained_variance" => regression::explained_variance(&a, &b, wk),
                        "median_absolute_error" => regression::median_absolute_error(&a, &b, wk),
                        "max_error" => regression::max_error(&a, &b),
                        "mean_bias_error" => regression::mean_bias_error(&a, &b, wk),
                        "mape" => regression::mape(&a, &b, wk),
                        "smape" => regression::smape(&a, &b, wk),
                        "msle" => regression::msle(&a, &b, wk),
                        "rmsle" => regression::rmsle(&a, &b, wk),
                        "rae" => regression::rae(&a, &b, wk),
                        "rse" => regression::rse(&a, &b, wk),
                        "huber_loss" => regression::huber_loss(&a, &b, 1.0, wk),
                        _ => regression::quantile_loss(&a, &b, 0.5, wk),
                    }
                    .map_err(err)?;
                    rows.push(row(k, name, v));
                }
                return render(
                    &format!("EvalSuite regression evaluation (n={})", t.n),
                    &rows,
                    &[("task", "regression".into()), ("n_samples", t.n.into())],
                    &out,
                );
            }
            let probs: Vec<Vec<f64>> =
                y_prob.iter().map(|c| t.nums(c)).collect::<Result<_, _>>()?;
            let (rows, kind) = if integral {
                let y: Vec<i64> = yt.iter().map(|v| v.trim().parse().unwrap()).collect();
                let p: Option<Vec<i64>> = y_pred
                    .as_deref()
                    .map(|c| {
                        t.col(c)?
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                v.trim().parse().map_err(|_| {
                                    format!("row {}: {v:?} is not an integer label", i + 1)
                                })
                            })
                            .collect()
                    })
                    .transpose()?;
                classification(&y, p.as_deref(), &probs, w.as_deref(), &metrics)?
            } else {
                let y: Vec<String> = yt.to_vec();
                let p: Option<Vec<String>> = y_pred
                    .as_deref()
                    .map(|c| t.col(c).map(<[String]>::to_vec))
                    .transpose()?;
                classification(&y, p.as_deref(), &probs, w.as_deref(), &metrics)?
            };
            render(
                &format!("EvalSuite classification evaluation ({kind}, n={})", t.n),
                &rows,
                &[
                    ("task", "classification".into()),
                    ("target_type", kind.into()),
                    ("n_samples", t.n.into()),
                ],
                &out,
            )
        }
        Command::Diagnostic {
            file,
            y_true,
            y_pred,
            level,
            out,
        } => {
            let t = read_table(&file)?;
            let (yt, yp) = (t.col(&y_true)?, t.col(&y_pred)?);
            let parse = |v: &[String]| {
                v.iter()
                    .map(|x| x.trim().parse::<i64>())
                    .collect::<Result<Vec<i64>, _>>()
            };
            let r = match (parse(yt), parse(yp)) {
                (Ok(a), Ok(b)) => clinical::diagnostic_report(
                    &a,
                    &b,
                    None,
                    level,
                    stats::ProportionMethod::Wilson,
                ),
                _ => clinical::diagnostic_report(
                    yt,
                    yp,
                    None,
                    level,
                    stats::ProportionMethod::Wilson,
                ),
            }
            .map_err(err)?;
            let iv = |k: &str, name: &str, i: stats::Interval, note: &str| Row {
                key: k.into(),
                name: name.into(),
                value: i.estimate,
                ci: Some((i.low, i.high)),
                note: note.into(),
            };
            let rows = vec![
                iv("sensitivity", "Sensitivity", r.sensitivity, "wilson"),
                iv("specificity", "Specificity", r.specificity, "wilson"),
                iv("ppv", "PPV", r.ppv, "wilson"),
                iv("npv", "NPV", r.npv, "wilson"),
                iv("lr_positive", "LR+", r.lr_positive, "log (Simel 1991)"),
                iv("lr_negative", "LR−", r.lr_negative, "log (Simel 1991)"),
                iv(
                    "diagnostic_odds_ratio",
                    "Diagnostic odds ratio",
                    r.diagnostic_odds_ratio,
                    "log (Woolf)",
                ),
                iv("youden_j", "Youden's J", r.youden_j, "wald"),
                iv("accuracy", "Accuracy", r.accuracy, "wilson"),
                iv("prevalence", "Prevalence", r.prevalence, "wilson"),
            ];
            let (a, b, c, d) = r.table;
            let pct = (level * 100.0).round();
            render(
                &format!("EvalSuite diagnostic accuracy (n={}, {pct}% CIs)\n  2×2 table: TP={a}  FP={b}  FN={c}  TN={d}", t.n),
                &rows,
                &[("n_samples", t.n.into()), ("table", serde_json::json!({"tp": a, "fp": b, "fn": c, "tn": d}))],
                &out,
            )
        }
        Command::Calibration {
            file,
            y_true,
            y_prob,
            n_bins,
            out,
        } => {
            let t = read_table(&file)?;
            let q = t.nums(&y_prob)?;
            let y: Vec<i64> = t
                .col(&y_true)?
                .iter()
                .map(|v| {
                    v.trim()
                        .parse::<i64>()
                        .map_err(|_| format!("{v:?} is not a 0/1 label"))
                })
                .collect::<Result<_, _>>()?;
            let strategy = cls::binary::Strategy::Uniform;
            let (stat, p) = calibration::hosmer_lemeshow(&y, &q, 10, None).map_err(err)?;
            let (slope, _) = calibration::calibration_slope(&y, &q, None, None).map_err(err)?;
            let mut hl = row("hosmer_lemeshow", "Hosmer–Lemeshow χ²", stat);
            hl.note = format!("p = {p:.4}");
            let rows = vec![
                row(
                    "brier_score",
                    "Brier score",
                    cls::binary::brier_score(&y, &q, None, None).map_err(err)?,
                ),
                row(
                    "ece",
                    "ECE",
                    cls::binary::expected_calibration_error(&y, &q, n_bins, strategy, None, None)
                        .map_err(err)?,
                ),
                row(
                    "mce",
                    "MCE",
                    calibration::maximum_calibration_error(&y, &q, n_bins, strategy, None, None)
                        .map_err(err)?,
                ),
                row(
                    "calibration_intercept",
                    "Calibration intercept",
                    calibration::calibration_intercept(&y, &q, None, None).map_err(err)?,
                ),
                row("calibration_slope", "Calibration slope", slope),
                hl,
            ];
            render(
                &format!("EvalSuite calibration (n={})", t.n),
                &rows,
                &[("n_samples", t.n.into())],
                &out,
            )
        }
        Command::Text {
            file,
            prediction,
            reference,
            metrics,
            out,
        } => {
            let t = read_table(&file)?;
            let preds = t.col(&prediction)?.to_vec();
            let cols: Vec<&[String]> = reference
                .iter()
                .map(|c| t.col(c))
                .collect::<Result<_, _>>()?;
            let refs: Vec<Vec<String>> = (0..t.n)
                .map(|i| {
                    cols.iter()
                        .map(|c| c[i].clone())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
                .collect();
            let all: Vec<(&str, &str)> = TEXT.iter().map(|x| (x.0, x.1)).collect();
            let picked = chosen(
                &all,
                &metrics,
                &TEXT.iter().map(|x| x.0).collect::<Vec<_>>(),
            )?;
            let m = |v: Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
            let mut rows = Vec::new();
            for (k, name) in picked {
                let v = match k {
                    "bleu" => generation::bleu(&refs, &preds, &generation::BleuOpts::default()),
                    "chrf" => generation::chrf(&refs, &preds, &generation::ChrfOpts::default()),
                    "chrf_pp" => generation::chrf(
                        &refs,
                        &preds,
                        &generation::ChrfOpts {
                            word_order: 2,
                            ..Default::default()
                        },
                    ),
                    "ter" => generation::ter(&refs, &preds, false),
                    "rouge_1" => generation::rouge_1(&refs, &preds).map(m),
                    "rouge_2" => generation::rouge_2(&refs, &preds).map(m),
                    "rouge_l" => generation::rouge_l(&refs, &preds).map(m),
                    "exact_match" => qa::exact_match(&refs, &preds, qa::Norm::Squad).map(m),
                    _ => qa::token_f1(&refs, &preds, qa::Norm::Squad).map(m),
                }
                .map_err(err)?;
                let mut r = row(k, name, v);
                r.note = TEXT.iter().find(|x| x.0 == k).unwrap().2.into();
                rows.push(r);
            }
            let max_refs = refs.iter().map(Vec::len).max().unwrap_or(0);
            render(
                &format!(
                    "EvalSuite text evaluation ({} examples, {max_refs} reference(s) max)",
                    t.n
                ),
                &rows,
                &[("n_examples", t.n.into())],
                &out,
            )
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("evalsuite: error: {e}");
            ExitCode::from(2)
        }
    }
}
