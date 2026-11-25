use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn run(args: &[&str], files: &[(&str, &str)]) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!(
        "evalsuite-cli-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, body) in files {
        std::fs::write(dir.join(name), body).unwrap();
    }
    let out = Command::new(env!("CARGO_BIN_EXE_evalsuite"))
        .current_dir(&dir)
        .args(args)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

const CLS: &str = "y,p,prob\n1,1,0.9\n0,0,0.2\n1,0,0.4\n1,1,0.7\n0,1,0.6\n0,0,0.1\n";

fn value(json: &str, key: &str) -> f64 {
    let v: serde_json::Value = serde_json::from_str(json).unwrap();
    v["metrics"][key]["value"].as_f64().unwrap()
}

// Expected values are those of the Python package's CLI on the same files.
#[test]
fn evaluate_binary() {
    let (ok, out, err) = run(
        &[
            "evaluate", "c.csv", "--y-true", "y", "--y-pred", "p", "--y-prob", "prob", "-f", "json",
        ],
        &[("c.csv", CLS)],
    );
    assert!(ok, "{err}");
    assert!((value(&out, "accuracy") - 2.0 / 3.0).abs() < 1e-15);
    assert!((value(&out, "roc_auc") - 0.8888888888888888).abs() < 1e-15);
    assert!((value(&out, "average_precision") - 0.9166666666666666).abs() < 1e-15);
    assert!((value(&out, "mcc") - 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn evaluate_regression_and_formats() {
    let reg = "t,p\n3,2.5\n-0.5,0\n2,2\n7,8\n";
    let (ok, out, _) = run(
        &[
            "evaluate",
            "r.csv",
            "--y-true",
            "t",
            "--y-pred",
            "p",
            "--task",
            "regression",
            "-f",
            "csv",
        ],
        &[("r.csv", reg)],
    );
    assert!(ok);
    assert!(out.starts_with("metric,name,value"));
    assert!(out.contains("rmse,\"RMSE\",0.6124"));
    let (ok, out, _) = run(
        &[
            "evaluate",
            "r.json",
            "--y-true",
            "t",
            "--y-pred",
            "p",
            "--task",
            "regression",
            "--metrics",
            "mae",
        ],
        &[(
            "r.json",
            r#"[{"t":3,"p":2.5},{"t":-0.5,"p":0},{"t":2,"p":2},{"t":7,"p":8}]"#,
        )],
    );
    assert!(ok);
    assert!(out.contains("MAE") && out.contains("0.5000"));
}

#[test]
fn text_and_diagnostic() {
    let txt = "ref,pred\nthe cat sat,the cat sat down\na dog,the dog\n";
    let (ok, out, _) = run(
        &[
            "text",
            "t.csv",
            "--prediction",
            "pred",
            "--reference",
            "ref",
            "-f",
            "json",
        ],
        &[("t.csv", txt)],
    );
    assert!(ok);
    assert!((value(&out, "bleu") - 53.7285).abs() < 1e-4);
    assert!((value(&out, "chrf_pp") - 78.7976).abs() < 1e-4);
    assert!((value(&out, "token_f1") - 0.9).abs() < 1e-12);
    let (ok, out, _) = run(
        &["diagnostic", "c.csv", "--y-true", "y", "--y-pred", "p"],
        &[("c.csv", CLS)],
    );
    assert!(ok);
    assert!(out.contains("TP=2  FP=1  FN=1  TN=2"));
    assert!(out.contains("(0.3342–11.9693)"));
}

#[test]
fn errors_are_reported() {
    let (ok, _, err) = run(
        &["evaluate", "c.csv", "--y-true", "nope", "--y-pred", "p"],
        &[("c.csv", CLS)],
    );
    assert!(!ok);
    assert!(err.contains("column \"nope\" not found"));
    let (ok, _, err) = run(
        &[
            "evaluate",
            "c.csv",
            "--y-true",
            "y",
            "--y-pred",
            "p",
            "--metrics",
            "accuracy,foo",
        ],
        &[("c.csv", CLS)],
    );
    assert!(!ok);
    assert!(err.contains("unknown metric \"foo\""));
}
