mod common;

use common::{check, nums, rows, Got};
use evalsuite::semantic::*;
use serde_json::Value;

fn mats(v: &Value) -> Vec<Vec<Vec<f64>>> {
    v.as_array().unwrap().iter().map(rows).collect()
}

fn list(r: evalsuite::Result<Vec<f64>>) -> Got {
    match r {
        Ok(v) => Got::List(v),
        Err(e) => Got::Err(e.to_string()),
    }
}

#[test]
fn semantic_parity() {
    let (n, _) = check("semantic", 1e-9, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, ""));
        match name {
            "bertscore" => {
                let m = match key {
                    "precision" => BertMeasure::Precision,
                    "recall" => BertMeasure::Recall,
                    _ => BertMeasure::F1,
                };
                if key == "weighted" {
                    list(bertscore(
                        &mats(&a["refs"]),
                        &mats(&a["preds"]),
                        Some(&rows(&a["rw"])),
                        Some(&rows(&a["pw"])),
                        m,
                        Some(0.3),
                    ))
                } else {
                    list(bertscore(
                        &mats(&a["refs"]),
                        &mats(&a["preds"]),
                        None,
                        None,
                        m,
                        None,
                    ))
                }
            }
            "moverscore" => {
                if key == "weighted" {
                    list(moverscore(
                        &mats(&a["refs"]),
                        &mats(&a["preds"]),
                        Some(&rows(&a["rw"])),
                        Some(&rows(&a["pw"])),
                    ))
                } else {
                    list(moverscore(
                        &mats(&a["refs"]),
                        &mats(&a["preds"]),
                        None,
                        None,
                    ))
                }
            }
            "embedding_similarity" => {
                let m = match a["metric"].as_str().unwrap() {
                    "euclidean" => Similarity::Euclidean,
                    "manhattan" => Similarity::Manhattan,
                    _ => Similarity::Cosine,
                };
                list(embedding_similarity(
                    &rows(&a["a"]),
                    &rows(&a["b"]),
                    m,
                    a["normalize"].as_bool().unwrap(),
                ))
            }
            "mauve_from_histograms" => Got::Num(mauve_from_histograms(
                &nums(&a["p"]),
                &nums(&a["q"]),
                a["disc"].as_u64().unwrap() as usize,
                a["sf"].as_f64().unwrap(),
            )),
            _ => Got::Skip,
        }
    });
    assert!(n > 500);
}

#[test]
fn mauve_behaves() {
    // Same distribution -> close to 1; disjoint clusters -> close to 0.
    let mut s = 1u64;
    let mut r = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let a: Vec<Vec<f64>> = (0..200)
        .map(|i| vec![(i % 4) as f64 + 0.1 * r(), 0.1 * r(), 1.0 + 0.1 * r()])
        .collect();
    let b: Vec<Vec<f64>> = (0..200)
        .map(|i| vec![(i % 4) as f64 + 0.1 * r(), 0.1 * r(), 1.0 + 0.1 * r()])
        .collect();
    let c: Vec<Vec<f64>> = (0..200)
        .map(|_| vec![0.1 * r(), 5.0 + 0.1 * r(), -1.0 + 0.1 * r()])
        .collect();
    let o = MauveOpts::default();
    let same = mauve(&a, &b, &o).unwrap().value;
    let far = mauve(&a, &c, &o).unwrap().value;
    assert!(same > 0.9, "{same}");
    assert!(far < 0.05, "{far}");
}
