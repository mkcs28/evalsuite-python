use std::time::Instant;
fn main() {
    let n = 10_000;
    let t: Vec<f64> = (0..n).map(|i| (i % 97) as f64).collect();
    let p: Vec<f64> = (0..n)
        .map(|i| (i % 97) as f64 + ((i * 7) % 5) as f64 - 2.0)
        .collect();
    let y: Vec<i64> = (0..n).map(|i| i64::from(i % 3 == 0)).collect();
    let q: Vec<f64> = (0..n).map(|i| ((i * 37) % 100) as f64 / 100.0).collect();
    let reg = serde_json::json!({"y_true": t, "y_pred": p, "ci": "bootstrap", "n_resamples": 1000})
        .to_string();
    let cls = serde_json::json!({"y_true": y, "y_prob": q, "ci": "bootstrap", "n_resamples": 1000})
        .to_string();
    for (m, a) in [("mae", &reg), ("roc_auc", &cls), ("brier_score", &cls)] {
        let s = Instant::now();
        let out = evalsuite::bridge::call(m, a).unwrap();
        println!(
            "{m}: {:.0} ms  {}",
            s.elapsed().as_secs_f64() * 1e3,
            &out[..out.len().min(90)]
        );
    }
}
