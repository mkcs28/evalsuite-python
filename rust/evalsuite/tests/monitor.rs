mod common;

use common::{check, nums, Got};
use evalsuite::monitor::*;

#[test]
fn drift_parity() {
    let (n, _) = check("monitor", 1e-12, |f, a| match f {
        "psi" => match population_stability_index(
            &nums(&a["r"]),
            &nums(&a["c"]),
            a["bins"].as_u64().unwrap() as usize,
            1e-4,
        ) {
            Ok(v) => Got::Num(v),
            Err(e) => Got::Err(e.to_string()),
        },
        "ks" => Got::Num(ks_drift(&nums(&a["r"]), &nums(&a["c"])).unwrap().0),
        _ => Got::Skip,
    });
    assert!(n > 200);
}

#[test]
fn monitor_alerts() {
    let mut m = Monitor::new(100).alert("accuracy", Op::Lt, 0.8).min_n(20);
    let good: Vec<i64> = (0..50).map(|i| i % 2).collect();
    assert!(m.update(&good, Some(&good), None).unwrap().is_empty());
    let bad: Vec<i64> = good.iter().map(|x| 1 - x).collect();
    let a = m.update(&good, Some(&bad), None).unwrap();
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].n, 100);
    assert!((a[0].value - 0.5).abs() < 1e-12);
    assert!(m.update(&good, None, Some(&[0.5; 50])).is_err());
    let r: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0).powi(2)).collect();
    let mut d = Monitor::new(500).reference(&r).alert("psi", Op::Gt, 0.25);
    let same: Vec<f64> = r.iter().step_by(2).copied().collect();
    assert!(d.observe(&same).unwrap().is_empty());
    assert!(d.drift().unwrap()[0].1 < 0.01);
    let shifted: Vec<f64> = (0..500).map(|i| 0.9 + i as f64 / 5000.0).collect();
    let mut d2 = Monitor::new(500).reference(&r).alert("psi", Op::Gt, 0.25);
    assert_eq!(d2.observe(&shifted).unwrap()[0].name, "psi");
}
