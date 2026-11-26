use evalsuite::bridge::call;

#[test]
fn bridge_matches_direct_calls() {
    let f1 = call("f1", r#"{"y_true":[0,1,1,0],"y_pred":[0,1,0,0]}"#).unwrap();
    assert_eq!(f1, r#"{"metric":"f1","value":0.6666666666666666}"#);
    let s = call(
        "accuracy",
        r#"{"y_true":["a","b","c"],"y_pred":["a","b","b"]}"#,
    )
    .unwrap();
    assert!(s.contains("0.666"));
    let pc = call(
        "recall",
        r#"{"y_true":[0,1,2,2],"y_pred":[0,1,2,1],"average":"none"}"#,
    )
    .unwrap();
    assert!(pc.contains("\"values\":[1.0,1.0,0.5]"));
    let auc = call(
        "roc_auc",
        r#"{"y_true":[0,0,1,1],"y_prob":[0.1,0.4,0.35,0.8]}"#,
    )
    .unwrap();
    assert!(auc.contains("0.75"));
    let mae = call("mae", r#"{"y_true":[1,2,3],"y_pred":[1,2,4]}"#).unwrap();
    assert!(mae.contains("0.333"));
    let b = call(
        "bleu",
        r#"{"references":["the cat sat on the mat"],"predictions":["the cat sat on the mat"]}"#,
    )
    .unwrap();
    assert!(b.contains("100"));
    let ks = call("ks", r#"{"reference":[1,2,3,4],"current":[1,2,3,4]}"#).unwrap();
    assert!(ks.contains("\"statistic\":0.0"));
    assert!(call("nope", "{}")
        .unwrap_err()
        .to_string()
        .contains("unknown metric"));
    assert!(call("f1", r#"{"y_true":[0,1]}"#).is_err());
    assert!(call("f1", "not json").is_err());
}

fn val(s: &str, k: &str) -> f64 {
    let v: serde_json::Value = serde_json::from_str(s).unwrap();
    v[k].as_f64().unwrap()
}

#[test]
fn bridge_new_metrics_and_intervals() {
    let a = r#"{"y_true":[0,1,1,0,1,0,1,1,0,0],"y_pred":[0,1,0,0,1,1,1,1,0,0],"y_prob":[0.1,0.9,0.4,0.2,0.8,0.6,0.7,0.95,0.3,0.05]"#;
    for m in [
        "sensitivity",
        "ppv",
        "npv",
        "youden_j",
        "lr_positive",
        "net_benefit",
        "calibration_slope",
        "calibration_intercept",
        "expected_calibration_error",
        "maximum_calibration_error",
    ] {
        assert!(call(m, &format!("{a}}}")).is_ok(), "{m}");
    }
    assert!((val(&call("sensitivity", &format!("{a}}}")).unwrap(), "value") - 0.8).abs() < 1e-15);
    let hl = call("hosmer_lemeshow", &format!("{a}, \"n_bins\": 3}}")).unwrap();
    assert!(hl.contains("p_value"));
    // analytic: Wilson for accuracy, DeLong for ROC AUC
    let acc = call("accuracy", &format!("{a}, \"ci\": \"analytic\"}}")).unwrap();
    assert!(
        acc.contains("\"ci_method\":\"wilson\"")
            && val(&acc, "ci_low") < 0.8
            && val(&acc, "ci_high") > 0.8
    );
    let auc = call("roc_auc", &format!("{a}, \"ci\": true}}")).unwrap();
    assert!(auc.contains("delong"));
    // bootstrap for anything else; seeded and reproducible
    let b1 = call("mae", r#"{"y_true":[1,2,3,4,5,6],"y_pred":[1.5,2,2.5,4,6,5],"ci":"bootstrap","n_resamples":300,"seed":3}"#).unwrap();
    let b2 = call("mae", r#"{"y_true":[1,2,3,4,5,6],"y_pred":[1.5,2,2.5,4,6,5],"ci":"bootstrap","n_resamples":300,"seed":3}"#).unwrap();
    assert_eq!(b1, b2);
    assert!(val(&b1, "ci_low") <= val(&b1, "value") && val(&b1, "value") <= val(&b1, "ci_high"));
    // retrieval
    let r = call(
        "mrr",
        r#"{"relevant":[["a"],["x","y"]],"retrieved":[["b","a"],["y"]]}"#,
    )
    .unwrap();
    assert!((val(&r, "value") - 0.75).abs() < 1e-15);
    let n = call(
        "ndcg",
        r#"{"relevant":[{"a":3,"b":1}],"retrieved":[["a","b"]],"k":2}"#,
    )
    .unwrap();
    assert!((val(&n, "value") - 1.0).abs() < 1e-12);
    assert!(call(
        "precision_at_k",
        r#"{"relevant":[["a"]],"retrieved":[["a"]]}"#
    )
    .is_err());
    // segmentation
    let d = call("dice", r#"{"shape":[2,2],"y_true":[0,1,1,0],"y_pred":[0,1,0,0],"num_classes":2,"include_background":false}"#).unwrap();
    assert!((val(&d, "value") - 2.0 / 3.0).abs() < 1e-12);
    // errors
    assert!(call(
        "f1",
        r#"{"y_true":[0,1,2],"y_pred":[0,1,1],"average":"none","ci":"bootstrap"}"#
    )
    .is_err());
    assert!(call("f1", r#"{"y_true":[0,1],"y_pred":[0,1],"ci":"maybe"}"#).is_err());
}
