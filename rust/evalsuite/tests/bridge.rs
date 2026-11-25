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
