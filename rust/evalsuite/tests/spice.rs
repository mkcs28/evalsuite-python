mod common;

use common::{check, Got};
use evalsuite::spice::spice;
use serde_json::Value;

fn tuples(v: &Value) -> Vec<Vec<Vec<String>>> {
    v.as_array().unwrap().iter().map(|img| img.as_array().unwrap().iter().map(common::strs).collect()).collect()
}

fn syn(w: &str) -> Vec<String> {
    match w {
        "dog" | "puppy" => vec!["canine".into()],
        "grass" | "lawn" => vec!["lawn".into()],
        "red" | "crimson" => vec!["crimson".into()],
        _ => vec![],
    }
}

#[test]
fn spice_parity() {
    let (n, _) = check("spice", 1e-12, |f, a| {
        let s: Option<&dyn Fn(&str) -> Vec<String>> = if a["syn"] == true { Some(&syn) } else { None };
        match spice(&tuples(&a["refs"]), &tuples(&a["cands"]), s) {
            Err(e) => Got::Err(e.to_string()),
            Ok(r) => match f {
                "spice.object" => Got::Num(r.object_f),
                "spice.attribute" => Got::Num(r.attribute_f),
                "spice.relation" => Got::Num(r.relation_f),
                _ => Got::List(r.per_image),
            },
        }
    });
    assert!(n > 1000);
}
