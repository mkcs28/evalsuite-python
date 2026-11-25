mod common;

use common::*;
use evalsuite::retrieval::{self as r, Gains, Judgements, QueryScores};

#[test]
fn retrieval_matches_python() {
    let (n, s) = check("retrieval", 1e-12, |f, a| {
        let rel: Vec<Judgements<i64>> = a["relevant"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), num(v))).collect())
            .collect();
        let ret: Vec<Vec<i64>> = a["retrieved"].as_array().unwrap().iter().map(ints).collect();
        let k = a["k"].as_u64().map(|k| k as usize);
        let out: evalsuite::Result<QueryScores> = match f {
            "precision_at_k" => r::precision_at_k(&rel, &ret, k.unwrap()),
            "recall_at_k" => r::recall_at_k(&rel, &ret, k.unwrap()),
            "hit_rate_at_k" => r::hit_rate_at_k(&rel, &ret, k.unwrap()),
            "mrr" => r::mrr(&rel, &ret, k),
            "mean_average_precision_at_k" => r::mean_average_precision_at_k(&rel, &ret, k),
            "ndcg_at_k" => r::ndcg_at_k(&rel, &ret, k, if a["gains"] == "linear" { Gains::Linear } else { Gains::Exponential }),
            _ => return Got::Skip,
        };
        out.map_or_else(|e| Got::Err(e.to_string()), |q| Got::List(q.per_query))
    });
    assert!(n > 500 && s == 0);
}
