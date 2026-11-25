mod common;

use std::collections::HashSet;

use common::{check, nums, rows, strs, Got};
use evalsuite::{factuality::*, rag::*};
use serde_json::Value;

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> Got) -> Got {
    match r {
        Ok(x) => f(x),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn flags(v: &Value) -> Vec<Vec<bool>> {
    v.as_array().unwrap().iter().map(bools).collect()
}

fn claims(v: &Value) -> Vec<Vec<Claim>> {
    v.as_array().unwrap().iter().map(|a| strs(a).iter().map(|s| Claim::parse(s).unwrap()).collect()).collect()
}

fn avg(a: &Value) -> Average {
    if a["average"] == "micro" {
        Average::Micro
    } else {
        Average::Macro
    }
}

#[test]
fn grounding_parity() {
    let (n, _) = check("grounding", 1e-10, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        match name {
            "context_precision" => g(context_precision(&flags(&a["flags"])), |r| Got::List(r.per_query)),
            "context_recall" => g(context_recall(&flags(&a["flags"])), |r| if key == "mean" { Got::Num(r.mean) } else { Got::List(r.per_query) }),
            "context_relevance" => g(context_relevance(&flags(&a["flags"])), |r| Got::Num(r.mean)),
            "latency_summary" => g(latency_summary(&nums(&a["latencies"])), |s| Got::Num(match key {
                "mean" => s.mean,
                "p50" => s.p50,
                "p90" => s.p90,
                "p95" => s.p95,
                _ => s.p99,
            })),
            "task_success_rate" => g(task_success_rate(&bools(&a["success"])), |r| Got::Num(r.ci_low)),
            "failure_attribution" => {
                let s: Vec<Option<&str>> = a["stages"].as_array().unwrap().iter().map(Value::as_str).collect();
                g(failure_attribution(&s, &STAGES), Got::List)
            }
            "faithfulness" => g(faithfulness(&claims(&a["claims"]), avg(a)), |r| Got::Num(r.value)),
            "hallucination_rate" => {
                let k = if key == "contradicted" { Hallucination::Contradicted } else { Hallucination::Unsupported };
                g(hallucination_rate(&claims(&a["claims"]), k, avg(a)), |r| Got::Num(r.value))
            }
            "groundedness" => g(groundedness(&rows(&a["scores"]), avg(a)), |r| Got::Num(r.value)),
            "citation_recall" | "citation_precision" => {
                let c: Vec<Vec<Statement>> = a["citations"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|ans| ans.as_array().unwrap().iter().map(|s| Statement { supported: s["supported"].as_bool().unwrap(), citations: bools(&s["citations"]) }).collect())
                    .collect();
                let r = if name == "citation_recall" { citation_recall(&c, avg(a)) } else { citation_precision(&c, avg(a)) };
                g(r, |r| Got::Num(r.value))
            }
            "knowledge_consistency" => {
                let facts: Vec<Vec<String>> = a["facts"].as_array().unwrap().iter().map(strs).collect();
                let kb: HashSet<String> = ["f0", "f2", "f3"].iter().map(|s| s.to_string()).collect();
                g(knowledge_consistency(&facts, &kb, avg(a)), |r| Got::Num(r.value))
            }
            "claim_verification_accuracy" => {
                let p = |v: &Value| -> Vec<Claim> { strs(v).iter().map(|s| Claim::parse(s).unwrap()).collect() };
                g(claim_verification_accuracy(&p(&a["gold"]), &p(&a["predicted"])), |r| Got::Num(if key == "macro_f1" { r.macro_f1 } else { r.value }))
            }
            "answer_correctness" => {
                let sim = a.get("sim").map(nums);
                g(answer_correctness(&nums(&a["tp"]), &nums(&a["fp"]), &nums(&a["fneg"]), sim.as_deref(), (0.75, 0.25)), Got::List)
            }
            "answer_relevance" => {
                let gq: Vec<Vec<Vec<f64>>> = a["g"].as_array().unwrap().iter().map(rows).collect();
                g(answer_relevance(&rows(&a["q"]), &gq), Got::List)
            }
            "abstention_accuracy" => g(abstention_accuracy(&bools(&a["should"]), &bools(&a["abstained"]), Some(&bools(&a["correct"]))), |r| Got::Num(match key {
                "abstention_precision" => r.abstention_precision,
                "abstention_recall" => r.abstention_recall,
                "coverage" => r.coverage.unwrap(),
                "answered_accuracy" => r.answered_accuracy.unwrap(),
                _ => r.value,
            })),
            _ => Got::Skip,
        }
    });
    assert!(n > 1500);
}
