mod common;

use common::*;
use evalsuite::generation::{self as g, BleuOpts, ChrfOpts, Measure, MeteorOpts, Rouge, Smooth, Tokenize};
use serde_json::Value;

fn refs(a: &Value) -> Vec<Vec<String>> {
    a["references"].as_array().unwrap().iter().map(strs).collect()
}

fn v(r: evalsuite::Result<f64>) -> Got {
    r.map_or_else(|e| Got::Err(e.to_string()), Got::Num)
}

fn l(r: evalsuite::Result<Vec<f64>>) -> Got {
    r.map_or_else(|e| Got::Err(e.to_string()), Got::List)
}

#[test]
fn generation_matches_python() {
    let (n, sk) = check("generation", 1e-9, |f, a| {
        let tok = |x: &Value| if x.as_str() == Some("none") { Tokenize::None } else if x.as_str() == Some("13a") { Tokenize::Thirteen } else { Tokenize::None };
        if f == "tokenize_13a" {
            let t = g::tokenize_13a(a["line"].as_str().unwrap());
            let mut out = vec![t.len() as f64];
            out.extend(t.join("\u{0}").chars().map(|c| u32::from(c) as f64));
            return Got::List(out);
        }
        if f == "cross_entropy" || f == "perplexity" {
            let lp = rows(&a["token_logprobs"]);
            return if f == "cross_entropy" { v(g::cross_entropy(&lp, num(&a["base"]))) } else { v(g::perplexity(&lp, a["average"] == "sequences")) };
        }
        let p = strs(&a["predictions"]);
        match f {
            "self_bleu" => v(g::self_bleu(&p, 4, Tokenize::Thirteen)),
            "distinct_n" => v(g::distinct_n(&p, a["n"].as_u64().unwrap() as usize, a["lowercase"].as_bool().unwrap())),
            _ => {
                let r = refs(a);
                match f {
                    "bleu" => {
                        let smooth = match a["smooth"].as_str().unwrap() {
                            "floor" => Smooth::Floor(0.1),
                            "add-k" => Smooth::AddK(1.0),
                            "none" => Smooth::None,
                            _ => Smooth::Exp,
                        };
                        let o = BleuOpts { smooth, lowercase: a["lowercase"].as_bool().unwrap(), tokenize: tok(&a["tokenize"]), ..Default::default() };
                        v(g::bleu(&r, &p, &o))
                    }
                    "sentence_bleu" => l(g::sentence_bleu(&r, &p, &BleuOpts::default())),
                    "chrf" => {
                        let o = ChrfOpts {
                            word_order: a["word_order"].as_u64().unwrap() as usize,
                            lowercase: a["lowercase"].as_bool().unwrap(),
                            whitespace: a["whitespace"].as_bool().unwrap(),
                            eps_smoothing: a["eps_smoothing"].as_bool().unwrap_or(false),
                            ..Default::default()
                        };
                        v(g::chrf(&r, &p, &o))
                    }
                    "ter" => v(g::ter(&r, &p, a["case_sensitive"].as_bool().unwrap())),
                    "rouge_1" | "rouge_2" | "rouge_l" | "rouge_lsum" => {
                        let variant = match f {
                            "rouge_1" => Rouge::N(1),
                            "rouge_2" => Rouge::N(2),
                            "rouge_l" => Rouge::L,
                            _ => Rouge::Lsum,
                        };
                        let m = match a["measure"].as_str().unwrap() {
                            "precision" => Measure::Precision,
                            "recall" => Measure::Recall,
                            _ => Measure::Fmeasure,
                        };
                        l(g::rouge(&r, &p, variant, m, None))
                    }
                    "meteor" => {
                        let o = MeteorOpts {
                            alpha: a["alpha"].as_f64().unwrap_or(0.9),
                            beta: a["beta"].as_f64().unwrap_or(3.0),
                            gamma: a["gamma"].as_f64().unwrap_or(0.5),
                            tokenize: tok(&a["tokenize"]),
                            ..Default::default()
                        };
                        l(g::meteor(&r, &p, &o))
                    }
                    "cider" => l(g::cider(&r, &p, 4, 6.0, a["lowercase"].as_bool().unwrap(), Tokenize::None)),
                    _ => Got::Skip,
                }
            }
        }
    });
    assert!(n > 1100 && sk == 0);
}
