//! Throws random and malformed arguments at every bridge metric; reports panics (which would abort C, R and
//! Julia callers and kill a WebAssembly instance) and non-finite results.
use evalsuite::bridge::{call, METRICS};
use serde_json::{json, Value};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn weird_number(r: &mut Rng) -> Value {
    match r.below(12) {
        0 => json!(-1),
        1 => json!(2),
        2 => json!(1e308),
        3 => json!(-0.0),
        4 => json!("x"),
        5 => Value::Null,
        6 => json!(0.5),
        7 => json!(true),
        _ => json!(r.below(2)),
    }
}

fn arr(r: &mut Rng, n: usize, kind: u64) -> Value {
    Value::Array(
        (0..n)
            .map(|_| match kind {
                0 => json!(r.below(2)),
                1 => json!(r.below(4)),
                2 => json!(r.f()),
                3 => json!(r.f() * 10.0 - 5.0),
                4 => json!(["a", "b", "c"][r.below(3) as usize]),
                5 => json!(r.f() > 0.5),
                _ => weird_number(r),
            })
            .collect(),
    )
}

fn args(r: &mut Rng) -> Value {
    let n = match r.below(6) {
        0 => 0,
        1 => 1,
        2 => 2,
        _ => 3 + r.below(40) as usize,
    };
    let m = if r.below(8) == 0 { n + 1 } else { n };
    let mut a = serde_json::Map::new();
    let k = r.below(7);
    a.insert("y_true".into(), arr(r, n, k));
    a.insert("y_pred".into(), {
        let kk = if r.below(4) == 0 { r.below(7) } else { k };
        arr(r, m, kk)
    });
    a.insert("y_prob".into(), {
        let kk = if r.below(5) == 0 { 6 } else { 2 };
        arr(r, n, kk)
    });
    if r.below(4) == 0 {
        a.insert("sample_weight".into(), {
            let kk = if r.below(3) == 0 { 6 } else { 2 };
            arr(r, n, kk)
        });
    }
    let words = |r: &mut Rng| {
        (0..r.below(8))
            .map(|_| {
                [
                    "the",
                    "cat",
                    "",
                    "  ",
                    "é",
                    "3.5",
                    "-",
                    "<skipped>",
                    "&amp;",
                ][r.below(9) as usize]
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let preds: Vec<Value> = (0..n).map(|_| json!(words(r))).collect();
    let refs: Vec<Value> = (0..m)
        .map(|_| {
            if r.below(3) == 0 {
                json!([words(r), words(r)])
            } else if r.below(9) == 0 {
                json!([])
            } else {
                json!(words(r))
            }
        })
        .collect();
    a.insert("predictions".into(), Value::Array(preds));
    a.insert("references".into(), Value::Array(refs));
    let ids = |r: &mut Rng| -> Value {
        match r.below(5) {
            0 => json!([]),
            1 => json!({"a": r.f() * 3.0, "b": -1}),
            2 => json!("a"),
            _ => json!(["a", "b", 3]),
        }
    };
    a.insert(
        "relevant".into(),
        Value::Array((0..n).map(|_| ids(r)).collect()),
    );
    a.insert(
        "retrieved".into(),
        Value::Array((0..m).map(|_| ids(r)).collect()),
    );
    a.insert("reference".into(), {
        let kk = if r.below(5) == 0 { 6 } else { 3 };
        arr(r, n, kk)
    });
    a.insert("current".into(), arr(r, m, 3));
    for (key, vals) in [
        (
            "k",
            vec![json!(0), json!(1), json!(5), json!(-1), json!(1e20)],
        ),
        (
            "average",
            vec![
                json!("macro"),
                json!("none"),
                json!("binary"),
                json!("weird"),
                json!(3),
            ],
        ),
        (
            "threshold",
            vec![json!(0.0), json!(1.0), json!(0.3), json!(-2), json!("a")],
        ),
        (
            "n_bins",
            vec![json!(0), json!(1), json!(10), json!(100000000000u64)],
        ),
        (
            "bins",
            vec![json!(0), json!(1), json!(10), json!(100000000000u64)],
        ),
        (
            "ci",
            vec![
                json!("analytic"),
                json!("bootstrap"),
                json!(true),
                json!("x"),
            ],
        ),
        ("level", vec![json!(0.0), json!(1.0), json!(0.95), json!(2)]),
        ("n_resamples", vec![json!(0), json!(1), json!(20)]),
        (
            "pos_label",
            vec![json!(1), json!("a"), json!(7), Value::Null],
        ),
        (
            "shape",
            vec![
                json!([2, 2]),
                json!([0, 3]),
                json!([3]),
                json!([1e10, 1e10]),
                json!([-1, 2]),
            ],
        ),
        ("num_classes", vec![json!(0), json!(2), json!(1)]),
        ("strategy", vec![json!("quantile"), json!("uniform")]),
        ("gains", vec![json!("exponential")]),
    ] {
        if r.below(3) == 0 {
            a.insert(
                key.into(),
                vals[r.below(vals.len() as u64) as usize].clone(),
            );
        }
    }
    let models = (0..r.below(4))
        .map(|i| {
            (format!("m{i}"), {
                let kk = if r.below(5) == 0 { 6 } else { 0 };
                {
                    let nn = if r.below(6) == 0 { n + 1 } else { n };
                    arr(r, nn, kk)
                }
            })
        })
        .collect::<serde_json::Map<_, _>>();
    a.insert("models".into(), Value::Object(models));
    Value::Object(a)
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    std::panic::set_hook(Box::new(|_| {}));
    let mut r = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut panics, mut ok, mut err) = (0usize, 0usize, 0usize);
    let mut seen = std::collections::BTreeMap::<String, String>::new();
    let mut okm = std::collections::BTreeSet::<&str>::new();
    for _ in 0..iters {
        for &m in METRICS {
            let a = args(&mut r).to_string();
            match std::panic::catch_unwind(|| call(m, &a)) {
                Err(p) => {
                    panics += 1;
                    let msg = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_default();
                    seen.entry(format!("{m}: {msg}"))
                        .or_insert_with(|| a.chars().take(400).collect());
                }
                Ok(Ok(s)) => {
                    ok += 1;
                    okm.insert(m);
                    if s.contains("null") && !s.contains("\"values\"") {
                        seen.entry(format!("{m}: null in result")).or_insert(s);
                    }
                }
                Ok(Err(_)) => err += 1,
            }
        }
    }
    println!("ok={ok} err={err} panics={panics}");
    let never: Vec<&&str> = METRICS.iter().filter(|m| !okm.contains(**m)).collect();
    println!("never succeeded: {never:?}");
    for (k, v) in seen.iter().take(40) {
        println!("--- {k}\n    {v}");
    }
    if panics > 0 {
        std::process::exit(1);
    }
}
