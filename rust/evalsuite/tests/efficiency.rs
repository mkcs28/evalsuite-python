mod common;

use common::{check, nums, Got};
use evalsuite::efficiency::*;
use evalsuite::summary::Summary;

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> f64) -> Got {
    match r {
        Ok(x) => Got::Num(f(x)),
        Err(e) => Got::Err(e.to_string()),
    }
}

fn s(x: Summary, key: &str) -> f64 {
    match key {
        "p50" => x.p50,
        "p90" => x.p90,
        "p99" => x.p99,
        "std" => x.std,
        _ => x.mean,
    }
}

#[test]
fn efficiency_parity() {
    let (n, _) = check("efficiency", 1e-10, |f, a| {
        let key = f.rsplit('.').next().unwrap();
        let v = |k: &str| nums(&a[k]);
        match f.split('.').next().unwrap() {
            "time_to_first_token" => g(time_to_first_token(&v("st"), &v("ft")), |x| s(x, key)),
            "time_per_output_token" => {
                g(time_per_output_token(&v("ft"), &v("et"), &v("toks")), |x| {
                    s(x.0, key)
                })
            }
            "latency_percentiles" => {
                let lat: Vec<f64> = v("et").iter().zip(v("st")).map(|(e, s)| e - s).collect();
                g(latency_percentiles(&lat, a["q"].as_f64().unwrap()), |x| x.0)
            }
            "throughput" => g(throughput(&v("toks"), &v("st"), &v("et")), |x| match key {
                "per_request_tokens_per_second" => x.per_request_tokens_per_second,
                "window_s" => x.window_s,
                _ => x.value,
            }),
            "token_usage" => g(token_usage(&v("inp"), &v("toks")), |x| match key {
                "input_p90" => x.input.p90,
                "total_output" => x.total_output,
                _ => x.value,
            }),
            "inference_cost" => {
                let c = v("cached");
                let cached = if f.contains("nocache") {
                    None
                } else {
                    Some((c.as_slice(), 0.3))
                };
                g(
                    inference_cost(&v("inp"), &v("toks"), 3.0, 15.0, cached),
                    |x| match key {
                        "total_cost" => x.total_cost,
                        "cost_per_1k_tokens" => x.cost_per_1k_tokens,
                        _ => x.value,
                    },
                )
            }
            "resource_utilization" => {
                let (gpu, mem) = (v("gpu"), v("mem"));
                g(
                    resource_utilization(&[("gpu_util", &gpu), ("GPU_Memory", &mem)]),
                    |x| if key == "p95" { x.2[0].p95 } else { x.0 },
                )
            }
            "energy_per_request" => g(
                energy_per_request(
                    Some(&v("power")),
                    Some(0.5),
                    a["n"].as_u64().unwrap() as usize,
                    None,
                ),
                |x| x.0,
            ),
            "requests_per_second" => g(
                requests_per_second(&v("et"), if key == "start" { Some(0.0) } else { None }),
                |x| x,
            ),
            "availability" => {
                let ok: Vec<bool> = a["ok"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_bool().unwrap())
                    .collect();
                g(availability(&ok, 0.99), |x| match key {
                    "error_budget_remaining" => x.error_budget_remaining,
                    "ci_low" => x.rate.ci_low,
                    _ => x.rate.value,
                })
            }
            _ => Got::Skip,
        }
    });
    assert!(n > 1500);
}
