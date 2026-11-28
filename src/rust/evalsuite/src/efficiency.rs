//! Serving-efficiency metrics: time to first token, time per output token, latency percentiles, throughput,
//! token usage, cost, resource utilisation, energy, request rate and availability, as EvalSuite's Python
//! `evalsuite.llmsys.efficiency`. Times are in seconds unless stated.
//!
//! ```
//! use evalsuite::efficiency::time_to_first_token;
//! let s = time_to_first_token(&[0.0, 1.0], &[0.25, 1.75]).unwrap();
//! assert_eq!(s.mean, 0.5);
//! ```

use crate::error::{invalid, Result};
use crate::summary::{floats, rate, same, Rate, Summary};
use crate::util::{mean, np_sum, percentile, sorted};

fn durations(start: &[f64], end: &[f64], a: &str, b: &str) -> Result<Vec<f64>> {
    floats(start, a, None, None)?;
    floats(end, b, None, None)?;
    same(start.len(), end.len(), a, b)?;
    let d: Vec<f64> = end.iter().zip(start).map(|(e, s)| e - s).collect();
    if d.iter().any(|&x| x < 0.0) {
        return invalid(format!("{b} must not be earlier than {a}."));
    }
    Ok(d)
}

/// Time to first token per request, summarised (the value is the mean).
pub fn time_to_first_token(request_times: &[f64], first_token_times: &[f64]) -> Result<Summary> {
    Ok(Summary::of(&durations(
        request_times,
        first_token_times,
        "request_times",
        "first_token_times",
    )?))
}

/// Time per output token after the first, over requests with at least two output tokens; returns the
/// summary and the number of requests used.
pub fn time_per_output_token(
    first_token_times: &[f64],
    end_times: &[f64],
    output_tokens: &[f64],
) -> Result<(Summary, usize)> {
    let d = durations(
        first_token_times,
        end_times,
        "first_token_times",
        "end_times",
    )?;
    floats(output_tokens, "output_tokens", Some(0.0), None)?;
    same(d.len(), output_tokens.len(), "end_times", "output_tokens")?;
    let per: Vec<f64> = d
        .iter()
        .zip(output_tokens)
        .filter(|(_, &n)| n > 1.0)
        .map(|(d, n)| d / (n - 1.0))
        .collect();
    if per.is_empty() {
        return invalid("TPOT needs requests with at least two output tokens.");
    }
    Ok((Summary::of(&per), per.len()))
}

/// The `percentile`-th latency (linear interpolation), with the full summary.
pub fn latency_percentiles(latencies: &[f64], percentile_: f64) -> Result<(f64, Summary)> {
    floats(latencies, "latencies", Some(0.0), None)?;
    if !(0.0..=100.0).contains(&percentile_) {
        return invalid("percentile must be in [0, 100].");
    }
    Ok((
        percentile(&sorted(latencies), percentile_),
        Summary::of(latencies),
    ))
}

/// Throughput: output tokens per second over the whole window, and the mean per-request rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Throughput {
    pub value: f64,
    pub per_request_tokens_per_second: f64,
    pub window_s: f64,
}

/// Aggregate output-token throughput over the measurement window (first start to last end).
pub fn throughput(
    output_tokens: &[f64],
    start_times: &[f64],
    end_times: &[f64],
) -> Result<Throughput> {
    floats(output_tokens, "output_tokens", Some(0.0), None)?;
    let d = durations(start_times, end_times, "start_times", "end_times")?;
    same(output_tokens.len(), d.len(), "output_tokens", "end_times")?;
    let span = end_times.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - start_times.iter().copied().fold(f64::INFINITY, f64::min);
    if span <= 0.0 {
        return invalid("The measurement window has zero length.");
    }
    let per: Vec<f64> = output_tokens
        .iter()
        .zip(&d)
        .filter(|(_, &d)| d > 0.0)
        .map(|(n, d)| n / d)
        .collect();
    Ok(Throughput {
        value: np_sum(output_tokens) / span,
        per_request_tokens_per_second: if per.is_empty() { f64::NAN } else { mean(&per) },
        window_s: span,
    })
}

/// Token usage: mean total tokens per request, with input / output summaries and totals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenUsage {
    pub value: f64,
    pub input: Summary,
    pub output: Summary,
    pub total_input: f64,
    pub total_output: f64,
}

/// Tokens per request.
pub fn token_usage(input_tokens: &[f64], output_tokens: &[f64]) -> Result<TokenUsage> {
    floats(input_tokens, "input_tokens", Some(0.0), None)?;
    floats(output_tokens, "output_tokens", Some(0.0), None)?;
    same(
        input_tokens.len(),
        output_tokens.len(),
        "input_tokens",
        "output_tokens",
    )?;
    let tot: Vec<f64> = input_tokens
        .iter()
        .zip(output_tokens)
        .map(|(a, b)| a + b)
        .collect();
    Ok(TokenUsage {
        value: mean(&tot),
        input: Summary::of(input_tokens),
        output: Summary::of(output_tokens),
        total_input: np_sum(input_tokens),
        total_output: np_sum(output_tokens),
    })
}

/// Cost per request with the total and the cost per 1,000 tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    pub value: f64,
    pub total_cost: f64,
    pub cost_per_1k_tokens: f64,
}

/// Cost from token counts and prices per million tokens; cached input tokens (a subset of the input) are
/// billed at `cached = Some((tokens, price))`.
pub fn inference_cost(
    input_tokens: &[f64],
    output_tokens: &[f64],
    input_price: f64,
    output_price: f64,
    cached: Option<(&[f64], f64)>,
) -> Result<Cost> {
    floats(input_tokens, "input_tokens", Some(0.0), None)?;
    floats(output_tokens, "output_tokens", Some(0.0), None)?;
    same(
        input_tokens.len(),
        output_tokens.len(),
        "input_tokens",
        "output_tokens",
    )?;
    if input_price < 0.0 || output_price < 0.0 {
        return invalid("Prices must be non-negative.");
    }
    let mut cost: Vec<f64> = input_tokens
        .iter()
        .zip(output_tokens)
        .map(|(i, o)| i * input_price / 1e6 + o * output_price / 1e6)
        .collect();
    if let Some((c, price)) = cached {
        if price < 0.0 {
            return invalid("cached_tokens needs a non-negative cached_price.");
        }
        floats(c, "cached_tokens", Some(0.0), None)?;
        same(input_tokens.len(), c.len(), "input_tokens", "cached_tokens")?;
        if c.iter().zip(input_tokens).any(|(c, i)| c > i) {
            return invalid("cached_tokens cannot exceed input_tokens.");
        }
        for (k, c) in cost.iter_mut().zip(c) {
            *k -= c * (input_price - price) / 1e6;
        }
    }
    let tokens: Vec<f64> = input_tokens
        .iter()
        .zip(output_tokens)
        .map(|(a, b)| a + b)
        .collect();
    let tokens = np_sum(&tokens);
    let total = np_sum(&cost);
    Ok(Cost {
        value: mean(&cost),
        total_cost: total,
        cost_per_1k_tokens: if tokens > 0.0 {
            total / tokens * 1000.0
        } else {
            f64::NAN
        },
    })
}

/// Peak, mean and p95 of one resource series.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: String,
    pub peak: f64,
    pub mean: f64,
    pub p95: f64,
}

/// Resource utilisation per series; the value is the peak of the first series whose name contains "mem"
/// (case-insensitive), else of the first series. Returns (value, primary series name, all series).
pub fn resource_utilization(readings: &[(&str, &[f64])]) -> Result<(f64, String, Vec<Series>)> {
    if readings.is_empty() {
        return invalid("readings must map a series name to its samples.");
    }
    let mut out = Vec::new();
    for (k, v) in readings {
        floats(v, &format!("readings[{k:?}]"), Some(0.0), None)?;
        let s = sorted(v);
        out.push(Series {
            name: (*k).to_owned(),
            peak: s[s.len() - 1],
            mean: mean(v),
            p95: percentile(&s, 95.0),
        });
    }
    let primary = out
        .iter()
        .find(|s| s.name.to_lowercase().contains("mem"))
        .unwrap_or(&out[0]);
    Ok((primary.peak, primary.name.clone(), out.clone()))
}

/// Energy per request in watt-hours, from power samples taken every `interval_s` seconds (trapezoidal
/// integration) or from a measured total in joules. Returns (Wh per request, total Wh, joules per request).
pub fn energy_per_request(
    power_watts: Option<&[f64]>,
    interval_s: Option<f64>,
    n_requests: usize,
    total_joules: Option<f64>,
) -> Result<(f64, f64, f64)> {
    if n_requests < 1 {
        return invalid("n_requests must be a positive integer.");
    }
    let joules = match total_joules {
        Some(j) => {
            if j < 0.0 {
                return invalid("total_joules must be non-negative.");
            }
            j
        }
        None => {
            let (Some(p), Some(dt)) = (power_watts, interval_s) else {
                return invalid("Pass power_watts with a positive interval_s, or total_joules.");
            };
            if dt <= 0.0 {
                return invalid("Pass power_watts with a positive interval_s, or total_joules.");
            }
            floats(p, "power_watts", Some(0.0), None)?;
            if p.len() > 1 {
                let t: Vec<f64> = (1..p.len()).map(|i| (p[i] + p[i - 1]) / 2.0).collect();
                np_sum(&t) * dt
            } else {
                p[0] * dt
            }
        }
    };
    let wh = joules / 3600.0;
    Ok((wh / n_requests as f64, wh, joules / n_requests as f64))
}

/// Completed requests per second: n / (t_last − start_time), or (n − 1) / (t_last − t_first) without a start.
pub fn requests_per_second(completion_times: &[f64], start_time: Option<f64>) -> Result<f64> {
    floats(completion_times, "completion_times", None, None)?;
    let t = sorted(completion_times);
    let (span, n) = match start_time {
        Some(s) => (t[t.len() - 1] - s, t.len()),
        None => {
            if t.len() < 2 {
                return invalid("Need at least two completions (or pass start_time).");
            }
            (t[t.len() - 1] - t[0], t.len() - 1)
        }
    };
    if span <= 0.0 {
        return invalid("The measurement window has zero length.");
    }
    Ok(n as f64 / span)
}

/// Availability with its error rate and the remaining error budget against an SLO.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Availability {
    pub rate: Rate,
    pub error_rate: f64,
    pub error_budget_remaining: f64,
}

/// Share of successful requests (`success[i]`), against a target `slo` in (0, 1).
pub fn availability(success: &[bool], slo: f64) -> Result<Availability> {
    if success.is_empty() {
        return invalid("success is empty. Provide at least one example.");
    }
    if !(slo > 0.0 && slo < 1.0) {
        return invalid("slo must be in (0, 1).");
    }
    let r = rate(success);
    let err = 1.0 - r.value;
    Ok(Availability {
        rate: r,
        error_rate: err,
        error_budget_remaining: 1.0 - err / (1.0 - slo),
    })
}
