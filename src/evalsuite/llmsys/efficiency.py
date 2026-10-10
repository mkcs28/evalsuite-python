"""Inference efficiency and operational cost (v0.5.0): time to first token, time per output token, end-to-end
latency and its percentiles, throughput, token counts, cost, resource utilisation, energy, request rate and
availability. Inputs are the timestamps, counts and readings your serving stack or load test records; times are
in seconds unless stated.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import floats, rate, same_length, seq, summary

__all__ = [
    "availability",
    "energy_per_request",
    "inference_cost",
    "latency_percentiles",
    "requests_per_second",
    "resource_utilization",
    "throughput",
    "time_per_output_token",
    "time_to_first_token",
    "token_usage",
]

_C = "efficiency"
_REF_VLLM = (
    "Kwon W, Li Z, Zhuang S, et al. Efficient memory management for large language model serving with "
    "PagedAttention. SOSP. 2023:611-626."
)
_REF_MLPERF = "Reddi VJ, Cheng C, Kanter D, et al. MLPerf inference benchmark. ISCA. 2020:446-459."
_REF_DEAN = "Dean J, Barroso LA. The tail at scale. Commun ACM. 2013;56(2):74-80."
_REF_LUCCIONI = (
    "Luccioni S, Jernite Y, Strubell E. Power hungry processing: watts driving the cost of AI deployment? ACM "
    "FAccT. "
    "2024:85-99."
)
_REF_SRE = "Beyer B, Jones C, Petoff J, Murphy NR. Site Reliability Engineering. O'Reilly; 2016."


def _durations(start: Any, end: Any, a: str, b: str) -> np.ndarray:
    s, e = floats(start, a), floats(end, b)
    same_length((a, s), (b, e))
    d = e - s
    if np.any(d < 0):
        raise InputValidationError(f"{b} must not be earlier than {a}.")
    return np.asarray(d, dtype=np.float64)


def _stat_result(
    metric: str, name: str, x: np.ndarray, unit: str, extra: Optional[dict[str, Any]] = None
) -> MetricResult:
    return MetricResult(metric, name, float(x.mean()), {"unit": unit, **summary(x), **(extra or {})})


@register(
    category=_C,
    task="latency",
    name="Time to First Token (TTFT)",
    definition="Time from sending a request to receiving the first output token; mean with p50/p90/p95/p99.",
    formula="TTFT = t_first_token − t_request",
    range="[0, ∞) s",
    input_requirements=("request_times", "first_token_times"),
    references=(_REF_VLLM, _REF_MLPERF),
    higher_is_better=False,
)
def time_to_first_token(request_times: Any, first_token_times: Any) -> MetricResult:
    return _stat_result(
        "time_to_first_token",
        "Time to first token",
        _durations(request_times, first_token_times, "request_times", "first_token_times"),
        "s",
    )


@register(
    category=_C,
    task="latency",
    name="Time per Output Token (TPOT)",
    definition="Mean time between output tokens after the first (inter-token latency), per request: "
    "(t_end − t_first) / (output tokens − 1).",
    formula="TPOT = (t_end − t_first) / (n_out − 1)",
    range="[0, ∞) s",
    input_requirements=("first_token_times", "end_times", "output_tokens"),
    references=(_REF_VLLM, _REF_MLPERF),
    higher_is_better=False,
)
def time_per_output_token(first_token_times: Any, end_times: Any, output_tokens: Any) -> MetricResult:
    d = _durations(first_token_times, end_times, "first_token_times", "end_times")
    n = floats(output_tokens, "output_tokens", lo=0)
    same_length(("end_times", d), ("output_tokens", n))
    keep = n > 1
    if not keep.any():
        raise InputValidationError("TPOT needs requests with at least two output tokens.")
    return _stat_result(
        "time_per_output_token", "Time per output token", d[keep] / (n[keep] - 1), "s", {"n_used": int(keep.sum())}
    )


@register(
    category=_C,
    task="latency",
    name="p50 / p95 / p99 latency",
    definition="End-to-end request latency percentiles; tail latency (p95, p99) governs user experience at "
    "scale. The value is p95; mean, p50, p90 and p99 are in ``params``.",
    formula="p-th percentile of latency (linear interpolation)",
    range="[0, ∞)",
    input_requirements=("latencies",),
    references=(_REF_DEAN,),
    higher_is_better=False,
)
def latency_percentiles(latencies: Any, *, unit: str = "s", percentile: float = 95) -> MetricResult:
    x = floats(latencies, "latencies", lo=0)
    if not 0 <= percentile <= 100:
        raise InputValidationError("percentile must be in [0, 100].")
    s = summary(x)
    return MetricResult(
        "latency_percentiles", f"p{percentile:g} latency", float(np.percentile(x, percentile)), {"unit": unit, **s}
    )


@register(
    category=_C,
    task="throughput",
    name="Throughput / tokens per second",
    definition="Output tokens generated per second: system throughput over the measurement window (total tokens / "
    "wall-clock span), and the mean per-request decode speed when request durations are given.",
    formula="Σ output tokens / (t_last_end − t_first_start)",
    range="[0, ∞) tokens/s",
    input_requirements=("output_tokens", "start_times", "end_times"),
    references=(_REF_VLLM, _REF_MLPERF),
)
def throughput(output_tokens: Any, start_times: Any, end_times: Any) -> MetricResult:
    n = floats(output_tokens, "output_tokens", lo=0)
    d = _durations(start_times, end_times, "start_times", "end_times")
    same_length(("output_tokens", n), ("end_times", d))
    s, e = floats(start_times, "start_times"), floats(end_times, "end_times")
    span = float(e.max() - s.min())
    if span <= 0:
        raise InputValidationError("The measurement window has zero length.")
    per = n[d > 0] / d[d > 0]
    return MetricResult(
        "throughput",
        "Throughput (tokens/s)",
        float(n.sum() / span),
        {"per_request_tokens_per_second": float(per.mean()) if per.size else float("nan"), "window_s": span},
    )


@register(
    category=_C,
    task="usage",
    name="Input / output token counts",
    definition="Token usage per request: mean input and output tokens (the value is mean total tokens), with "
    "percentiles and totals.",
    formula="mean_r (input_r + output_r)",
    range="[0, ∞)",
    input_requirements=("input_tokens", "output_tokens"),
    references=(_REF_MLPERF,),
    higher_is_better=False,
)
def token_usage(input_tokens: Any, output_tokens: Any) -> MetricResult:
    i = floats(input_tokens, "input_tokens", lo=0)
    o = floats(output_tokens, "output_tokens", lo=0)
    same_length(("input_tokens", i), ("output_tokens", o))
    return MetricResult(
        "token_usage",
        "Tokens per request",
        float((i + o).mean()),
        {"input": summary(i), "output": summary(o), "total_input": float(i.sum()), "total_output": float(o.sum())},
    )


@register(
    category=_C,
    task="cost",
    name="Cost per request / per 1,000 tokens",
    definition="Monetary cost from token counts and per-million-token prices (input and output priced "
    "separately): mean cost per request (the value) and blended cost per 1,000 tokens.",
    formula="cost_r = in_r · p_in / 10⁶ + out_r · p_out / 10⁶",
    range="[0, ∞) currency units",
    input_requirements=("input_tokens", "output_tokens", "input_price", "output_price"),
    references=(_REF_LUCCIONI,),
    higher_is_better=False,
)
def inference_cost(
    input_tokens: Any,
    output_tokens: Any,
    *,
    input_price: float,
    output_price: float,
    cached_tokens: Any = None,
    cached_price: Optional[float] = None,
) -> MetricResult:
    """Prices are per million tokens. ``cached_tokens`` (part of the input billed at ``cached_price``)."""
    i = floats(input_tokens, "input_tokens", lo=0)
    o = floats(output_tokens, "output_tokens", lo=0)
    same_length(("input_tokens", i), ("output_tokens", o))
    if input_price < 0 or output_price < 0:
        raise InputValidationError("Prices must be non-negative.")
    cost = i * input_price / 1e6 + o * output_price / 1e6
    if cached_tokens is not None:
        if cached_price is None or cached_price < 0:
            raise InputValidationError(
                "cached_price (non-negative, per million tokens) is required with cached_tokens."
            )
        c = floats(cached_tokens, "cached_tokens", lo=0)
        same_length(("input_tokens", i), ("cached_tokens", c))
        if np.any(c > i):
            raise InputValidationError("cached_tokens cannot exceed input_tokens.")
        cost = cost - c * (input_price - cached_price) / 1e6
    tokens = float((i + o).sum())
    return MetricResult(
        "inference_cost",
        "Cost per request",
        float(cost.mean()),
        {
            "total_cost": float(cost.sum()),
            "cost_per_1k_tokens": float(cost.sum() / tokens * 1000) if tokens else float("nan"),
        },
    )


@register(
    category=_C,
    task="resources",
    name="Peak memory / CPU / GPU utilization",
    definition="From periodic resource readings (memory in GB, CPU and GPU utilisation in %), the peak and mean "
    "of each series; the value is peak memory when memory readings are given, otherwise the first series' peak.",
    formula="peak = max_t reading_t; mean = mean_t reading_t",
    range="[0, ∞)",
    input_requirements=("readings",),
    references=(_REF_MLPERF,),
    higher_is_better=False,
)
def resource_utilization(readings: Mapping[str, Any]) -> MetricResult:
    """``readings``: mapping series name (``"memory_gb"``, ``"gpu_util"``, ``"cpu_util"``...) -> samples."""
    if not isinstance(readings, Mapping) or not readings:
        raise InputValidationError("readings must map a series name to its samples.")
    out = {}
    for k, v in readings.items():
        x = floats(v, f"readings[{k!r}]", lo=0)
        out[str(k)] = {"peak": float(x.max()), "mean": float(x.mean()), "p95": float(np.percentile(x, 95))}
    key = next((k for k in out if "mem" in k.lower()), next(iter(out)))
    return MetricResult("resource_utilization", f"Peak {key}", out[key]["peak"], {"series": out, "primary": key})


@register(
    category=_C,
    task="energy",
    name="Energy per request",
    definition="Energy consumed per request in watt-hours, from power readings sampled at a fixed interval over "
    "the run (trapezoidal integration) or from a measured total, divided by the requests served.",
    formula="E = ∫ P dt / 3600 / n_requests",
    range="[0, ∞) Wh",
    input_requirements=("power_watts", "interval_s", "n_requests"),
    references=(_REF_LUCCIONI,),
    higher_is_better=False,
)
def energy_per_request(
    power_watts: Any = None,
    *,
    interval_s: Optional[float] = None,
    n_requests: int,
    total_joules: Optional[float] = None,
) -> MetricResult:
    if isinstance(n_requests, bool) or not isinstance(n_requests, (int, np.integer)) or n_requests < 1:
        raise InputValidationError("n_requests must be a positive integer.")
    if total_joules is not None:
        if total_joules < 0:
            raise InputValidationError("total_joules must be non-negative.")
        joules = float(total_joules)
    else:
        if power_watts is None or interval_s is None or interval_s <= 0:
            raise InputValidationError("Pass power_watts with a positive interval_s, or total_joules.")
        p = floats(power_watts, "power_watts", lo=0)
        joules = float(np.sum((p[1:] + p[:-1]) / 2) * interval_s) if p.size > 1 else float(p[0] * interval_s)
    wh = joules / 3600
    return MetricResult(
        "energy_per_request",
        "Energy per request (Wh)",
        wh / n_requests,
        {"total_wh": wh, "joules_per_request": joules / n_requests},
    )


@register(
    category=_C,
    task="throughput",
    name="Requests per second",
    definition="Completed requests per second over the measurement window, from request completion times.",
    formula="(n − 1) / (t_last − t_first)",
    range="[0, ∞)",
    input_requirements=("completion_times",),
    references=(_REF_MLPERF,),
)
def requests_per_second(completion_times: Any, *, start_time: Optional[float] = None) -> MetricResult:
    """With ``start_time`` (when the load test began) the rate is n / (t_last − start_time)."""
    t = np.sort(floats(completion_times, "completion_times"))
    if start_time is not None:
        span, n = float(t[-1] - start_time), t.size
    else:
        if t.size < 2:
            raise InputValidationError("Need at least two completions (or pass start_time).")
        span, n = float(t[-1] - t[0]), t.size - 1
    if span <= 0:
        raise InputValidationError("The measurement window has zero length.")
    return MetricResult(
        "requests_per_second", "Requests per second", n / span, {"window_s": span, "completed": int(t.size)}
    )


@register(
    category=_C,
    task="reliability",
    name="Error rate / availability",
    definition="Availability: share of requests served successfully (the value), with the error rate and the "
    "error budget remaining against a service-level objective.",
    formula="availability = successful / total; budget = 1 − (1 − availability) / (1 − SLO)",
    range="[0, 1]",
    input_requirements=("success",),
    references=(_REF_SRE,),
)
def availability(success: Any, *, slo: float = 0.999) -> MetricResult:
    items = seq(success, "success")
    ok = np.array(
        [
            s is True or s == 1 or (isinstance(s, (int, np.integer)) and 200 <= int(s) < 400) or s == "ok"
            for s in items
        ]
    )
    if not 0 < slo < 1:
        raise InputValidationError("slo must be in (0, 1).")
    r = rate("availability", "Availability", ok)
    err = 1 - float(ok.mean())
    return MetricResult(
        r.metric,
        r.name,
        r.value,
        {**r.params, "error_rate": err, "slo": slo, "error_budget_remaining": 1 - err / (1 - slo)},
    )
