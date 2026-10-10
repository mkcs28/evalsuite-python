"""Agent and tool-use evaluation (v0.5.0).

A trajectory is the list of tool calls an agent made for one task, each a dict with ``"name"``, optional
``"arguments"`` (dict or JSON string) and optional ``"ok"`` (whether the call executed successfully). Tool-call
precision / recall / F1 and tool-selection and argument accuracy against a reference are the existing
``es.tool_call_f1``, ``es.tool_selection_accuracy`` and ``es.tool_argument_accuracy``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ..text.structured import validate_json_schema
from ._common import bools, floats, lists_of, rate, same_length, seq, summary

__all__ = [
    "agent_cost_per_task",
    "human_intervention_rate",
    "invalid_tool_call_rate",
    "plan_adherence",
    "state_tracking_accuracy",
    "steps_per_task",
    "task_completion_rate",
    "tool_failure_recovery_rate",
    "tool_use_efficiency",
    "unnecessary_tool_call_rate",
]

_C = "agents"
_REF_AGENTBENCH = "Liu X, Yu H, Zhang H, et al. AgentBench: evaluating LLMs as agents. ICLR. 2024."
_REF_AGENTBOARD = (
    "Ma C, Zhang J, Zhu Z, et al. AgentBoard: an analytical evaluation board of multi-turn LLM agents. NeurIPS "
    "Datasets and Benchmarks. 2024."
)
_REF_TAU = (
    "Yao S, Shinn N, Razavi P, Narasimhan K. τ-bench: a benchmark for tool-agent-user interaction in real-world "
    "domains. arXiv:2406.12045. 2024."
)
_REF_BFCL = (
    "Patil SG, Mao H, Cheng-Jie Ji C, et al. The Berkeley Function Calling Leaderboard (BFCL): from tool use to "
    "agentic evaluation of large language models. ICML. 2025."
)
_REF_DST = (
    "Mrkšić N, Ó Séaghdha D, Wen TH, Thomson B, Young S. Neural belief tracker: data-driven dialogue state "
    "tracking. ACL. 2017:1777-1788."
)
_REF_TOOLEMU = (
    "Ruan Y, Dong H, Wang A, et al. Identifying the risks of LM agents with an LM-emulated sandbox. ICLR. 2024."
)


def _trajectories(x: Any, name: str = "trajectories") -> list[list[dict[str, Any]]]:
    out = []
    for i, traj in enumerate(lists_of(x, name)):
        calls = []
        for j, c in enumerate(traj):
            if not isinstance(c, Mapping) or "name" not in c:
                raise InputValidationError(f"{name}[{i}][{j}] must be a dict with a 'name'.")
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {"__unparseable__": args}
            calls.append({"name": c["name"], "arguments": args, "ok": c.get("ok")})
        out.append(calls)
    return out


def _key(call: Mapping[str, Any]) -> str:
    return json.dumps([call["name"], call["arguments"]], sort_keys=True, default=str)


@register(
    category=_C,
    task="agents",
    name="Task / goal completion rate",
    definition="Share of tasks the agent completes (its goal state reached or judged successful), with the mean "
    "progress rate when partial progress is scored (AgentBoard).",
    formula="completed tasks / tasks",
    range="[0, 1]",
    input_requirements=("completed",),
    references=(_REF_AGENTBENCH, _REF_AGENTBOARD),
)
def task_completion_rate(completed: Any, *, progress: Any = None) -> MetricResult:
    c = bools(completed, "completed")
    extra: dict[str, Any] = {}
    if progress is not None:
        p = floats(progress, "progress", lo=0, hi=1)
        same_length(("completed", c), ("progress", p))
        extra["progress_rate"] = float(p.mean())
    return rate("task_completion_rate", "Task completion rate", c, extra)


@register(
    category=_C,
    task="tool-use",
    name="Invalid tool-call / execution failure rate",
    definition="Share of tool calls that are invalid (unknown tool, unparseable or schema-violating arguments, "
    "checked against each tool's JSON Schema) and, where execution results are recorded, the share that failed "
    "to execute.",
    formula="invalid calls / calls",
    range="[0, 1]",
    input_requirements=("trajectories", "tools"),
    references=(_REF_BFCL,),
    higher_is_better=False,
)
def invalid_tool_call_rate(trajectories: Any, tools: Mapping[str, Any]) -> MetricResult:
    """``tools``: tool name -> JSON Schema of its arguments (``{}`` accepts anything)."""
    if not isinstance(tools, Mapping) or not tools:
        raise InputValidationError("tools must be a non-empty mapping tool name -> argument JSON Schema.")
    trajs = _trajectories(trajectories)
    invalid, failed, executed = [], 0, 0
    reasons = {"unknown_tool": 0, "unparseable_arguments": 0, "schema": 0}
    for traj in trajs:
        for call in traj:
            bad = False
            if call["name"] not in tools:
                reasons["unknown_tool"] += 1
                bad = True
            elif isinstance(call["arguments"], Mapping) and "__unparseable__" in call["arguments"]:
                reasons["unparseable_arguments"] += 1
                bad = True
            elif validate_json_schema(call["arguments"], tools[call["name"]] or {}):
                reasons["schema"] += 1
                bad = True
            invalid.append(bad)
            if call["ok"] is not None:
                executed += 1
                failed += not bool(call["ok"])
    if not invalid:
        raise InputValidationError("The trajectories contain no tool calls.")
    n = len(invalid)
    extra = {
        "by_reason": {k: v / n for k, v in reasons.items()},
        "execution_failure_rate": failed / executed if executed else float("nan"),
    }
    return rate("invalid_tool_call_rate", "Invalid tool-call rate", np.array(invalid), extra)


@register(
    category=_C,
    task="tool-use",
    name="Tool-use efficiency",
    definition="How close the number of tool calls is to the minimum needed: the mean of optimal / actual calls "
    "per task (1 = no wasted calls), with the mean excess calls.",
    formula="mean_t min(1, optimal_t / actual_t)",
    range="[0, 1]",
    input_requirements=("n_calls", "optimal_calls"),
    references=(_REF_AGENTBOARD,),
)
def tool_use_efficiency(n_calls: Any, optimal_calls: Any) -> MetricResult:
    a = floats(n_calls, "n_calls", lo=0)
    o = floats(optimal_calls, "optimal_calls", lo=0)
    same_length(("n_calls", a), ("optimal_calls", o))
    eff = np.where(a == 0, np.where(o == 0, 1.0, 0.0), np.minimum(1.0, o / np.where(a == 0, 1, a)))
    return MetricResult(
        "tool_use_efficiency",
        "Tool-use efficiency",
        float(eff.mean()),
        {"mean_excess_calls": float(np.maximum(0, a - o).mean()), "n_tasks": int(a.size)},
    )


@register(
    category=_C,
    task="agents",
    name="Steps / tool calls per task",
    definition="Distribution of the number of steps (and tool calls) per task: mean, median and high "
    "percentiles, overall and for completed tasks only.",
    formula="mean_t steps_t",
    range="[0, ∞)",
    input_requirements=("steps",),
    references=(_REF_AGENTBOARD, _REF_TAU),
    higher_is_better=False,
)
def steps_per_task(steps: Any, *, tool_calls: Any = None, completed: Any = None) -> MetricResult:
    s = floats(steps, "steps", lo=0)
    extra: dict[str, Any] = {"steps": summary(s)}
    if tool_calls is not None:
        t = floats(tool_calls, "tool_calls", lo=0)
        same_length(("steps", s), ("tool_calls", t))
        extra["tool_calls"] = summary(t)
    if completed is not None:
        c = bools(completed, "completed")
        same_length(("steps", s), ("completed", c))
        extra["mean_steps_completed"] = float(s[c].mean()) if c.any() else float("nan")
    return MetricResult("steps_per_task", "Steps per task", float(s.mean()), extra)


def _lcs(a: Sequence[Any], b: Sequence[Any]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


@register(
    category=_C,
    task="planning",
    name="Planning accuracy / plan adherence",
    definition="Agreement between the steps executed and a reference plan, in order: precision and recall of the "
    "longest common subsequence, their F1 (the value), and the exact-plan match rate.",
    formula="P = LCS / |executed|, R = LCS / |plan|, F1 = 2PR / (P + R)",
    range="[0, 1]",
    input_requirements=("plans", "executed"),
    references=(_REF_AGENTBOARD,),
)
def plan_adherence(plans: Any, executed: Any) -> MetricResult:
    """``plans`` and ``executed``: per task, a list of step identifiers (tool names, action labels, ...)."""
    ps, es_ = lists_of(plans, "plans"), lists_of(executed, "executed")
    same_length(("plans", ps), ("executed", es_))
    f1s, prec, rec = [], [], []
    for p, e in zip(ps, es_):
        lcs = _lcs(p, e)
        pr = lcs / len(e) if e else float(not p)
        rc = lcs / len(p) if p else 1.0
        prec.append(pr)
        rec.append(rc)
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return MetricResult(
        "plan_adherence",
        "Plan adherence (F1)",
        float(np.mean(f1s)),
        {
            "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)),
            "exact_match": float(np.mean([list(p) == list(e) for p, e in zip(ps, es_)])),
        },
    )


@register(
    category=_C,
    task="state-tracking",
    name="State-tracking accuracy",
    definition="Dialogue / environment state tracking: joint goal accuracy (every slot of the state correct at a "
    "turn) and slot accuracy (share of slots correct over the union of expected and predicted slots).",
    formula="JGA = mean_turns 1[state = expected]",
    range="[0, 1]",
    input_requirements=("expected_states", "predicted_states"),
    references=(_REF_DST,),
)
def state_tracking_accuracy(expected_states: Any, predicted_states: Any) -> MetricResult:
    e, p = seq(expected_states, "expected_states"), seq(predicted_states, "predicted_states")
    same_length(("expected_states", e), ("predicted_states", p))
    joint, slots_ok, slots_n = [], 0, 0
    for i, (a, b) in enumerate(zip(e, p)):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            raise InputValidationError(f"States at turn {i} must be dicts slot -> value.")
        joint.append(dict(a) == dict(b))
        keys = set(a) | set(b)
        slots_n += len(keys)
        slots_ok += sum(k in a and k in b and a[k] == b[k] for k in keys)
    return MetricResult(
        "state_tracking_accuracy",
        "Joint goal accuracy",
        float(np.mean(joint)),
        {"slot_accuracy": slots_ok / slots_n if slots_n else 1.0, "n_turns": len(e)},
    )


@register(
    category=_C,
    task="tool-use",
    name="Recovery from tool failures",
    definition="Among failed tool calls, the share after which the agent later succeeded with the same tool in "
    "the same task (retried correctly or fixed the arguments).",
    formula="#failed calls followed by a successful call of that tool / #failed calls",
    range="[0, 1]",
    input_requirements=("trajectories",),
    references=(_REF_TAU, _REF_TOOLEMU),
)
def tool_failure_recovery_rate(trajectories: Any) -> MetricResult:
    flags = []
    for traj in _trajectories(trajectories):
        for i, call in enumerate(traj):
            if call["ok"] is False:
                flags.append(any(c["name"] == call["name"] and c["ok"] is True for c in traj[i + 1 :]))
    if not flags:
        raise InputValidationError("No tool call failed (no 'ok': False): the recovery rate is undefined.")
    return rate("tool_failure_recovery_rate", "Tool-failure recovery rate", np.array(flags))


@register(
    category=_C,
    task="tool-use",
    name="Unnecessary tool-call / loop rate",
    definition="Share of tool calls that exactly repeat an earlier call (same tool and arguments) in the same "
    "task, "
    "and the share of tasks stuck in a loop (the same call repeated consecutively at least ``loop_length`` "
    "times).",
    formula="repeated calls / calls",
    range="[0, 1]",
    input_requirements=("trajectories",),
    references=(_REF_AGENTBOARD,),
    higher_is_better=False,
)
def unnecessary_tool_call_rate(trajectories: Any, *, loop_length: int = 3) -> MetricResult:
    if isinstance(loop_length, bool) or not isinstance(loop_length, int) or loop_length < 2:
        raise InputValidationError("loop_length must be an integer >= 2.")
    repeats, total, loops = [], 0, []
    trajs = _trajectories(trajectories)
    for traj in trajs:
        seen: set[str] = set()
        run, longest, prev = 0, 0, None
        for call in traj:
            k = _key(call)
            repeats.append(k in seen)
            seen.add(k)
            run = run + 1 if k == prev else 1
            longest = max(longest, run)
            prev = k
            total += 1
        loops.append(longest >= loop_length)
    if not total:
        raise InputValidationError("The trajectories contain no tool calls.")
    return rate(
        "unnecessary_tool_call_rate",
        "Unnecessary tool-call rate",
        np.array(repeats),
        {"loop_rate": float(np.mean(loops)), "loop_length": loop_length},
    )


@register(
    category=_C,
    task="agents",
    name="Human intervention rate",
    definition="Share of tasks that needed at least one human intervention (correction, approval override, "
    "takeover), with the mean number of interventions per task.",
    formula="tasks with ≥1 intervention / tasks",
    range="[0, 1]",
    input_requirements=("interventions",),
    references=(_REF_TAU,),
    higher_is_better=False,
)
def human_intervention_rate(interventions: Any) -> MetricResult:
    n = floats(interventions, "interventions", lo=0)
    return rate(
        "human_intervention_rate",
        "Human intervention rate",
        n > 0,
        {"mean_per_task": float(n.mean())},
    )


@register(
    category=_C,
    task="agents",
    name="End-to-end execution time / cost per task",
    definition="Mean wall-clock time and monetary cost per task, and the cost per successful task (total cost "
    "divided by completed tasks), the figure that matters when failures are retried.",
    formula="cost per success = Σ cost / #completed",
    range="[0, ∞)",
    input_requirements=("costs",),
    references=(_REF_AGENTBENCH,),
    higher_is_better=False,
)
def agent_cost_per_task(costs: Any, *, durations: Any = None, completed: Any = None) -> MetricResult:
    c = floats(costs, "costs", lo=0)
    extra: dict[str, Any] = {"total_cost": float(c.sum())}
    if durations is not None:
        d = floats(durations, "durations", lo=0)
        same_length(("costs", c), ("durations", d))
        extra["time"] = summary(d)
    if completed is not None:
        ok = bools(completed, "completed")
        same_length(("costs", c), ("completed", ok))
        extra["cost_per_success"] = float(c.sum()) / int(ok.sum()) if ok.any() else float("inf")
        extra["success_rate"] = float(ok.mean())
    return MetricResult("agent_cost_per_task", "Cost per task", float(c.mean()), extra)
