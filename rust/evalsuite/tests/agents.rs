mod common;

use common::{check, nums, Got};
use evalsuite::agents::*;
use serde_json::{json, Map, Value};

fn trajs(v: &Value) -> Vec<Vec<ToolCall>> {
    v.as_array().unwrap().iter().map(|t| t.as_array().unwrap().iter().map(|c| ToolCall::from_json(c).unwrap()).collect()).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn states(v: &Value) -> Vec<Map<String, Value>> {
    v.as_array().unwrap().iter().map(|x| x.as_object().unwrap().clone()).collect()
}

fn g<T>(r: evalsuite::Result<T>, f: impl Fn(T) -> f64) -> Got {
    match r {
        Ok(x) => Got::Num(f(x)),
        Err(e) => Got::Err(e.to_string()),
    }
}

#[test]
fn agents_parity() {
    let tools = json!({
        "search": {"type": "object", "required": ["q"], "properties": {"q": {"type": "string", "minLength": 1}, "k": {"type": "integer", "minimum": 1}}, "additionalProperties": false},
        "calc": {"type": "object", "properties": {"expr": {"type": "string", "pattern": "^[0-9+*/ -]+$"}}},
        "open": null,
    });
    let tools = tools.as_object().unwrap();
    let (n, _) = check("agents", 1e-12, |f, a| {
        let (name, key) = f.split_once('.').unwrap_or((f, "value"));
        match name {
            "invalid_tool_call_rate" => g(invalid_tool_call_rate(&trajs(&a["trajectories"]), tools), |r| match key {
                "schema" => r.schema,
                "unknown_tool" => r.unknown_tool,
                "execution_failure_rate" => r.execution_failure_rate,
                "ci_low" => r.rate.ci_low,
                _ => r.rate.value,
            }),
            "tool_failure_recovery_rate" => g(tool_failure_recovery_rate(&trajs(&a["trajectories"])), |r| r.value),
            "unnecessary_tool_call_rate" => g(
                unnecessary_tool_call_rate(&trajs(&a["trajectories"]), a["loop_length"].as_u64().unwrap() as usize),
                |r| if key == "loop_rate" { r.loop_rate } else { r.rate.value },
            ),
            "task_completion_rate" => {
                let p = a.get("progress").map(nums);
                g(task_completion_rate(&bools(&a["completed"]), p.as_deref()), |r| if key == "progress_rate" { r.progress_rate.unwrap() } else { r.rate.value })
            }
            "tool_use_efficiency" => g(tool_use_efficiency(&nums(&a["n_calls"]), &nums(&a["optimal_calls"])), |r| {
                if key == "mean_excess_calls" {
                    r.mean_excess_calls
                } else {
                    r.value
                }
            }),
            "steps_per_task" => g(steps_per_task(&nums(&a["steps"]), None, Some(&bools(&a["completed"]))), |r| match key {
                "std" => r.steps.std,
                "p90" => r.steps.p90,
                "p99" => r.steps.p99,
                "mean_steps_completed" => r.mean_steps_completed.unwrap(),
                _ => r.value,
            }),
            "human_intervention_rate" => g(human_intervention_rate(&nums(&a["interventions"])), |r| r.0.value),
            "agent_cost_per_task" => g(
                agent_cost_per_task(&nums(&a["costs"]), Some(&nums(&a["durations"])), Some(&bools(&a["completed"]))),
                |r| match key {
                    "cost_per_success" => r.cost_per_success.unwrap(),
                    "p50" => r.time.unwrap().p50,
                    _ => r.value,
                },
            ),
            "plan_adherence" => {
                let s = |v: &Value| -> Vec<Vec<String>> { v.as_array().unwrap().iter().map(common::strs).collect() };
                g(plan_adherence(&s(&a["plans"]), &s(&a["executed"])), |r| match key {
                    "precision" => r.precision,
                    "recall" => r.recall,
                    "exact_match" => r.exact_match,
                    _ => r.value,
                })
            }
            "state_tracking_accuracy" => g(state_tracking_accuracy(&states(&a["expected"]), &states(&a["predicted"])), |r| {
                if key == "slot_accuracy" {
                    r.slot_accuracy
                } else {
                    r.value
                }
            }),
            _ => Got::Skip,
        }
    });
    assert!(n > 1500);
}
