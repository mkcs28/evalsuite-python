//! Agent metrics (task completion, tool-call validity and efficiency, plan adherence, state tracking, failure
//! recovery, loops, human intervention, cost), as EvalSuite's Python `evalsuite.llmsys.agents`.
//!
//! A trajectory is the list of [`ToolCall`]s an agent made for one task.
//!
//! ```
//! use evalsuite::agents::{tool_failure_recovery_rate, ToolCall};
//! use serde_json::json;
//! let traj = vec![vec![
//!     ToolCall::new("search", json!({"q": "x"})).ok(false),
//!     ToolCall::new("search", json!({"q": "x y"})).ok(true),
//! ]];
//! assert_eq!(tool_failure_recovery_rate(&traj).unwrap().value, 1.0);
//! ```

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::error::{invalid, Result};
use crate::json_schema::{canonical, py_equal, validate};
use crate::summary::{floats, nonempty, rate, same, Rate, Summary};
use crate::util::mean;

/// One tool call: its name, arguments (a JSON value) and, optionally, whether it executed successfully.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
    pub ok: Option<bool>,
}

impl ToolCall {
    /// A call with JSON arguments. A string is parsed as JSON (an empty string is `{}`); a string that does not
    /// parse becomes `{"__unparseable__": <string>}`, which counts as an invalid call.
    pub fn new(name: impl Into<String>, arguments: Value) -> ToolCall {
        let arguments = match arguments {
            Value::String(s) if s.trim().is_empty() => Value::Object(Map::new()),
            Value::String(s) => serde_json::from_str(&s).unwrap_or_else(|_| {
                let mut m = Map::new();
                m.insert("__unparseable__".into(), Value::String(s));
                Value::Object(m)
            }),
            other => other,
        };
        ToolCall {
            name: name.into(),
            arguments,
            ok: None,
        }
    }

    /// Sets whether the call executed successfully.
    pub fn ok(mut self, ok: bool) -> ToolCall {
        self.ok = Some(ok);
        self
    }

    /// A call from a JSON object `{"name": .., "arguments"?: .., "ok"?: bool}`.
    pub fn from_json(v: &Value) -> Result<ToolCall> {
        let Some(o) = v.as_object() else {
            return invalid("A tool call must be an object with a 'name'.");
        };
        let Some(name) = o.get("name").and_then(Value::as_str) else {
            return invalid("A tool call must be an object with a 'name'.");
        };
        let mut c = ToolCall::new(
            name,
            o.get("arguments")
                .cloned()
                .unwrap_or(Value::Object(Map::new())),
        );
        c.ok = o.get("ok").and_then(Value::as_bool);
        Ok(c)
    }

    /// The call's identity: name and arguments, canonically encoded.
    pub fn key(&self) -> String {
        canonical(&Value::Array(vec![
            Value::String(self.name.clone()),
            self.arguments.clone(),
        ]))
    }
}

fn trajectories(t: &[Vec<ToolCall>]) -> Result<()> {
    nonempty(t, "trajectories")
}

fn bool_mean(b: &[bool]) -> f64 {
    b.iter().filter(|&&x| x).count() as f64 / b.len() as f64
}

/// Task completion rate, with the mean partial progress (in [0, 1]) when given.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskCompletion {
    pub rate: Rate,
    pub progress_rate: Option<f64>,
}

/// Share of tasks completed.
pub fn task_completion_rate(
    completed: &[bool],
    progress: Option<&[f64]>,
) -> Result<TaskCompletion> {
    nonempty(completed, "completed")?;
    let progress_rate = match progress {
        Some(p) => {
            floats(p, "progress", Some(0.0), Some(1.0))?;
            same(completed.len(), p.len(), "completed", "progress")?;
            Some(mean(p))
        }
        None => None,
    };
    Ok(TaskCompletion {
        rate: rate(completed),
        progress_rate,
    })
}

/// Invalid tool-call rate with the share of calls per reason and the execution failure rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidCalls {
    pub rate: Rate,
    pub unknown_tool: f64,
    pub unparseable_arguments: f64,
    pub schema: f64,
    /// Failed executions among the calls with a known outcome (NaN when none has one).
    pub execution_failure_rate: f64,
}

/// Share of calls to an unknown tool, with unparseable arguments, or with arguments failing the tool's JSON
/// Schema. `tools` maps tool name → argument schema (`null` or `{}` accepts anything).
pub fn invalid_tool_call_rate(
    trajectories_: &[Vec<ToolCall>],
    tools: &Map<String, Value>,
) -> Result<InvalidCalls> {
    if tools.is_empty() {
        return invalid("tools must be a non-empty mapping tool name -> argument JSON Schema.");
    }
    trajectories(trajectories_)?;
    let (mut flags, mut failed, mut executed) = (Vec::new(), 0usize, 0usize);
    let (mut unknown, mut unparse, mut schema) = (0usize, 0usize, 0usize);
    let any = Value::Object(Map::new());
    for call in trajectories_.iter().flatten() {
        let bad = match tools.get(&call.name) {
            None => {
                unknown += 1;
                true
            }
            Some(_)
                if call
                    .arguments
                    .as_object()
                    .is_some_and(|o| o.contains_key("__unparseable__")) =>
            {
                unparse += 1;
                true
            }
            Some(s) => {
                let s = match s {
                    Value::Null | Value::Bool(false) => &any,
                    Value::Object(o) if o.is_empty() => &any,
                    s => s,
                };
                let e = !validate(&call.arguments, s)?.is_empty();
                schema += usize::from(e);
                e
            }
        };
        flags.push(bad);
        if let Some(ok) = call.ok {
            executed += 1;
            failed += usize::from(!ok);
        }
    }
    if flags.is_empty() {
        return invalid("The trajectories contain no tool calls.");
    }
    let n = flags.len() as f64;
    Ok(InvalidCalls {
        rate: rate(&flags),
        unknown_tool: unknown as f64 / n,
        unparseable_arguments: unparse as f64 / n,
        schema: schema as f64 / n,
        execution_failure_rate: if executed > 0 {
            failed as f64 / executed as f64
        } else {
            f64::NAN
        },
    })
}

/// Mean tool-use efficiency min(1, optimal / used) and the mean number of excess calls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Efficiency {
    pub value: f64,
    pub mean_excess_calls: f64,
    pub n_tasks: usize,
}

/// Tool-use efficiency per task: min(1, optimal / used); a task with no calls scores 1 if none were needed.
pub fn tool_use_efficiency(n_calls: &[f64], optimal_calls: &[f64]) -> Result<Efficiency> {
    floats(n_calls, "n_calls", Some(0.0), None)?;
    floats(optimal_calls, "optimal_calls", Some(0.0), None)?;
    same(
        n_calls.len(),
        optimal_calls.len(),
        "n_calls",
        "optimal_calls",
    )?;
    let eff: Vec<f64> = n_calls
        .iter()
        .zip(optimal_calls)
        .map(|(&a, &o)| {
            if a == 0.0 {
                f64::from(u8::from(o == 0.0))
            } else {
                (o / a).min(1.0)
            }
        })
        .collect();
    let excess: Vec<f64> = n_calls
        .iter()
        .zip(optimal_calls)
        .map(|(a, o)| (a - o).max(0.0))
        .collect();
    Ok(Efficiency {
        value: mean(&eff),
        mean_excess_calls: mean(&excess),
        n_tasks: eff.len(),
    })
}

/// Mean steps per task, with distribution summaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Steps {
    pub value: f64,
    pub steps: Summary,
    pub tool_calls: Option<Summary>,
    /// Mean steps of the completed tasks (NaN when none completed), when `completed` is given.
    pub mean_steps_completed: Option<f64>,
}

/// Steps (and optionally tool calls) per task.
pub fn steps_per_task(
    steps: &[f64],
    tool_calls: Option<&[f64]>,
    completed: Option<&[bool]>,
) -> Result<Steps> {
    floats(steps, "steps", Some(0.0), None)?;
    let tool_calls = match tool_calls {
        Some(t) => {
            floats(t, "tool_calls", Some(0.0), None)?;
            same(steps.len(), t.len(), "steps", "tool_calls")?;
            Some(Summary::of(t))
        }
        None => None,
    };
    let mean_steps_completed = match completed {
        Some(c) => {
            same(steps.len(), c.len(), "steps", "completed")?;
            let s: Vec<f64> = steps
                .iter()
                .zip(c)
                .filter(|(_, &k)| k)
                .map(|(&s, _)| s)
                .collect();
            Some(if s.is_empty() { f64::NAN } else { mean(&s) })
        }
        None => None,
    };
    Ok(Steps {
        value: mean(steps),
        steps: Summary::of(steps),
        tool_calls,
        mean_steps_completed,
    })
}

fn lcs<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    for x in a {
        let mut cur = vec![0usize; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = if x == y {
                prev[j] + 1
            } else {
                prev[j + 1].max(cur[j])
            };
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Plan adherence: LCS-based F1 of executed vs planned steps, with its precision, recall and exact-match rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanAdherence {
    pub value: f64,
    pub precision: f64,
    pub recall: f64,
    pub exact_match: f64,
}

/// Order-aware agreement of the executed steps with the plan (longest common subsequence).
pub fn plan_adherence<T: PartialEq>(
    plans: &[Vec<T>],
    executed: &[Vec<T>],
) -> Result<PlanAdherence> {
    nonempty(plans, "plans")?;
    nonempty(executed, "executed")?;
    same(plans.len(), executed.len(), "plans", "executed")?;
    let (mut f1, mut pr, mut rc, mut ex) = (vec![], vec![], vec![], vec![]);
    for (p, e) in plans.iter().zip(executed) {
        let l = lcs(p, e) as f64;
        let a = if e.is_empty() {
            f64::from(u8::from(p.is_empty()))
        } else {
            l / e.len() as f64
        };
        let b = if p.is_empty() {
            1.0
        } else {
            l / p.len() as f64
        };
        pr.push(a);
        rc.push(b);
        f1.push(if a + b == 0.0 {
            0.0
        } else {
            2.0 * a * b / (a + b)
        });
        ex.push(p == e);
    }
    Ok(PlanAdherence {
        value: mean(&f1),
        precision: mean(&pr),
        recall: mean(&rc),
        exact_match: bool_mean(&ex),
    })
}

/// Joint goal accuracy (whole state correct) and slot accuracy over the union of slots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateTracking {
    pub value: f64,
    pub slot_accuracy: f64,
    pub n_turns: usize,
}

/// Dialogue-state tracking accuracy; each state maps slot → value.
pub fn state_tracking_accuracy(
    expected: &[Map<String, Value>],
    predicted: &[Map<String, Value>],
) -> Result<StateTracking> {
    nonempty(expected, "expected_states")?;
    nonempty(predicted, "predicted_states")?;
    same(
        expected.len(),
        predicted.len(),
        "expected_states",
        "predicted_states",
    )?;
    let (mut joint, mut ok, mut n) = (Vec::new(), 0usize, 0usize);
    for (a, b) in expected.iter().zip(predicted) {
        joint.push(
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| py_equal(v, w))),
        );
        let keys: HashSet<&String> = a.keys().chain(b.keys()).collect();
        n += keys.len();
        ok += keys
            .iter()
            .filter(|k| matches!((a.get(**k), b.get(**k)), (Some(x), Some(y)) if py_equal(x, y)))
            .count();
    }
    Ok(StateTracking {
        value: bool_mean(&joint),
        slot_accuracy: if n > 0 { ok as f64 / n as f64 } else { 1.0 },
        n_turns: expected.len(),
    })
}

/// Share of failed calls (`ok == Some(false)`) followed later by a successful call to the same tool.
pub fn tool_failure_recovery_rate(trajectories_: &[Vec<ToolCall>]) -> Result<Rate> {
    trajectories(trajectories_)?;
    let mut flags = Vec::new();
    for t in trajectories_ {
        for (i, c) in t.iter().enumerate() {
            if c.ok == Some(false) {
                flags.push(
                    t[i + 1..]
                        .iter()
                        .any(|d| d.name == c.name && d.ok == Some(true)),
                );
            }
        }
    }
    if flags.is_empty() {
        return invalid("No tool call failed (no 'ok': False): the recovery rate is undefined.");
    }
    Ok(rate(&flags))
}

/// Repeated-call rate and the share of trajectories with a loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Unnecessary {
    pub rate: Rate,
    /// Share of trajectories repeating one identical call at least `loop_length` times in a row.
    pub loop_rate: f64,
}

/// Share of calls repeating an earlier identical call (same name and arguments) of the same trajectory.
pub fn unnecessary_tool_call_rate(
    trajectories_: &[Vec<ToolCall>],
    loop_length: usize,
) -> Result<Unnecessary> {
    if loop_length < 2 {
        return invalid("loop_length must be an integer >= 2.");
    }
    trajectories(trajectories_)?;
    let (mut repeats, mut loops) = (Vec::new(), Vec::new());
    for t in trajectories_ {
        let mut seen = HashSet::new();
        let (mut run, mut longest, mut prev) = (0usize, 0usize, None::<String>);
        for c in t {
            let k = c.key();
            repeats.push(seen.contains(&k));
            run = if prev.as_ref() == Some(&k) {
                run + 1
            } else {
                1
            };
            longest = longest.max(run);
            seen.insert(k.clone());
            prev = Some(k);
        }
        loops.push(longest >= loop_length);
    }
    if repeats.is_empty() {
        return invalid("The trajectories contain no tool calls.");
    }
    Ok(Unnecessary {
        rate: rate(&repeats),
        loop_rate: bool_mean(&loops),
    })
}

/// Share of tasks needing at least one human intervention, and the mean count per task.
pub fn human_intervention_rate(interventions: &[f64]) -> Result<(Rate, f64)> {
    floats(interventions, "interventions", Some(0.0), None)?;
    let flags: Vec<bool> = interventions.iter().map(|&x| x > 0.0).collect();
    Ok((rate(&flags), mean(interventions)))
}

/// Mean cost per task, with totals and the cost per successful task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentCost {
    pub value: f64,
    pub total_cost: f64,
    pub time: Option<Summary>,
    /// Total cost / completed tasks (∞ when none completed), when `completed` is given.
    pub cost_per_success: Option<f64>,
    pub success_rate: Option<f64>,
}

/// Cost per task (any unit), optionally with durations and completion.
pub fn agent_cost_per_task(
    costs: &[f64],
    durations: Option<&[f64]>,
    completed: Option<&[bool]>,
) -> Result<AgentCost> {
    floats(costs, "costs", Some(0.0), None)?;
    let total = crate::util::np_sum(costs);
    let time = match durations {
        Some(d) => {
            floats(d, "durations", Some(0.0), None)?;
            same(costs.len(), d.len(), "costs", "durations")?;
            Some(Summary::of(d))
        }
        None => None,
    };
    let (cost_per_success, success_rate) = match completed {
        Some(c) => {
            same(costs.len(), c.len(), "costs", "completed")?;
            let k = c.iter().filter(|&&x| x).count();
            (
                Some(if k > 0 {
                    total / k as f64
                } else {
                    f64::INFINITY
                }),
                Some(bool_mean(c)),
            )
        }
        None => (None, None),
    };
    Ok(AgentCost {
        value: mean(costs),
        total_cost: total,
        time,
        cost_per_success,
        success_rate,
    })
}
