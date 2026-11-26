//! WebAssembly bindings: `npm install evalsuite`, then
//!
//! ```js
//! import init, { evaluate, metrics } from "evalsuite";
//! await init();
//! evaluate("f1", { y_true: [0, 1, 1, 0], y_pred: [0, 1, 0, 0] }); // { metric: "f1", value: 0.667 }
//! ```
//!
//! The same Rust engine as the Python package and the `evalsuite` crate, through [`evalsuite::bridge`].

use wasm_bindgen::prelude::*;

#[wasm_bindgen(
    inline_js = "export function parseResult(t) { return JSON.parse(t, (k, v) => v === 'NaN' ? NaN : v === 'Infinity' ? Infinity : v === '-Infinity' ? -Infinity : v); }"
)]
extern "C" {
    #[wasm_bindgen(js_name = parseResult, catch)]
    fn parse_result(text: &str) -> Result<JsValue, JsValue>;
}

/// Computes a metric. `args` is a JSON string (the JS wrapper passes `JSON.stringify(args)`); returns the
/// JSON result or throws an `Error` with EvalSuite's message.
#[wasm_bindgen(js_name = evaluateJson)]
pub fn evaluate_json(metric: &str, args: &str) -> Result<String, JsError> {
    evalsuite::bridge::call(metric, args).map_err(|e| JsError::new(&e.to_string()))
}

/// Computes a metric on a plain JS object of arguments and returns the result object
/// (`{ metric, value }` or `{ metric, values }`); throws an `Error` with EvalSuite's message. Undefined or
/// infinite results come back as `NaN` / `Infinity` (in `evaluateJson`'s text, as the strings "NaN",
/// "Infinity" and "-Infinity", since JSON has no such numbers).
#[wasm_bindgen]
pub fn evaluate(metric: &str, args: &JsValue) -> Result<JsValue, JsError> {
    let text: String = js_sys::JSON::stringify(args)
        .map_err(|_| JsError::new("arguments must be JSON-serialisable"))?
        .into();
    let out = evaluate_json(metric, &text)?;
    parse_result(&out).map_err(|_| JsError::new("internal: invalid JSON result"))
}

/// The metric names `evaluateJson` understands.
#[wasm_bindgen]
pub fn metrics() -> Vec<String> {
    evalsuite::bridge::METRICS
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
}

/// The engine version.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}
