//! EvalSuite's built-in JSON Schema validator (the common Draft 2020-12 keywords in [`SUPPORTED_KEYWORDS`];
//! local `$ref` only; `format` is annotation-only). Unknown keywords are an error, never silently ignored.
//!
//! ```
//! use evalsuite::json_schema::validate;
//! use serde_json::json;
//! let schema = json!({"type": "object", "required": ["id"], "properties": {"id": {"type": "integer"}}});
//! assert!(validate(&json!({"id": 3}), &schema).unwrap().is_empty());
//! assert_eq!(validate(&json!({"id": "x"}), &schema).unwrap(), ["$.id: expected integer, got string"]);
//! ```

use serde_json::{Map, Value};

use crate::error::{invalid, Result};

/// The keywords the validator understands.
pub const SUPPORTED_KEYWORDS: &[&str] = &[
    "$comment", "$defs", "$id", "$ref", "$schema", "additionalProperties", "allOf", "anyOf", "const", "contains",
    "default", "definitions", "deprecated", "description", "enum", "examples", "exclusiveMaximum", "exclusiveMinimum",
    "format", "items", "maxItems", "maxLength", "maxProperties", "maximum", "minItems", "minLength", "minProperties",
    "minimum", "multipleOf", "not", "oneOf", "pattern", "patternProperties", "prefixItems", "properties",
    "propertyNames", "readOnly", "required", "title", "type", "uniqueItems", "writeOnly",
];

/// JSON type name of a value; integral floats count as `"integer"`.
pub fn json_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() || n.as_f64().is_some_and(|f| f.fract() == 0.0) {
                "integer"
            } else {
                "number"
            }
        }
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// JSON equality: numbers compare by value (1 == 1.0), booleans never equal numbers, objects ignore key order.
pub fn json_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => num_eq(x, y),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_equal(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| json_equal(v, w))),
        _ => a == b,
    }
}

/// Python `==` on JSON values: as [`json_equal`] but `true == 1` and `false == 0`.
pub fn py_equal(a: &Value, b: &Value) -> bool {
    let as_num = |v: &Value| match v {
        Value::Bool(t) => Some(f64::from(u8::from(*t))),
        Value::Number(n) => n.as_f64(),
        _ => None,
    };
    match (a, b) {
        (Value::Bool(_), Value::Number(_)) | (Value::Number(_), Value::Bool(_)) => as_num(a) == as_num(b),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_equal(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_equal(v, w))),
        _ => json_equal(a, b),
    }
}

fn num_eq(x: &serde_json::Number, y: &serde_json::Number) -> bool {
    match (x.as_i64(), y.as_i64()) {
        (Some(p), Some(q)) => p == q,
        _ => match (x.as_u64(), y.as_u64()) {
            (Some(p), Some(q)) => p == q,
            _ => x.as_f64() == y.as_f64(),
        },
    }
}

/// A canonical string of a value (object keys sorted, compact), for hashing and set membership. As Python's
/// `json.dumps(sort_keys=True)`, `1` and `1.0` give different strings.
pub fn canonical(v: &Value) -> String {
    let mut s = String::new();
    write_canonical(v, &mut s);
    s
}

fn write_canonical(v: &Value, s: &mut String) {
    match v {
        Value::Array(a) => {
            s.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                write_canonical(x, s);
            }
            s.push(']');
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            s.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&Value::String((*k).clone()).to_string());
                s.push(':');
                write_canonical(&o[k.as_str()], s);
            }
            s.push('}');
        }
        other => s.push_str(&other.to_string()),
    }
}

fn resolve<'a>(r: &str, root: &'a Value) -> Result<&'a Value> {
    let Some(rest) = r.strip_prefix('#') else {
        return invalid(format!("Only local $ref (starting with '#') is supported; got {r:?}."));
    };
    let mut node = root;
    for part in rest.split('/').filter(|p| !p.is_empty()) {
        let part = part.replace("~1", "/").replace("~0", "~");
        match node.as_object().and_then(|o| o.get(&part)) {
            Some(n) => node = n,
            None => return invalid(format!("$ref {r:?} cannot be resolved.")),
        }
    }
    Ok(node)
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn re_search(pat: &str, text: &str) -> Result<bool> {
    match regex::Regex::new(pat) {
        Ok(re) => Ok(re.is_match(text)),
        Err(e) => invalid(format!("Invalid pattern {pat:?}: {e}")),
    }
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(_) => true,
    }
}

fn errors(inst: &Value, schema: &Value, root: &Value, path: &str, depth: usize) -> Result<Vec<String>> {
    if depth > 100 {
        return invalid("Schema recursion is too deep (cyclic $ref?).");
    }
    let s: &Map<String, Value> = match schema {
        Value::Bool(true) => return Ok(vec![]),
        Value::Bool(false) => return Ok(vec![format!("{path}: no value is allowed here")]),
        Value::Object(o) => o,
        _ => return invalid(format!("Invalid schema at {path}: expected an object or boolean.")),
    };
    let mut unknown: Vec<&String> = s.keys().filter(|k| !SUPPORTED_KEYWORDS.contains(&k.as_str())).collect();
    if !unknown.is_empty() {
        unknown.sort();
        return invalid(format!("Unsupported JSON Schema keyword(s) {unknown:?} at {path}."));
    }
    let sub = |i: &Value, sc: &Value, p: &str| errors(i, sc, root, p, depth + 1);
    let mut errs = Vec::new();
    if let Some(r) = s.get("$ref") {
        let r = r.as_str().ok_or_else(|| crate::Error::InvalidInput("$ref must be a string.".into()))?;
        errs.extend(sub(inst, resolve(r, root)?, path)?);
    }
    let t = json_type(inst);
    if let Some(ty) = s.get("type") {
        let allowed: Vec<&str> = match ty {
            Value::Array(a) => a.iter().filter_map(Value::as_str).collect(),
            Value::String(x) => vec![x.as_str()],
            _ => return invalid(format!("Invalid 'type' at {path}.")),
        };
        if !allowed.iter().any(|&a| a == t || (a == "number" && t == "integer")) {
            errs.push(format!("{path}: expected {}, got {t}", allowed.join("/")));
            return Ok(errs);
        }
    }
    if let Some(Value::Array(e)) = s.get("enum") {
        if !e.iter().any(|x| json_equal(inst, x)) {
            errs.push(format!("{path}: {inst} is not one of the allowed values"));
        }
    }
    if let Some(c) = s.get("const") {
        if !json_equal(inst, c) {
            errs.push(format!("{path}: must equal {c}"));
        }
    }
    if t == "integer" || t == "number" {
        let x = f(inst);
        if let Some(m) = s.get("minimum") {
            if x < f(m) {
                errs.push(format!("{path}: {x} < minimum {m}"));
            }
        }
        if let Some(m) = s.get("maximum") {
            if x > f(m) {
                errs.push(format!("{path}: {x} > maximum {m}"));
            }
        }
        if let Some(m) = s.get("exclusiveMinimum") {
            if x <= f(m) {
                errs.push(format!("{path}: {x} <= exclusiveMinimum {m}"));
            }
        }
        if let Some(m) = s.get("exclusiveMaximum") {
            if x >= f(m) {
                errs.push(format!("{path}: {x} >= exclusiveMaximum {m}"));
            }
        }
        if let Some(m) = s.get("multipleOf") {
            let q = x / f(m);
            if !((q - q.round_ties_even()).abs() <= 1e-9) {
                errs.push(format!("{path}: not a multiple of {m}"));
            }
        }
    }
    if let Value::String(text) = inst {
        let n = text.chars().count() as f64;
        if let Some(m) = s.get("minLength") {
            if n < f(m) {
                errs.push(format!("{path}: shorter than {m}"));
            }
        }
        if let Some(m) = s.get("maxLength") {
            if n > f(m) {
                errs.push(format!("{path}: longer than {m}"));
            }
        }
        if let Some(Value::String(p)) = s.get("pattern") {
            if !re_search(p, text)? {
                errs.push(format!("{path}: does not match pattern {p:?}"));
            }
        }
    }
    if let Value::Array(items) = inst {
        let n = items.len() as f64;
        if let Some(m) = s.get("minItems") {
            if n < f(m) {
                errs.push(format!("{path}: fewer than {m} items"));
            }
        }
        if let Some(m) = s.get("maxItems") {
            if n > f(m) {
                errs.push(format!("{path}: more than {m} items"));
            }
        }
        if truthy(s.get("uniqueItems")) && (0..items.len()).any(|i| (0..i).any(|j| json_equal(&items[i], &items[j]))) {
            errs.push(format!("{path}: items are not unique"));
        }
        let empty = Vec::new();
        let prefix = s.get("prefixItems").and_then(Value::as_array).unwrap_or(&empty);
        for (i, (item, sc)) in items.iter().zip(prefix).enumerate() {
            errs.extend(sub(item, sc, &format!("{path}[{i}]"))?);
        }
        if let Some(sc) = s.get("items") {
            for (i, item) in items.iter().enumerate().skip(prefix.len()) {
                errs.extend(sub(item, sc, &format!("{path}[{i}]"))?);
            }
        }
        if let Some(sc) = s.get("contains") {
            let mut any = false;
            for x in items {
                if sub(x, sc, path)?.is_empty() {
                    any = true;
                    break;
                }
            }
            if !any {
                errs.push(format!("{path}: no item matches 'contains'"));
            }
        }
    }
    if let Value::Object(obj) = inst {
        let empty = Map::new();
        let props = s.get("properties").and_then(Value::as_object).unwrap_or(&empty);
        if let Some(Value::Array(req)) = s.get("required") {
            for k in req.iter().filter_map(Value::as_str) {
                if !obj.contains_key(k) {
                    errs.push(format!("{path}: missing required property {k:?}"));
                }
            }
        }
        let n = obj.len() as f64;
        if let Some(m) = s.get("minProperties") {
            if n < f(m) {
                errs.push(format!("{path}: fewer than {m} properties"));
            }
        }
        if let Some(m) = s.get("maxProperties") {
            if n > f(m) {
                errs.push(format!("{path}: more than {m} properties"));
            }
        }
        let pattern_props = s.get("patternProperties").and_then(Value::as_object).unwrap_or(&empty);
        for (k, v) in obj {
            let mut matched = false;
            if let Some(sc) = props.get(k) {
                matched = true;
                errs.extend(sub(v, sc, &format!("{path}.{k}"))?);
            }
            for (pat, sc) in pattern_props {
                if re_search(pat, k)? {
                    matched = true;
                    errs.extend(sub(v, sc, &format!("{path}.{k}"))?);
                }
            }
            if !matched {
                if let Some(sc) = s.get("additionalProperties") {
                    errs.extend(sub(v, sc, &format!("{path}.{k}"))?);
                }
            }
            if let Some(sc) = s.get("propertyNames") {
                errs.extend(sub(&Value::String(k.clone()), sc, &format!("{path} (name {k:?})"))?);
            }
        }
    }
    if let Some(Value::Array(all)) = s.get("allOf") {
        for sc in all {
            errs.extend(sub(inst, sc, path)?);
        }
    }
    if let Some(Value::Array(any)) = s.get("anyOf") {
        let mut ok = false;
        for sc in any {
            if sub(inst, sc, path)?.is_empty() {
                ok = true;
                break;
            }
        }
        if !ok {
            errs.push(format!("{path}: matches none of anyOf"));
        }
    }
    if let Some(Value::Array(one)) = s.get("oneOf") {
        let mut n_ok = 0;
        for sc in one {
            n_ok += usize::from(sub(inst, sc, path)?.is_empty());
        }
        if n_ok != 1 {
            errs.push(format!("{path}: matches {n_ok} of oneOf (exactly one required)"));
        }
    }
    if let Some(sc) = s.get("not") {
        if sub(inst, sc, path)?.is_empty() {
            errs.push(format!("{path}: must not match 'not'"));
        }
    }
    Ok(errs)
}

/// Validation errors of `instance` against `schema` (empty when valid). Errors in the schema itself (unknown
/// keywords, unresolvable `$ref`, bad patterns) are returned as `Err`.
pub fn validate(instance: &Value, schema: &Value) -> Result<Vec<String>> {
    errors(instance, schema, schema, "$", 0)
}
