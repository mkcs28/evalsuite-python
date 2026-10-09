"""Instruction following and structured output: JSON validity and JSON Schema compliance, XML validity,
required-field accuracy, tool-call selection and arguments, API-call success, format compliance, constraint
satisfaction, multi-turn instruction retention and unwanted extra content.

JSON Schema validation is built in (the common Draft 2020-12 keywords listed in ``SUPPORTED_KEYWORDS``), so no
extra dependency is needed; unknown keywords raise an error instead of being ignored silently.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import per_item

__all__ = [
    "SUPPORTED_KEYWORDS",
    "api_call_success_rate",
    "constraint_satisfaction_rate",
    "extract_json",
    "extra_content_rate",
    "format_compliance",
    "instruction_compliance_rate",
    "instruction_retention",
    "json_schema_compliance",
    "json_validity",
    "required_field_accuracy",
    "tool_argument_accuracy",
    "tool_call_f1",
    "tool_selection_accuracy",
    "validate_json_schema",
    "xml_validity",
]

_C = "text"
_REF_IFEVAL = (
    "Zhou J, Lu T, Mishra S, et al. Instruction-following evaluation for large language models (IFEval). "
    "arXiv:2311.07911. 2023."
)
_REF_BFCL = (
    "Patil SG, Mao H, Yan F, et al. The Berkeley Function Calling Leaderboard (BFCL): from tool use to agentic "
    "evaluation of large language models. ICML. 2025."
)
_REF_JSONSCHEMA = "JSON Schema: a media type for describing JSON documents. Draft 2020-12. json-schema.org."
_REF_XML = "Extensible Markup Language (XML) 1.0, 5th ed. W3C Recommendation. 2008."
_REF_MULTI = (
    "Kwan WC, Zeng X, Jiang Y, et al. MT-Eval: a multi-turn capabilities evaluation benchmark for large "
    "language models. EMNLP. 2024."
)


def _outputs(outputs: Any) -> list[str]:
    if isinstance(outputs, (str, bytes)) or outputs is None:
        raise InputValidationError("outputs must be a list of strings (one model output per example).")
    items = list(outputs)
    if not items:
        raise InputValidationError("outputs is empty.")
    for i, o in enumerate(items):
        if not isinstance(o, str):
            raise InputValidationError(f"outputs[{i}] must be a string; got {type(o).__name__}.")
    return items


def _rate(
    metric: str,
    name: str,
    ok: Sequence[bool],
    extra: Optional[dict[str, Any]] = None,
    average: Optional[str] = "mean",
) -> MetricResult:
    arr = np.asarray(ok, dtype=float)
    if average not in ("mean", None):
        raise InputValidationError("average must be 'mean' or None (per example).")
    value: Any = arr if average is None else float(arr.mean())
    return MetricResult(metric, name, value, {"n": int(arr.size), **(extra or {})})


_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.S)


def extract_json(text: str, *, mode: str = "strict") -> Any:
    """Parse JSON from a model output. ``mode="strict"``: the whole output (surrounding whitespace allowed);
    ``"fenced"``: the first ```json code block if present, else the whole output; ``"lenient"``: also the
    first balanced ``{...}`` or ``[...]`` span. Raises ``ValueError`` when nothing parses."""
    if mode not in ("strict", "fenced", "lenient"):
        raise InputValidationError("mode must be 'strict', 'fenced' or 'lenient'.")
    candidates = [text.strip()]
    if mode in ("fenced", "lenient"):
        m = _FENCE.search(text)
        if m:
            candidates.insert(0, m.group(1).strip())
    if mode == "lenient":
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if 0 <= start < end:
                candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            return json.loads(c)
        except (ValueError, RecursionError):
            continue
    raise ValueError("no valid JSON found")


def _try_json(text: str, mode: str) -> tuple[bool, Any]:
    try:
        return True, extract_json(text, mode=mode)
    except ValueError:
        return False, None


@register(
    category=_C,
    task="structured-output",
    name="JSON validity",
    definition="Share of outputs that parse as JSON (the whole output, or the fenced / embedded JSON in lenient "
    "modes).",
    formula="parseable outputs / outputs",
    range="[0, 1]",
    input_requirements=("outputs",),
    references=("Bray T. The JavaScript Object Notation (JSON) data interchange format. RFC 8259. 2017.",),
)
@per_item
def json_validity(outputs: Any, *, mode: str = "strict", average: Optional[str] = "mean") -> MetricResult:
    """JSON validity. ``mode``: ``"strict"`` (default), ``"fenced"`` or ``"lenient"`` (see ``extract_json``)."""
    outs = _outputs(outputs)
    return _rate("json_validity", "JSON validity", [_try_json(o, mode)[0] for o in outs], {"mode": mode}, average)


# ---------------------------------------------------------------- JSON Schema (built-in validator)
SUPPORTED_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "patternProperties",
        "minProperties",
        "maxProperties",
        "propertyNames",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "uniqueItems",
        "contains",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "format",
        "$defs",
        "definitions",
        "$ref",
    }
)


def _json_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "integer" if v.is_integer() else "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _resolve(ref: str, root: Mapping[str, Any]) -> Any:
    if not ref.startswith("#"):
        raise InputValidationError(f"Only local $ref (starting with '#') is supported; got {ref!r}.")
    node: Any = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            raise InputValidationError(f"$ref {ref!r} cannot be resolved.")
        node = node[part]
    return node


def _errors(inst: Any, schema: Any, root: Mapping[str, Any], path: str, depth: int = 0) -> list[str]:
    if depth > 100:
        raise InputValidationError("Schema recursion is too deep (cyclic $ref?).")
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    if not isinstance(schema, Mapping):
        raise InputValidationError(f"Invalid schema at {path}: expected an object or boolean.")
    unknown = set(schema) - SUPPORTED_KEYWORDS
    if unknown:
        raise InputValidationError(f"Unsupported JSON Schema keyword(s) {sorted(unknown)} at {path}.")
    errs: list[str] = []
    if "$ref" in schema:
        errs += _errors(inst, _resolve(schema["$ref"], root), root, path, depth + 1)
    t = _json_type(inst)
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(a == t or (a == "number" and t == "integer") for a in allowed):
            errs.append(f"{path}: expected {'/'.join(allowed)}, got {t}")
            return errs
    if "enum" in schema and not any(_json_equal(inst, e) for e in schema["enum"]):
        errs.append(f"{path}: {inst!r} is not one of the allowed values")
    if "const" in schema and not _json_equal(inst, schema["const"]):
        errs.append(f"{path}: must equal {schema['const']!r}")
    if t in ("integer", "number"):
        x = inst
        if "minimum" in schema and x < schema["minimum"]:
            errs.append(f"{path}: {x} < minimum {schema['minimum']}")
        if "maximum" in schema and x > schema["maximum"]:
            errs.append(f"{path}: {x} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and x <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {x} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and x >= schema["exclusiveMaximum"]:
            errs.append(f"{path}: {x} >= exclusiveMaximum {schema['exclusiveMaximum']}")
        if "multipleOf" in schema:
            q = x / schema["multipleOf"]
            if not math.isclose(q, round(q), rel_tol=0, abs_tol=1e-9):
                errs.append(f"{path}: not a multiple of {schema['multipleOf']}")
    if t == "string":
        n = len(inst)
        if "minLength" in schema and n < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and n > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            errs.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if t == "array":
        if "minItems" in schema and len(inst) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(inst) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            for i in range(len(inst)):
                if any(_json_equal(inst[i], inst[j]) for j in range(i)):
                    errs.append(f"{path}: items are not unique")
                    break
        prefix = schema.get("prefixItems", [])
        for i, (item, sub) in enumerate(zip(inst, prefix)):
            errs += _errors(item, sub, root, f"{path}[{i}]", depth + 1)
        if "items" in schema:
            for i in range(len(prefix), len(inst)):
                errs += _errors(inst[i], schema["items"], root, f"{path}[{i}]", depth + 1)
        if "contains" in schema and not any(
            not _errors(x, schema["contains"], root, path, depth + 1) for x in inst
        ):
            errs.append(f"{path}: no item matches 'contains'")
    if t == "object":
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in inst:
                errs.append(f"{path}: missing required property {k!r}")
        if "minProperties" in schema and len(inst) < schema["minProperties"]:
            errs.append(f"{path}: fewer than {schema['minProperties']} properties")
        if "maxProperties" in schema and len(inst) > schema["maxProperties"]:
            errs.append(f"{path}: more than {schema['maxProperties']} properties")
        pattern_props = schema.get("patternProperties", {})
        for k, v in inst.items():
            matched = False
            if k in props:
                matched = True
                errs += _errors(v, props[k], root, f"{path}.{k}", depth + 1)
            for pat, sub in pattern_props.items():
                if re.search(pat, k):
                    matched = True
                    errs += _errors(v, sub, root, f"{path}.{k}", depth + 1)
            if not matched and "additionalProperties" in schema:
                errs += _errors(v, schema["additionalProperties"], root, f"{path}.{k}", depth + 1)
            if "propertyNames" in schema:
                errs += _errors(k, schema["propertyNames"], root, f"{path} (name {k!r})", depth + 1)
    if "allOf" in schema:
        for sub in schema["allOf"]:
            errs += _errors(inst, sub, root, path, depth + 1)
    if "anyOf" in schema and all(_errors(inst, sub, root, path, depth + 1) for sub in schema["anyOf"]):
        errs.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        n_ok = sum(not _errors(inst, sub, root, path, depth + 1) for sub in schema["oneOf"])
        if n_ok != 1:
            errs.append(f"{path}: matches {n_ok} of oneOf (exactly one required)")
    if "not" in schema and not _errors(inst, schema["not"], root, path, depth + 1):
        errs.append(f"{path}: must not match 'not'")
    return errs


def validate_json_schema(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Return the list of validation errors of ``instance`` against ``schema`` (empty if valid). ``format`` is
    annotation-only, as in Draft 2020-12 by default."""
    return _errors(instance, schema, schema, "$")


@register(
    category=_C,
    task="structured-output",
    name="JSON Schema compliance",
    definition="Share of outputs that parse as JSON and validate against a JSON Schema (types, required and "
    "additional properties, enums, ranges, patterns, arrays, combinators, local $ref).",
    formula="outputs valid against the schema / outputs",
    range="[0, 1]",
    input_requirements=("outputs", "schema"),
    references=(_REF_JSONSCHEMA,),
)
@per_item
def json_schema_compliance(
    outputs: Any, schema: Mapping[str, Any], *, mode: str = "strict", average: Optional[str] = "mean"
) -> MetricResult:
    """JSON Schema compliance; unparseable outputs count as non-compliant. ``params["errors"]`` keeps the first
    error of each failing output, for error analysis."""
    if not isinstance(schema, Mapping):
        raise InputValidationError("schema must be a JSON Schema object (a dict).")
    outs = _outputs(outputs)
    ok, errors = [], {}
    for i, o in enumerate(outs):
        parsed, value = _try_json(o, mode)
        if not parsed:
            ok.append(False)
            errors[i] = "not valid JSON"
            continue
        e = validate_json_schema(value, schema)
        ok.append(not e)
        if e:
            errors[i] = e[0]
    return _rate("json_schema_compliance", "JSON Schema compliance", ok, {"mode": mode, "errors": errors}, average)


@register(
    category=_C,
    task="structured-output",
    name="XML validity",
    definition="Share of outputs that are well-formed XML (one root element); document type declarations are "
    "rejected so that untrusted output cannot trigger entity expansion.",
    formula="well-formed outputs / outputs",
    range="[0, 1]",
    input_requirements=("outputs",),
    references=(_REF_XML,),
)
@per_item
def xml_validity(outputs: Any, *, root_tag: Optional[str] = None, average: Optional[str] = "mean") -> MetricResult:
    """Well-formedness (and, with ``root_tag``, the expected root element)."""
    outs = _outputs(outputs)
    ok = []
    for o in outs:
        text = o.strip()
        if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
            ok.append(False)
            continue
        try:
            root = ET.fromstring(text)  # noqa: S314  (DOCTYPE/ENTITY rejected above)
        except ET.ParseError:
            ok.append(False)
            continue
        ok.append(root_tag is None or root.tag == root_tag)
    return _rate("xml_validity", "XML validity", ok, {"root_tag": root_tag}, average)


def _get(obj: Any, path: str) -> tuple[bool, Any]:
    node = obj
    for part in path.split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return False, None
    return True, node


@register(
    category=_C,
    task="structured-output",
    name="Required-field accuracy",
    definition="Share of expected fields (dotted paths) present in the parsed output with the expected value, "
    "pooled over all examples; a missing field or unparseable output counts as wrong.",
    formula="correct fields / expected fields",
    range="[0, 1]",
    input_requirements=("expected", "outputs"),
    references=(_REF_BFCL,),
)
@per_item
def required_field_accuracy(
    expected: Any, outputs: Any, *, mode: str = "strict", normalize: Any = None, average: Optional[str] = "micro"
) -> MetricResult:
    """``expected`` holds one dict per example mapping dotted field paths (``"address.city"``) to the expected
    value; ``outputs`` the model outputs (JSON strings or already parsed objects). ``normalize`` optionally maps
    values before comparison (e.g. ``str.lower`` for strings)."""
    exp = list(expected)
    outs = list(outputs)
    if not exp or len(exp) != len(outs):
        raise InputValidationError("expected and outputs must be non-empty and the same length.")
    hits, totals = [], []
    for i, (e, o) in enumerate(zip(exp, outs)):
        if not isinstance(e, Mapping) or not e:
            raise InputValidationError(f"expected[{i}] must be a non-empty dict of field path -> value.")
        if isinstance(o, str):
            parsed, obj = _try_json(o, mode)
        else:
            parsed, obj = True, o
        n_ok = 0
        if parsed:
            for path, want in e.items():
                found, got = _get(obj, path)
                if found:
                    a, b = (normalize(got), normalize(want)) if callable(normalize) else (got, want)
                    n_ok += _json_equal(a, b)
        hits.append(n_ok)
        totals.append(len(e))
    h, t = np.asarray(hits, float), np.asarray(totals, float)
    if average == "micro":
        value: Any = float(np.sum(h)) / float(np.sum(t))
    elif average in ("macro", None):
        per = h / t
        value = per if average is None else float(per.mean())
    else:
        raise InputValidationError("average must be 'micro', 'macro' or None.")
    return MetricResult(
        "required_field_accuracy", "Required-field accuracy", value, {"average": average, "n_fields": int(t.sum())}
    )


# ---------------------------------------------------------------- tool calls
def _calls(x: Any, name: str) -> list[list[dict[str, Any]]]:
    items = list(x)
    out = []
    for i, ex in enumerate(items):
        calls = [ex] if isinstance(ex, Mapping) else list(ex or [])
        norm = []
        for j, c in enumerate(calls):
            if not isinstance(c, Mapping) or "name" not in c:
                raise InputValidationError(
                    f"{name}[{i}][{j}] must be a dict with 'name' and optional 'arguments'."
                )
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {"__unparseable__": args}
            norm.append({"name": c["name"], "arguments": dict(args) if isinstance(args, Mapping) else args})
        out.append(norm)
    return out


def _pair_calls(exp: list[list[dict[str, Any]]], pred: list[list[dict[str, Any]]]) -> None:
    if not exp or len(exp) != len(pred):
        raise InputValidationError("expected_calls and predicted_calls must be non-empty and the same length.")


@register(
    category=_C,
    task="tool-use",
    name="Tool-selection accuracy",
    definition="Share of examples whose predicted tool calls name exactly the expected tools (as a multiset; "
    "order ignored unless ordered=True).",
    formula="mean[tools(predicted) = tools(expected)]",
    range="[0, 1]",
    input_requirements=("expected_calls", "predicted_calls"),
    references=(_REF_BFCL,),
)
@per_item
def tool_selection_accuracy(
    expected_calls: Any, predicted_calls: Any, *, ordered: bool = False, average: Optional[str] = "mean"
) -> MetricResult:
    """Each example is a call ``{"name": ..., "arguments": {...}}`` or a list of calls (``[]`` = no call
    expected)."""
    exp, pred = _calls(expected_calls, "expected_calls"), _calls(predicted_calls, "predicted_calls")
    _pair_calls(exp, pred)
    ok = []
    for e, p in zip(exp, pred):
        en, pn = [c["name"] for c in e], [c["name"] for c in p]
        ok.append(en == pn if ordered else sorted(map(str, en)) == sorted(map(str, pn)))
    return _rate("tool_selection_accuracy", "Tool-selection accuracy", ok, {"ordered": ordered}, average)


@register(
    category=_C,
    task="tool-use",
    name="Tool-argument accuracy",
    definition="Share of expected tool calls reproduced with the right tool and exactly the expected arguments "
    "(JSON equality after optional normalisation); calls are matched by tool name in order.",
    formula="expected calls matched with equal arguments / expected calls",
    range="[0, 1]",
    input_requirements=("expected_calls", "predicted_calls"),
    references=(_REF_BFCL,),
)
@per_item
def tool_argument_accuracy(expected_calls: Any, predicted_calls: Any, *, normalize: Any = None) -> MetricResult:
    """Pooled over all expected calls. ``normalize`` maps argument dicts before comparison (for example to
    lowercase strings or round numbers)."""
    exp, pred = _calls(expected_calls, "expected_calls"), _calls(predicted_calls, "predicted_calls")
    _pair_calls(exp, pred)
    hits = total = 0
    for e, p in zip(exp, pred):
        pool = list(p)
        for call in e:
            total += 1
            for k, cand in enumerate(pool):
                if cand["name"] == call["name"]:
                    a, b = cand["arguments"], call["arguments"]
                    if callable(normalize):
                        a, b = normalize(a), normalize(b)
                    if _json_equal(a, b):
                        hits += 1
                        pool.pop(k)
                        break
    if total == 0:
        raise InputValidationError("No expected tool calls to score.")
    return MetricResult(
        "tool_argument_accuracy", "Tool-argument accuracy", hits / total, {"n_expected_calls": total}
    )


@register(
    category=_C,
    task="tool-use",
    name="Tool-call precision, recall and F1",
    definition="Matching of predicted to expected calls (same tool, and same arguments unless "
    "match='name'), pooled over examples: precision over predicted calls, recall over expected calls.",
    formula="F1 = 2PR/(P+R), P = matched/predicted, R = matched/expected",
    range="[0, 1]",
    input_requirements=("expected_calls", "predicted_calls"),
    references=(_REF_BFCL,),
)
@per_item
def tool_call_f1(expected_calls: Any, predicted_calls: Any, *, match: str = "arguments") -> MetricResult:
    """Tool-call F1 with precision and recall in ``params``; ``params["invalid_call_rate"]`` counts predicted
    calls whose arguments were not valid JSON."""
    if match not in ("arguments", "name"):
        raise InputValidationError("match must be 'arguments' or 'name'.")
    exp, pred = _calls(expected_calls, "expected_calls"), _calls(predicted_calls, "predicted_calls")
    _pair_calls(exp, pred)
    matched = n_pred = n_exp = invalid = 0
    for e, p in zip(exp, pred):
        n_exp += len(e)
        n_pred += len(p)
        invalid += sum("__unparseable__" in c["arguments"] for c in p if isinstance(c["arguments"], dict))
        pool = list(p)
        for call in e:
            for k, cand in enumerate(pool):
                if cand["name"] == call["name"] and (
                    match == "name" or _json_equal(cand["arguments"], call["arguments"])
                ):
                    matched += 1
                    pool.pop(k)
                    break
    precision = matched / n_pred if n_pred else (1.0 if n_exp == 0 else 0.0)
    recall = matched / n_exp if n_exp else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return MetricResult(
        "tool_call_f1",
        "Tool-call F1",
        f1,
        {
            "precision": precision,
            "recall": recall,
            "match": match,
            "invalid_call_rate": invalid / n_pred if n_pred else 0.0,
            "n_expected": n_exp,
            "n_predicted": n_pred,
        },
    )


@register(
    category=_C,
    task="tool-use",
    name="API-call success rate",
    definition="Share of tool or API calls that completed successfully (status flags, HTTP status codes below "
    "400, or exceptions recorded as failures).",
    formula="successful calls / calls",
    range="[0, 1]",
    input_requirements=("results",),
    references=(_REF_BFCL,),
)
@per_item
def api_call_success_rate(results: Any) -> MetricResult:
    """``results``: booleans, HTTP status codes, or ``None`` / exception objects for failures."""
    items = list(results)
    if not items:
        raise InputValidationError("results is empty.")
    ok = []
    for r in items:
        if isinstance(r, (bool, np.bool_)):
            ok.append(bool(r))
        elif isinstance(r, (int, np.integer)):
            ok.append(100 <= int(r) < 400)
        elif r is None or isinstance(r, BaseException):
            ok.append(False)
        else:
            raise InputValidationError(
                f"Unrecognised call result {r!r}: use bool, an HTTP status, None or an exception."
            )
    return _rate("api_call_success_rate", "API-call success rate", ok)


# ---------------------------------------------------------------- instructions and format
def _check_matrix(checks: Any, name: str) -> list[list[bool]]:
    rows = list(checks)
    if not rows:
        raise InputValidationError(f"{name} is empty.")
    out = []
    for i, r in enumerate(rows):
        vals = [bool(v) for v in (r if isinstance(r, (list, tuple, np.ndarray)) else [r])]
        if not vals:
            raise InputValidationError(f"{name}[{i}] has no checks.")
        out.append(vals)
    return out


@register(
    category=_C,
    task="instruction-following",
    name="Instruction compliance rate",
    definition="Share of outputs that satisfy every instruction checked for them (IFEval prompt-level strict "
    "accuracy).",
    formula="outputs with all checks passed / outputs",
    range="[0, 1]",
    input_requirements=("checks",),
    references=(_REF_IFEVAL,),
)
@per_item
def instruction_compliance_rate(checks: Any) -> MetricResult:
    """``checks`` holds, per output, the list of pass/fail results of its verifiable instructions."""
    m = _check_matrix(checks, "checks")
    return _rate("instruction_compliance_rate", "Instruction compliance rate", [all(r) for r in m])


@register(
    category=_C,
    task="instruction-following",
    name="Constraint satisfaction rate",
    definition="Share of individual constraints satisfied, pooled over all outputs (IFEval instruction-level "
    "accuracy).",
    formula="satisfied constraints / constraints",
    range="[0, 1]",
    input_requirements=("checks",),
    references=(_REF_IFEVAL,),
)
@per_item
def constraint_satisfaction_rate(checks: Any) -> MetricResult:
    """Same input as ``instruction_compliance_rate``; every constraint counts once."""
    m = _check_matrix(checks, "checks")
    flat = [v for r in m for v in r]
    return MetricResult(
        "constraint_satisfaction_rate",
        "Constraint satisfaction rate",
        float(np.mean(flat)),
        {"n_constraints": len(flat), "n_outputs": len(m)},
    )


@register(
    category=_C,
    task="instruction-following",
    name="Format compliance",
    definition="Share of outputs that fully match a required format, given as a regular expression (for "
    "example a refusal template or an answer line such as 'Answer: <letter>').",
    formula="outputs matching the format / outputs",
    range="[0, 1]",
    input_requirements=("outputs", "pattern"),
    references=(_REF_IFEVAL,),
)
@per_item
def format_compliance(
    outputs: Any, pattern: str, *, flags: int = re.S, average: Optional[str] = "mean"
) -> MetricResult:
    """``re.fullmatch(pattern, output.strip())`` per output."""
    outs = _outputs(outputs)
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise InputValidationError(f"Invalid pattern: {exc}") from exc
    return _rate(
        "format_compliance",
        "Format compliance",
        [bool(rx.fullmatch(o.strip())) for o in outs],
        {"pattern": pattern},
        average,
    )


@register(
    category=_C,
    task="instruction-following",
    name="Multi-turn instruction retention",
    definition="Share of later turns in which instructions given earlier in the conversation are still "
    "satisfied, pooled over conversations.",
    formula="later-turn checks passed / later-turn checks",
    range="[0, 1]",
    input_requirements=("turn_checks",),
    references=(_REF_MULTI,),
)
@per_item
def instruction_retention(turn_checks: Any) -> MetricResult:
    """``turn_checks``: per conversation, a list over turns (after the instruction was given) of booleans or
    lists of booleans (one per earlier instruction still in force)."""
    convs = list(turn_checks)
    if not convs:
        raise InputValidationError("turn_checks is empty.")
    flat = [
        bool(v) for conv in convs for turn in conv for v in (turn if isinstance(turn, (list, tuple)) else [turn])
    ]
    if not flat:
        raise InputValidationError("No later-turn checks to score.")
    return MetricResult(
        "instruction_retention",
        "Instruction retention",
        float(np.mean(flat)),
        {"n_checks": len(flat), "n_conversations": len(convs)},
    )


@register(
    category=_C,
    task="structured-output",
    name="Unwanted extra-content rate",
    definition="Share of outputs that contain text beyond the requested payload, e.g. prose or code fences "
    "around JSON that should stand alone.",
    formula="outputs with extra content / outputs",
    range="[0, 1]",
    input_requirements=("outputs",),
    references=(_REF_IFEVAL,),
    higher_is_better=False,
)
@per_item
def extra_content_rate(outputs: Any, *, payload: str = "json", pattern: Optional[str] = None) -> MetricResult:
    """``payload="json"``: extra content means the output is not JSON on its own but JSON can be found inside it
    (fenced or embedded); outputs with no JSON at all are not counted here (see ``json_validity``).
    ``payload="regex"`` with ``pattern``: extra content means the pattern matches a part but not the whole."""
    outs = _outputs(outputs)
    flags = []
    if payload == "json":
        for o in outs:
            strict = _try_json(o, "strict")[0]
            flags.append(not strict and _try_json(o, "lenient")[0])
    elif payload == "regex" and pattern:
        rx = re.compile(pattern, re.S)
        flags = [bool(rx.search(o)) and not rx.fullmatch(o.strip()) for o in outs]
    else:
        raise InputValidationError("payload must be 'json', or 'regex' together with a pattern.")
    return _rate("extra_content_rate", "Extra-content rate", flags, {"payload": payload})
