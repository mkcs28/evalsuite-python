"""2.0.0: conformal prediction, data quality, comparisons, reclassification, auto report, LLM runs, workflow."""

from __future__ import annotations

import contextlib
import json
import sys
import types

import numpy as np
import pytest

import evalsuite as es
from evalsuite.core.exceptions import InputValidationError, OptionalDependencyError


def _formats(r):
    assert isinstance(r.summary(), str)
    assert isinstance(repr(r), str)
    d = r.to_dict()
    json.loads(r.to_json())
    assert isinstance(r.to_markdown(), str)
    assert "<" in r.to_html()
    assert "<" in r._repr_html_()
    return d


# ---------------------------------------------------------------------------------------------- conformal


def _cls(n=3000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k), n)
    y = np.array([rng.choice(k, p=q) for q in p])
    return y, p


@pytest.mark.parametrize("method", ["lac", "aps", "raps"])
def test_conformal_classification_coverage(method):
    y, p = _cls()
    r = es.conformal_classification(y[:1500], p[:1500], p[1500:], alpha=0.1, method=method, y_test=y[1500:])
    hit = np.mean(r.sets[np.arange(1500), y[1500:]])
    assert 0.86 < hit < 0.94
    assert r.sizes.shape == (1500,)
    assert 0 <= r.singleton_rate <= 1 and 0 <= r.empty_rate <= 1 and r.mean_size > 0
    assert isinstance(r.labels(0), list)
    _formats(r)


def test_conformal_classification_groups_and_classes():
    y, p = _cls()
    names = np.array(["a", "b", "c", "d"])
    r = es.conformal_classification(
        names[y[:1500]], p[:1500], p[1500:], classes=names, by="class", y_test=names[y[1500:]]
    )
    assert set(r.labels(0)) <= set(names)
    g = np.arange(3000) % 2
    r2 = es.conformal_classification(y[:1500], p[:1500], p[1500:], by=g[:1500], by_test=g[1500:], y_test=y[1500:])
    _formats(r2)
    with pytest.raises(InputValidationError):
        es.conformal_classification(y[:10], p[:10], p[10:20], method="nope")


@pytest.mark.parametrize("method", ["absolute", "normalized", "cqr"])
def test_conformal_regression(method):
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    s = 0.5 + np.abs(x)
    y = x + rng.normal(size=4000) * s
    kw = {}
    pred = x
    if method == "normalized":
        kw = {"scale_cal": s[:2000], "scale_test": s[2000:]}
    if method == "cqr":
        pred = np.column_stack([x - 1.64 * s, x + 1.64 * s])
    r = es.conformal_regression(y[:2000], pred[:2000], pred[2000:], method=method, y_test=y[2000:], **kw)
    cover = np.mean((y[2000:] >= r.lower) & (y[2000:] <= r.upper))
    assert 0.87 < cover < 0.93
    assert r.widths.shape == (2000,) and r.mean_width > 0
    _formats(r)


def test_conformal_regression_groups_and_errors():
    rng = np.random.default_rng(2)
    y = rng.normal(size=400)
    g = np.arange(400) % 2
    r = es.conformal_regression(y[:200], np.zeros(200), np.zeros(200), by=g[:200], by_test=g[200:], y_test=y[200:])
    _formats(r)
    with pytest.raises(InputValidationError):
        es.conformal_regression(y, y, y, method="bad")
    assert es.conformal_quantile(np.arange(100.0), 0.1) >= 89


# ---------------------------------------------------------------------------------------------- data quality


def test_leakage_report():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    train = pd.DataFrame(
        {
            "id": np.arange(200),
            "x": rng.normal(size=200),
            "z": rng.normal(size=200),
            "text": [f"the quick brown fox number {i} jumps" for i in range(200)],
            "y": rng.integers(0, 2, 200),
        }
    )
    test = train.iloc[:50].copy()
    test.loc[test.index[:10], "x"] += 1e-6
    test["id"] = np.r_[np.arange(5), np.arange(1000, 1045)]
    train["leak"] = train["y"] + rng.normal(0, 1e-3, 200)
    test["leak"] = test["y"]
    r = es.leakage_report(train, test, target="y", id_column="id", text_columns=["text"])
    d = _formats(r)
    assert d
    r2 = es.leakage_report(train[["x", "z"]], test[["x", "z"]])
    _formats(r2)


def test_label_issues_and_cleanlab():
    y, p = _cls(1000, 3, seed=4)
    y = y.copy()
    y[:100] = (y[:100] + 1) % 3
    r = es.label_issues(y, p)
    _formats(r)
    assert 0 <= r.noise_rate <= 1
    assert r.indices.size == r.given.size == r.suggested.size
    cl = pytest.importorskip("cleanlab.filter")
    ref = np.flatnonzero(cl.find_label_issues(y, p, filter_by="confident_learning"))
    assert sorted(r.indices.tolist()) == ref.tolist()
    with pytest.raises(InputValidationError):
        es.label_issues(y[:10], p)
    named = es.label_issues(np.array(list("abc"))[y], p, classes=list("abc"))
    assert set(named.given) <= set("abc")


def test_dataset_card():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.choice(["u", "v"], 100), "y": rng.integers(0, 2, 100)})
    df.loc[3, "a"] = np.nan
    _formats(es.dataset_card(df, target="y", groups=["b"]))
    _formats(es.dataset_card({"a": [1.0, 2.0, 3.0]}))


# ---------------------------------------------------------------------------------------------- comparisons


def test_noninferiority_and_equivalence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    a = np.where(rng.random(600) < 0.85, y, 1 - y)
    b = np.where(rng.random(600) < 0.84, y, 1 - y)
    r = es.noninferiority_test("accuracy", y, a, b, margin=0.05, n_resamples=300, random_state=0)
    assert isinstance(r.p_value, float)
    r2 = es.equivalence_test("accuracy", y, a, b, margin=0.05, n_resamples=300, random_state=0)
    assert isinstance(r2.p_value, float)
    with pytest.raises((InputValidationError, ValueError)):
        es.noninferiority_test("accuracy", y, a, b, margin=-1)


def test_nemenyi_and_plot():
    rng = np.random.default_rng(0)
    scores = rng.random((15, 4)) + np.linspace(0, 0.4, 4)
    r = es.nemenyi_test(scores, models=["a", "b", "c", "d"])
    assert r.p_values.shape == (4, 4)
    _formats(r)
    sp = pytest.importorskip("scikit_posthocs")
    np.testing.assert_allclose(r.p_values, sp.posthoc_nemenyi_friedman(scores).to_numpy(), atol=1e-12)
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from evalsuite import plot

    assert plot.critical_difference(r) is not None


def test_ipw_metric():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.random(500) < 0.8, y, 1 - y)
    obs = rng.random(500) < 0.7
    r = es.ipw_metric("accuracy", y, pred, observed=obs, p_observed=np.full(500, 0.7))
    assert 0 <= float(getattr(r, "value", r)) <= 1


# ---------------------------------------------------------------------------------------------- clinical


def test_reclassification_hand_example():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    old = np.clip(0.3 + 0.2 * y + rng.normal(0, 0.15, 400), 0.01, 0.99)
    new = np.clip(old + 0.1 * (2 * y - 1), 0.01, 0.99)
    r = es.reclassification(y, old, new, cutoffs=[0.2, 0.5], n_resamples=100, random_state=0)
    _formats(r)
    r2 = es.reclassification(y, old, new, n_resamples=100, random_state=0)
    _formats(r2)
    ev = es.external_validation(y[:200], old[:200], y[200:], new[200:])
    _formats(ev)


# ---------------------------------------------------------------------------------------------- auto report


def test_auto_report_and_cli(tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame(
        {
            "label": y,
            "prob": np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 300), 0, 1),
            "group": rng.choice(["a", "b"], 300),
        }
    )
    assert es.detect_columns(list(df.columns))["target"] == "label"
    r = es.auto_report(df, path=str(tmp_path / "r.html"), groups=["group"])
    _formats(r)
    assert (tmp_path / "r.html").exists()
    reg = pd.DataFrame({"y_true": rng.normal(size=100)})
    reg["y_pred"] = reg["y_true"] + rng.normal(0, 0.1, 100)
    _formats(es.auto_report(reg))
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    from evalsuite.cli import main

    main(["report", str(csv)])
    assert (tmp_path / "d.report.html").exists()


# ---------------------------------------------------------------------------------------------- LLM


def test_run_model_callable_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALSUITE_CACHE", str(tmp_path))
    calls = []

    def fn(p):
        calls.append(p)
        return {"output": p.upper(), "input_tokens": 3, "output_tokens": 2}

    fn.cacheable = True
    out = es.run_model(["a", {"prompt": "b", "id": "x"}], fn, concurrency=2)
    assert [o["prediction"] for o in out] == ["A", "B"] and out[1]["id"] == "x"
    again = es.run_model(["a", {"prompt": "b", "id": "x"}], fn)
    assert all(o["cached"] for o in again) and len(calls) == 2
    plain = es.run_model(["q"], lambda p: "r", cache=False)
    assert plain[0]["prediction"] == "r"
    with pytest.raises(InputValidationError):
        es.run_model(["a"], "nope:x")
    with pytest.raises(InputValidationError):
        es.run_model([{"x": 1}], fn)
    with pytest.raises(InputValidationError):
        es.run_model(["a"], 3)


def test_run_model_retries():
    n = {"i": 0}

    def flaky(p):
        n["i"] += 1
        if n["i"] < 2:
            raise RuntimeError("temporary")
        return "ok"

    out = es.run_model(["a"], flaky, cache=False, retries=3, concurrency=1)
    assert out[0]["prediction"] == "ok"


def test_run_model_providers_mocked(monkeypatch, tmp_path):
    monkeypatch.setenv("EVALSUITE_CACHE", str(tmp_path))
    ns = types.SimpleNamespace

    class OpenAI:
        def __init__(self):
            self.chat = ns(
                completions=ns(
                    create=lambda **kw: ns(
                        choices=[ns(message=ns(content="hi"))], usage=ns(prompt_tokens=4, completion_tokens=1)
                    )
                )
            )

    class Anthropic:
        def __init__(self):
            self.messages = ns(
                create=lambda **kw: ns(content=[ns(text="yo")], usage=ns(input_tokens=5, output_tokens=2))
            )

    monkeypatch.setitem(sys.modules, "openai", ns(OpenAI=OpenAI))
    monkeypatch.setitem(sys.modules, "anthropic", ns(Anthropic=Anthropic))
    o = es.run_model(["p"], "openai:gpt-x", system="s", temperature=0)
    assert o[0]["prediction"] == "hi" and o[0]["input_tokens"] == 4
    a = es.run_model(["p"], "anthropic:claude-x", system="s", max_tokens=10)
    assert a[0]["prediction"] == "yo" and a[0]["output_tokens"] == 2
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(OptionalDependencyError):
        es.run_model(["zz"], "openai:other", cache=False)


def test_llm_reports():
    rng = np.random.default_rng(0)
    eps = [
        {
            "success": bool(rng.random() < 0.7),
            "steps": int(rng.integers(1, 8)),
            "cost": 0.01,
            "latency_s": 1.0,
            "invalid_tool_calls": 0,
            "tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
            "expected_tool_calls": [{"name": "search", "arguments": {"q": "x"}}],
        }
        for _ in range(40)
    ]
    _formats(es.agent_report(eps))
    _formats(es.agent_report({"a": eps, "b": eps[:20]}))
    dialogues = [{"turns": list(rng.random(4)), "success": True} for _ in range(20)]
    dialogues.append({"turns": [{"score": 0.5}, {"score": 0.7}]})
    _formats(es.conversation_report(dialogues, n_resamples=100))
    runs = {
        "a": [{"correct": True, "input_tokens": 100, "output_tokens": 10, "latency_s": 1}] * 10,
        "b": [{"score": 0.5, "cost": 0.001, "latency_s": 0.5}] * 10,
    }
    r = es.cost_quality_report(runs, prices={"a": (1.0, 2.0)})
    _formats(r)
    assert r.pareto()
    items = ["the cat sat on the mat and looked at the dog today", {"prompt": "unrelated words here only"}]
    corpus = ["yesterday the cat sat on the mat and looked at the dog today in town"]
    c = es.contamination_check(items, corpus, n=5)
    _formats(c)
    with pytest.raises(InputValidationError):
        es.agent_report([])


# ---------------------------------------------------------------------------------------------- workflow


def test_metrics_dict_history_regression(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = np.where(rng.random(300) < 0.95, y, 1 - y)
    bad = np.where(rng.random(300) < 0.6, y, 1 - y)
    rg = es.evaluate(y, good, metrics=["accuracy"], random_state=0)
    rb = es.evaluate(y, bad, metrics=["accuracy"], random_state=0)
    flat = es.metrics_dict(rg, prefix="t/")
    assert "t/accuracy" in flat
    h = tmp_path / "h.jsonl"
    monkeypatch.setenv("EVALSUITE_HISTORY", str(h))
    es.record_history(rg, model="v1", note="first")
    assert len(es.load_history(model="v1")) == 1 and es.load_history(model="none") == []
    out = es.check_regression(rb, model="v1")
    assert [r["metric"] for r in out["regressed"]] == ["accuracy"]
    up = es.check_regression(rg, baseline={"metrics": {"accuracy": 0.5}})
    assert up["improved"]
    assert es.check_regression(rg, model="zz", dataset="none")["baseline"] is None
    with pytest.raises(InputValidationError):
        es.metrics_dict(object())
    h.write_text("{not json\n")
    with pytest.raises(InputValidationError):
        es.load_history()
    from evalsuite.cli import main

    h.unlink()
    es.record_history(rg, model="v1")
    main(["history", "list"])


def test_mlflow_wandb_mocked(monkeypatch):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    r = es.evaluate(y, y, metrics=["accuracy"], random_state=0)
    logged = {}
    mlflow = types.SimpleNamespace(
        log_metrics=lambda d: logged.setdefault("m", d),
        log_dict=lambda d, n: logged.setdefault("d", n),
        log_text=lambda t, n: logged.setdefault("t", n),
        start_run=lambda run_id=None: object(),
        end_run=lambda: logged.setdefault("end", True),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    es.log_to_mlflow(r, run_id="abc")
    assert logged["m"]["accuracy"] == 1.0 and logged["end"]
    wandb = types.SimpleNamespace(
        run=None, log=lambda d: logged.setdefault("w", d), Table=lambda columns, data: ("table", data)
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    with pytest.raises(InputValidationError):
        es.log_to_wandb(r)
    wandb.run = object()
    es.log_to_wandb(r)
    assert "accuracy" in logged["w"]
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(OptionalDependencyError):
        es.log_to_mlflow(r)


def test_http_handler():
    import io
    from http.server import BaseHTTPRequestHandler

    from evalsuite.workflow import _handler_class

    H = _handler_class(BaseHTTPRequestHandler)

    def call(method, path, body=None):
        h = H.__new__(H)
        raw = json.dumps(body).encode() if body is not None else b""
        h.path = path
        h.headers = {"Content-Length": str(len(raw))}
        h.rfile = io.BytesIO(raw)
        h.wfile = io.BytesIO()
        h.request_version = "HTTP/1.1"
        h.requestline = f"{method} {path}"
        h.command = method
        sent = {}
        h.send_response = lambda c: sent.setdefault("code", c)
        h.send_header = lambda *a: None
        h.end_headers = lambda: None
        getattr(h, f"do_{method}")()
        h.log_message("%s", "x")
        return sent["code"], json.loads(h.wfile.getvalue() or b"{}")

    assert call("GET", "/health")[0] == 200
    assert call("GET", "/metrics")[0] == 200
    assert call("GET", "/x")[0] == 404
    assert call("POST", "/x", {})[0] == 404
    assert call("POST", "/evaluate")[0] == 400
    assert call("POST", "/evaluate", {"y_true": [0, 1, 1, 0], "y_pred": [0, 1, 0, 0], "ci": False})[0] == 200
    assert call("POST", "/evaluate", {"y_true": [0, 1], "bogus": 1})[0] == 400


def test_dataframe_accessor():
    pd = pytest.importorskip("pandas")
    from evalsuite.dataframe import register_accessors

    assert "pandas" in register_accessors()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    df = pd.DataFrame(
        {
            "y": y,
            "a": y,
            "b": 1 - y,
            "p": np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 200), 0.01, 0.99),
            "w": np.ones(200),
        }
    )
    assert df.evalsuite.evaluate("y", "a", y_prob="p", sample_weight="w", metrics=["accuracy"], ci=False)
    assert df.evalsuite.compare("y", {"a": "a", "b": "b"}, n_resamples=100, random_state=0) is not None
    assert df.evalsuite.card(target="y") is not None
    assert df.evalsuite.leakage(df) is not None
    assert df.evalsuite.report() is not None
    with pytest.raises(InputValidationError):
        df.evalsuite.evaluate("missing", "a")
    assert df.evalsuite._cols(["a", "b"]).shape == (200, 2)


# ---------------------------------------------------------------------------------------------- cards / misc


@pytest.mark.parametrize("standard", ["stard", "consort-ai"])
def test_report_card_standards(standard):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 300), 0, 1)
    card = es.report_card(y, p, standard=standard, n_resamples=100, random_state=0)
    _formats(card)


def test_did_you_mean_and_deprecation():
    with pytest.raises(AttributeError, match="Did you mean"):
        es.evaluat  # noqa: B018
    from evalsuite.core.exceptions import EvalSuiteDeprecationWarning, deprecated

    with pytest.warns(EvalSuiteDeprecationWarning, match="use g"):
        deprecated("f", "g", since="2.0")


def test_forest_plot():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from evalsuite import plot

    rows = {"a": (0.8, 0.7, 0.9), "b": (0.6, 0.5, 0.7)}
    assert plot.forest(rows, reference=0.7) is not None
    listed = [{"label": "a", "value": 0.8, "low": 0.7, "high": 0.9}]
    assert plot.forest(listed) is not None


# ---------------------------------------------------------------------------------------------- input checks

_Y, _P = _cls(200, 3, seed=9)


@pytest.mark.parametrize(
    "call",
    [
        lambda: es.conformal_quantile([], 0.1),
        lambda: es.conformal_quantile([1.0, 2.0], 1.5),
        lambda: es.conformal_classification(_Y, _P, _P[:, :2]),
        lambda: es.conformal_classification(_Y[:10], _P, _P),
        lambda: es.conformal_classification(_Y, _P * 2, _P),
        lambda: es.conformal_classification(_Y, _P[:, 0], _P),
        lambda: es.conformal_classification(np.arange(200), _P, _P),
        lambda: es.conformal_classification(_Y, _P, _P, by=_Y),
        lambda: es.conformal_classification(_Y, _P, _P, by=_Y[:5], by_test=_Y),
        lambda: es.conformal_classification(_Y, _P, _P, y_test=_Y[:5]),
        lambda: es.conformal_classification(np.array(list("abc"))[_Y], _P, _P, classes=["a", "b", "z"]),
        lambda: es.conformal_regression([1.0, np.nan], [0.0, 0.0], [0.0]),
        lambda: es.conformal_regression([1.0, 2.0], [0.0], [0.0]),
        lambda: es.conformal_regression([1.0, 2.0], [[0, 1], [0, 1]], [0.0], method="cqr"),
        lambda: es.conformal_regression([1.0, 2.0], [[0, 1]], [[0, 1]], method="cqr"),
        lambda: es.conformal_regression([1.0, 2.0], [[0, np.nan], [0, 1]], [[0, 1]], method="cqr"),
        lambda: es.conformal_regression([1.0, 2.0], [0.0, 0.0], [0.0], method="normalized"),
        lambda: es.conformal_regression(
            [1.0, 2.0], [0.0, 0.0], [0.0], method="normalized", scale_cal=[1.0, -1.0], scale_test=[1.0]
        ),
        lambda: es.conformal_regression([1.0, 2.0], [0.0, 0.0], [0.0], by=[0, 1]),
        lambda: es.conformal_regression([1.0, 2.0], [0.0, 0.0], [0.0], y_test=[1.0, 2.0]),
        lambda: es.nemenyi_test(np.ones(5)),
        lambda: es.nemenyi_test(np.ones((5, 3)), models=["a"]),
        lambda: es.nemenyi_test(np.ones((1, 3))),
        lambda: es.nemenyi_test(np.array([[1.0, np.nan], [1.0, 2.0]])),
        lambda: es.nemenyi_test(np.random.default_rng(0).random((5, 3)), alpha=2),
        lambda: es.noninferiority_test("accuracy", [0, 1], [0, 1], [0, 1], margin=0.1, alpha=0.7),
        lambda: es.ipw_metric("accuracy", [0, 1, 1], [0, 1, 1], observed=[1, 1], p_observed=[0.5, 0.5]),
        lambda: es.ipw_metric("accuracy", [0, 1], [0, 1], observed=[1, 1], p_observed=[0.0, 2.0]),
        lambda: es.ipw_metric("accuracy", [0, 1], [0, 1], observed=[1, 0], p_observed=[0.5, 0.5]),
        lambda: es.ipw_metric("accuracy", [0, 1, 1], None, observed=[1, 1, 1], p_observed=[0.5] * 3),
        lambda: es.ipw_metric("accuracy", [0, 1], [0, 1], observed=[1, 1], p_observed=[0.5, 0.5, 0.5]),
    ],
)
def test_bad_inputs(call):
    with pytest.raises(InputValidationError):
        call()


def test_nemenyi_inputs_and_pairs():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.random((10, 3)), columns=["a", "b", "c"])
    r = es.nemenyi_test(df)
    assert r.models == ("a", "b", "c")
    r2 = es.nemenyi_test({m: df[m].to_numpy() for m in "abc"}, higher_is_better=False)
    assert len(r2.average_ranks) == 3
    two = es.nemenyi_test(rng.random((8, 2)))
    assert two.friedman is not None
    for name in dir(r):
        if (
            not name.startswith("_")
            and callable(getattr(r, name))
            and name not in ("summary", "to_dict", "to_json", "to_markdown", "to_html")
        ):
            with contextlib.suppress(TypeError):
                getattr(r, name)("a", "b")


def test_ipw_with_probabilities():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 300), 0.01, 0.99)
    obs = rng.random(300) < 0.8
    r = es.ipw_metric("roc_auc", y, observed=obs, p_observed=np.full(300, 0.8), y_prob=p)
    assert 0.5 < r.value <= 1


def test_auto_report_inputs(tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    probs = rng.dirichlet(np.ones(3), 300)
    df = pd.DataFrame(
        {
            "target": y,
            "prediction": probs.argmax(1),
            "prob_0": probs[:, 0],
            "prob_1": probs[:, 1],
            "prob_2": probs[:, 2],
            "site": rng.choice(["x", "y"], 300),
            "patient": np.arange(300) // 3,
        }
    )
    found = es.detect_columns(list(df.columns))
    assert found["prob"] == ["prob_0", "prob_1", "prob_2"]
    r = es.auto_report(
        df,
        target="target",
        pred="prediction",
        prob=["prob_0", "prob_1", "prob_2"],
        groups=["site"],
        clusters="patient",
        title="t",
    )
    assert r.task
    for ext, write in (
        (".tsv", lambda p: df.to_csv(p, sep="\t", index=False)),
        (".json", lambda p: df.to_json(p, orient="records")),
        (".jsonl", lambda p: df.to_json(p, orient="records", lines=True)),
    ):
        f = tmp_path / f"d{ext}"
        write(f)
        assert es.auto_report(str(f)).task
    assert es.auto_report({c: df[c].tolist() for c in ("target", "prediction")}).task
    with pytest.raises(InputValidationError):
        es.auto_report(str(tmp_path / "missing.csv"))
    with pytest.raises(InputValidationError):
        es.auto_report(42)
    with pytest.raises(InputValidationError):
        es.auto_report(df, target="nope")
    with pytest.raises(InputValidationError):
        es.auto_report(df, groups=["nope"])
    with pytest.raises(InputValidationError):
        es.auto_report(pd.DataFrame({"target": y, "other": y}))
    with pytest.raises(InputValidationError):
        es.auto_report(pd.DataFrame({"a": y, "b": y}))
    with pytest.raises(InputValidationError):
        es.auto_report(
            pd.DataFrame({"y_true": rng.normal(size=50), "p": rng.random(50)}), prob="p", task="regression"
        )
    pl = pytest.importorskip("polars")
    assert es.auto_report(pl.from_pandas(df[["target", "prediction"]])).task


def test_workflow_edges(tmp_path):
    class Rep:
        metrics = {"auc": {"value": 0.8, "low": 0.7, "high": "x"}, "bad": {"value": None}}  # noqa: RUF012

    class One:
        value = 0.5
        params = {"ci_low": 0.4, "ci_high": 0.6}  # noqa: RUF012

    assert es.metrics_dict(Rep()) == {"auc": 0.8, "auc_ci_low": 0.7}
    assert es.metrics_dict(One()) == {"value": 0.5, "value_ci_low": 0.4, "value_ci_high": 0.6}
    h = tmp_path / "h.jsonl"
    assert es.load_history(path=str(h)) == []
    h.write_text('\n{"model": "m", "dataset": "default", "metrics": {}}\n')
    assert len(es.load_history(path=str(h))) == 1
