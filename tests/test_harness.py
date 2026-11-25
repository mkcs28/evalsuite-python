"""The LLM evaluation harness: scoring, intervals, rank groups, checks and outputs."""

import json
import subprocess
import sys

import numpy as np
import pytest

import evalsuite as es


def runs_with(accs, n=300, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, acc in accs.items():
        out[name] = [
            {
                "id": i,
                "category": ["a", "b"][i % 2],
                "prediction": str(i if rng.random() < acc else -i - 1),
                "reference": str(i),
            }
            for i in range(n)
        ]
    return out


def test_ranks_and_ties():
    runs = runs_with({"strong": 0.9, "weak": 0.5})
    runs["strong2"] = [dict(r) for r in runs["strong"]]  # identical answers: a statistical tie
    lb = es.leaderboard(runs)
    ranks = {r["model"]: r["rank"] for r in lb.rows}
    assert ranks["strong"] == ranks["strong2"] == 1 and ranks["weak"] == 3
    weak = next(r for r in lb.rows if r["model"] == "weak")
    assert weak["ci_low"] < weak["score"] < weak["ci_high"]
    assert {t["test"] for t in lb.tests} <= {"mcnemar", "mcnemar-exact"}
    assert set(lb.categories) == {"a", "b"}


def test_given_correct_and_graded_scores():
    a = [{"id": i, "correct": i % 4 != 0} for i in range(100)]
    b = [{"id": i, "correct": i % 2 == 0} for i in range(100)]
    lb = es.leaderboard({"a": a, "b": b})
    assert lb.rows[0]["model"] == "a" and lb.rows[0]["score"] == 0.75
    g = es.leaderboard(
        {
            "x": [{"id": i, "score": 0.5 + 0.004 * i} for i in range(100)],
            "y": [{"id": i, "score": 0.3} for i in range(100)],
        }
    )
    assert g.settings["binary"] is False and g.tests[0]["test"] == "paired-bootstrap" and g.tests[0]["significant"]


def test_references_scorers_and_checks(tmp_path):
    ref = tmp_path / "ref.jsonl"
    ref.write_text("\n".join(json.dumps({"id": i, "reference": f"#### {i}"}) for i in range(50)))
    run = [
        {
            "id": i,
            "prediction": f"so the answer is {i}" if i % 5 else "no idea",
            "prompt": f"what is {i} plus zero exactly ok",
        }
        for i in range(50)
    ]
    lb = es.leaderboard(
        {"m": run}, scorer="gsm8k", references=str(ref), train_corpus=["what is 3 plus zero exactly ok"], ngram=5
    )
    assert lb.rows[0]["score"] == pytest.approx(0.8)
    assert lb.checks["extraction_failures"]["m"] == 10
    assert lb.checks["contamination"]["contaminated_items"] == 1
    assert (
        es.leaderboard({"m": [{"id": 0, "prediction": "Paris!", "reference": "paris"}]}, scorer="contains").rows[
            0
        ]["score"]
        == 1.0
    )
    custom = es.leaderboard({"m": [{"id": 0, "prediction": "a", "reference": "b"}]}, scorer=lambda p, r: 0.25)
    assert custom.rows[0]["score"] == 0.25


def test_errors():
    with pytest.raises(es.InputValidationError):
        es.leaderboard({"m": [{"prediction": "x"}]})
    with pytest.raises(es.InputValidationError):
        es.leaderboard({"m": [{"id": 1, "correct": True}, {"id": 1, "correct": False}]})
    with pytest.raises(es.InputValidationError):
        es.leaderboard({"m": [{"id": 1, "prediction": "x"}]})
    with pytest.raises(es.InputValidationError):
        es.leaderboard({"m": [{"id": 1, "correct": True}]}, scorer="nope")


def test_outputs_and_cli(tmp_path):
    lb = es.leaderboard(runs_with({"p": 0.8, "q": 0.7}, n=120))
    assert "EvalSuite leaderboard" in lb.summary() and "Paired tests" in lb.summary()
    assert json.loads(lb.to_json())["rows"][0]["rank"] == 1
    assert lb.to_markdown().startswith("| Rank | Model |")
    assert r"\toprule" in lb.to_latex() and lb.to_csv().startswith("rank,model")
    assert "<table>" in lb.to_html()
    for name, recs in runs_with({"p": 0.8, "q": 0.7}, n=60).items():
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    out = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "evalsuite",
            "leaderboard",
            str(tmp_path / "p.jsonl"),
            str(tmp_path / "q.jsonl"),
            "-f",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert {r["model"] for r in json.loads(out.stdout)["rows"]} == {"p", "q"}
