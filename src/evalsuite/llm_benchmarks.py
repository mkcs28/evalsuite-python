"""Ready-made LLM benchmarks: the scorer, references, categories and prompts of common evaluation sets.

>>> import evalsuite as es
>>> items = es.load_benchmark("gsm8k", limit=200)          # needs `pip install datasets`
>>> # ... run your model on item["prompt"], store {"id": item["id"], "prediction": output} per model ...
>>> board = es.leaderboard({"model-a": runs_a, "model-b": runs_b}, benchmark="gsm8k", references=items)

``benchmark=`` sets the scorer (GSM8K's final-number match, letter matching for multiple choice, pass@1 for code)
and, with ``references`` from :func:`load_benchmark`, per-subject categories. Datasets are downloaded from the
Hugging Face Hub by the optional ``datasets`` package and are never bundled. Runs that already contain
``reference`` (or ``correct``) need no download at all.

HumanEval completions must be executed to be scored. EvalSuite never runs model-generated code: run them in your
own sandbox (for example the official ``human-eval`` harness) and pass ``passed`` (true / false) per item, or
``n_samples`` and ``n_correct`` for an unbiased pass@k estimate.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .core.exceptions import InputValidationError, OptionalDependencyError

__all__ = ["BENCHMARKS", "BenchmarkSpec", "format_multiple_choice", "load_benchmark"]


@dataclass(frozen=True)
class BenchmarkSpec:
    """How a benchmark is scored and where its data lives."""

    name: str
    scorer: str
    description: str
    hf_path: str
    hf_config: Optional[str]
    split: str
    convert: Callable[[int, dict[str, Any]], dict[str, Any]]
    multiple_choice: bool = False


def format_multiple_choice(question: str, choices: list[str], *, instruction: Optional[str] = None) -> str:
    """A standard multiple-choice prompt: the question, lettered options and "Answer:"."""
    if len(choices) > 26:
        raise InputValidationError("At most 26 options are supported.")
    lines = [question.strip(), ""]
    lines += [f"{string.ascii_uppercase[i]}. {c}" for i, c in enumerate(choices)]
    head = (instruction or "Answer with the letter of the correct option.").strip()
    return f"{head}\n\n" + "\n".join(lines) + "\nAnswer:"


def _letter(i: int) -> str:
    return string.ascii_uppercase[int(i)]


def _gsm8k(i: int, r: dict[str, Any]) -> dict[str, Any]:
    return {"id": i, "prompt": r["question"], "reference": r["answer"].split("####")[-1].strip()}


def _mmlu(i: int, r: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": i,
        "prompt": format_multiple_choice(r["question"], list(r["choices"])),
        "choices": list(r["choices"]),
        "reference": _letter(r["answer"]),
        "category": r.get("subject"),
    }


def _arc(i: int, r: dict[str, Any]) -> dict[str, Any]:
    labels = list(r["choices"]["label"])
    texts = list(r["choices"]["text"])
    key = str(r["answerKey"])
    gold = labels.index(key) if key in labels else int(key) - 1
    return {
        "id": r.get("id", i),
        "prompt": format_multiple_choice(r["question"], texts),
        "choices": texts,
        "reference": _letter(gold),
    }


def _truthfulqa(i: int, r: dict[str, Any]) -> dict[str, Any]:
    t = r["mc1_targets"]
    choices, labels = list(t["choices"]), list(t["labels"])
    return {
        "id": i,
        "prompt": format_multiple_choice(r["question"], choices),
        "choices": choices,
        "reference": _letter(labels.index(1)),
    }


def _humaneval(i: int, r: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": r.get("task_id", i),
        "prompt": r["prompt"],
        "reference": r.get("canonical_solution"),
        "entry_point": r.get("entry_point"),
        "test": r.get("test"),
    }


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "gsm8k": BenchmarkSpec(
        "gsm8k",
        "gsm8k",
        "Grade-school math word problems; the final number after '####' (Cobbe et al. 2021).",
        "openai/gsm8k",
        "main",
        "test",
        _gsm8k,
    ),
    "mmlu": BenchmarkSpec(
        "mmlu",
        "choice",
        "57-subject multiple choice (Hendrycks et al. 2021); categories are subjects.",
        "cais/mmlu",
        "all",
        "test",
        _mmlu,
        multiple_choice=True,
    ),
    "arc": BenchmarkSpec(
        "arc",
        "choice",
        "ARC-Challenge grade-school science multiple choice (Clark et al. 2018).",
        "allenai/ai2_arc",
        "ARC-Challenge",
        "test",
        _arc,
        multiple_choice=True,
    ),
    "arc-easy": BenchmarkSpec(
        "arc-easy",
        "choice",
        "ARC-Easy grade-school science multiple choice (Clark et al. 2018).",
        "allenai/ai2_arc",
        "ARC-Easy",
        "test",
        _arc,
        multiple_choice=True,
    ),
    "truthfulqa": BenchmarkSpec(
        "truthfulqa",
        "choice",
        "TruthfulQA MC1: one true answer among misconceptions (Lin et al. 2022).",
        "truthfulqa/truthful_qa",
        "multiple_choice",
        "validation",
        _truthfulqa,
        multiple_choice=True,
    ),
    "humaneval": BenchmarkSpec(
        "humaneval",
        "pass",
        "HumanEval Python functions scored by unit tests, pass@1 (Chen et al. 2021); "
        "execute completions in your own sandbox and pass 'passed' or n_samples / n_correct.",
        "openai/openai_humaneval",
        None,
        "test",
        _humaneval,
    ),
}


def spec(name: str) -> BenchmarkSpec:
    key = name.lower().replace("_", "-")
    if key not in BENCHMARKS:
        raise InputValidationError(f"Unknown benchmark {name!r}. Available: {', '.join(BENCHMARKS)}.")
    return BENCHMARKS[key]


def load_benchmark(
    name: str, *, split: Optional[str] = None, limit: Optional[int] = None, cache_dir: Optional[str] = None
) -> list[dict[str, Any]]:
    """Items of a benchmark as records with ``id``, ``prompt``, ``reference`` (and ``choices`` / ``category``
    where they exist), ready to prompt a model and to pass as ``references`` to :func:`evalsuite.leaderboard`.
    Downloads from the Hugging Face Hub with the optional ``datasets`` package
    (``pip install "evalsuite-python[benchmarks]"``) into EvalSuite's cache (``evalsuite cache info``), unless
    ``cache_dir`` is given."""
    s = spec(name)
    try:
        import datasets  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError("datasets", "benchmarks", "load_benchmark") from exc
    if cache_dir is None:
        from .cache import cache_dir as _cache_dir

        cache_dir = str(_cache_dir(create=True) / "datasets")
    ds = datasets.load_dataset(s.hf_path, s.hf_config, split=split or s.split, cache_dir=cache_dir)
    out = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        out.append(s.convert(i, dict(row)))
    return out
