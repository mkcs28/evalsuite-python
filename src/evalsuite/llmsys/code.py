"""Code generation and software engineering (v0.5.0).

EvalSuite never executes generated code: test outcomes, coverage, analyser findings and timings come from your
own sandboxed runner, and these functions aggregate them. Three measures are computed from source text directly,
with Python's standard library only: syntax validity (``compile``), CodeBLEU and cyclomatic complexity /
maintainability index (``ast``, the same counting rules as radon). Pass@k is the existing ``es.pass_at_k``.
"""

from __future__ import annotations

import ast
import io
import math
import tokenize
import warnings
from collections import Counter
from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np

from ..core.exceptions import InputValidationError
from ..core.registry import register
from ..core.result import MetricResult
from ._common import bools, floats, lists_of, rate, same_length, seq

__all__ = [
    "PYTHON_KEYWORDS",
    "code_complexity",
    "codebleu",
    "coverage_rate",
    "execution_success_rate",
    "patch_acceptance_rate",
    "resolved_rate",
    "runtime_efficiency",
    "security_vulnerability_rate",
    "static_analysis_violation_rate",
    "syntax_validity_rate",
    "unit_test_pass_rate",
]

_C = "code"
_REF_HUMANEVAL = (
    "Chen M, Tworek J, Jun H, et al. Evaluating large language models trained on code. arXiv:2107.03374. 2021."
)
_REF_CODEBLEU = (
    "Ren S, Guo D, Lu S, et al. CodeBLEU: a method for automatic evaluation of code synthesis. arXiv:2009.10297. "
    "2020."
)
_REF_SWEBENCH = (
    "Jimenez CE, Yang J, Wettig A, et al. SWE-bench: can language models resolve real-world GitHub issues? ICLR. "
    "2024."
)
_REF_MCCABE = "McCabe TJ. A complexity measure. IEEE Trans Softw Eng. 1976;SE-2(4):308-320."
_REF_MI = (
    "Coleman D, Ash D, Lowther B, Oman P. Using metrics to evaluate software system maintainability. Computer. "
    "1994;27(8):44-49."
)
_REF_PEARCE = (
    "Pearce H, Ahmad B, Tan B, Dolan-Gavitt B, Karri R. Asleep at the keyboard? Assessing the security of GitHub "
    "Copilot's code contributions. IEEE S&P. 2022:754-768."
)
_REF_EFFIBENCH = (
    "Huang D, Zhang JM, Qing Y, Cui H. EffiBench: benchmarking the efficiency of automatically generated code. "
    "NeurIPS Datasets and Benchmarks. 2024."
)
_REF_SWT = (
    "Mündler N, Müller MN, He J, Vechev M. SWT-Bench: testing and validating real-world bug-fixes with code "
    "agents. "
    "NeurIPS. 2024."
)
_REF_MBPP = (
    "Austin J, Odena A, Nye M, et al. Program synthesis with large language models. arXiv:2108.07732. 2021."
)
_REF_LINT = (
    "Liu Y, Le-Cong T, Widyasari R, et al. Refining ChatGPT-generated code: characterizing and mitigating code "
    "quality issues. ACM TOSEM. 2024;33(5):1-26."
)
_REF_COV = (
    "Siddiq ML, Santos JCS, Tanvir RH, et al. Using large language models to generate JUnit tests: an empirical "
    "study. EASE. 2024:313-322."
)


def _codes(x: Any, name: str) -> list[str]:
    items = seq(x, name)
    for i, c in enumerate(items):
        if not isinstance(c, str):
            raise InputValidationError(f"{name}[{i}] must be a string of source code.")
    return items


# ---------------------------------------------------------------------------------------------- tests
@register(
    category=_C,
    task="code-generation",
    name="Unit-test pass rate",
    definition="Per problem, the share of its unit tests the generated solution passes, averaged over problems; "
    "the strict rate (all tests pass) is reported alongside.",
    formula="mean_p (passed_p / tests_p); strict = mean_p 1[all pass]",
    range="[0, 1]",
    input_requirements=("test_results",),
    references=(_REF_HUMANEVAL, _REF_MBPP),
)
def unit_test_pass_rate(test_results: Any) -> MetricResult:
    """``test_results``: per problem, one boolean per unit test."""
    rows = lists_of(test_results, "test_results")
    if any(not r for r in rows):
        raise InputValidationError("Every problem needs at least one test result.")
    frac = np.array([np.mean([bool(v) for v in r]) for r in rows])
    strict = np.array([all(bool(v) for v in r) for r in rows])
    return MetricResult(
        "unit_test_pass_rate",
        "Unit-test pass rate",
        float(frac.mean()),
        {"strict_pass_rate": float(strict.mean()), "n_problems": len(rows), "n_tests": sum(len(r) for r in rows)},
    )


@register(
    category=_C,
    task="code-generation",
    name="Compilation / syntax validity rate",
    definition="Share of generated programs that parse (Python: compiled with ``compile`` in 'exec' mode, never "
    "run). Other languages: pass a ``checker`` that returns True when the code compiles.",
    formula="valid programs / programs",
    range="[0, 1]",
    input_requirements=("code",),
    references=(_REF_HUMANEVAL,),
)
def syntax_validity_rate(
    code: Any, *, language: str = "python", checker: Optional[Callable[[str], bool]] = None
) -> MetricResult:
    progs = _codes(code, "code")
    if checker is None:
        if language != "python":
            raise InputValidationError("Only Python is checked built-in; pass checker= for other languages.")

        def checker(src: str) -> bool:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    compile(src, "<generated>", "exec", dont_inherit=True)  # parses only; nothing runs
            except (SyntaxError, ValueError):
                return False
            return True

    flags = np.array([bool(checker(p)) for p in progs])
    return rate("syntax_validity_rate", "Syntax validity rate", flags, {"language": language})


@register(
    category=_C,
    task="code-quality",
    name="Static-analysis violation rate",
    definition="Linter / static-analyser findings in generated code: violations per 1,000 lines (the value) and "
    "the share of programs with at least one finding, from counts your analyser (ruff, pylint, ESLint) reported.",
    formula="1000 · Σ violations / Σ lines",
    range="[0, ∞)",
    input_requirements=("violations", "lines"),
    references=(_REF_LINT,),
    higher_is_better=False,
)
def static_analysis_violation_rate(violations: Any, lines: Any) -> MetricResult:
    v = floats(violations, "violations", lo=0)
    n = floats(lines, "lines", lo=0)
    same_length(("violations", v), ("lines", n))
    if n.sum() == 0:
        raise InputValidationError("lines sums to zero.")
    return MetricResult(
        "static_analysis_violation_rate",
        "Violations per KLOC",
        float(1000 * v.sum() / n.sum()),
        {"programs_with_violations": float((v > 0).mean()), "mean_per_program": float(v.mean())},
    )


@register(
    category=_C,
    task="code-generation",
    name="Code execution success / functional correctness",
    definition="Share of generated programs that run to completion and give the expected result in your sandbox, "
    "with the breakdown of failure kinds (runtime error, timeout, wrong answer, compile error).",
    formula="#passed / #programs",
    range="[0, 1]",
    input_requirements=("outcomes",),
    references=(_REF_HUMANEVAL, _REF_MBPP),
)
def execution_success_rate(outcomes: Any, *, success: Sequence[Any] = ("passed", "ok", True)) -> MetricResult:
    """``outcomes``: one outcome per program, e.g. ``"passed"``, ``"wrong_answer"``, ``"runtime_error"``,
    ``"timeout"``."""
    items = seq(outcomes, "outcomes")
    ok = set(success)
    flags = np.array([o in ok for o in items])
    fails = Counter(str(o) for o in items if o not in ok)
    return rate(
        "execution_success_rate",
        "Execution success rate",
        flags,
        {"by_failure": {k: v / len(items) for k, v in sorted(fails.items())}},
    )


@register(
    category=_C,
    task="testing",
    name="Generated-code test coverage",
    definition="Line (or branch) coverage achieved by generated tests, pooled over programs (covered / "
    "coverable), "
    "with the mean per-program coverage.",
    formula="Σ covered / Σ coverable",
    range="[0, 1]",
    input_requirements=("covered", "total"),
    references=(_REF_COV,),
)
def coverage_rate(covered: Any, total: Any) -> MetricResult:
    c = floats(covered, "covered", lo=0)
    t = floats(total, "total", lo=0)
    same_length(("covered", c), ("total", t))
    if np.any(c > t):
        raise InputValidationError("covered cannot exceed total.")
    if t.sum() == 0:
        raise InputValidationError("total sums to zero.")
    per = np.divide(c, t, out=np.ones_like(c), where=t > 0)
    return MetricResult(
        "coverage_rate", "Test coverage", float(c.sum()) / float(t.sum()), {"mean_per_program": float(per.mean())}
    )


@register(
    category=_C,
    task="software-engineering",
    name="Bug reproduction / patch acceptance rate",
    definition="Share of attempts accepted: generated patches merged or approved by reviewers, or generated tests "
    "that reproduce the reported bug (fail before the fix and pass after it, SWT-Bench).",
    formula="accepted / attempts",
    range="[0, 1]",
    input_requirements=("accepted",),
    references=(_REF_SWT, _REF_SWEBENCH),
)
def patch_acceptance_rate(
    accepted: Any = None, *, fails_before: Any = None, passes_after: Any = None
) -> MetricResult:
    """Either ``accepted`` (one bool per patch), or ``fails_before`` and ``passes_after`` (one bool per generated
    reproduction test) for the bug-reproduction rate."""
    if accepted is not None:
        return rate(
            "patch_acceptance_rate", "Patch acceptance rate", bools(accepted, "accepted"), {"mode": "acceptance"}
        )
    if fails_before is None or passes_after is None:
        raise InputValidationError("Pass accepted=, or both fails_before= and passes_after=.")
    fb, pa = bools(fails_before, "fails_before"), bools(passes_after, "passes_after")
    same_length(("fails_before", fb), ("passes_after", pa))
    return rate("patch_acceptance_rate", "Bug reproduction rate", fb & pa, {"mode": "reproduction"})


@register(
    category=_C,
    task="software-engineering",
    name="Repository task success / SWE-bench resolved rate",
    definition="Share of task instances resolved: every FAIL_TO_PASS test now passes and every PASS_TO_PASS test "
    "still passes after applying the generated patch (SWE-bench).",
    formula="mean_i 1[all F2P pass ∧ all P2P pass]",
    range="[0, 1]",
    input_requirements=("fail_to_pass", "pass_to_pass"),
    references=(_REF_SWEBENCH,),
)
def resolved_rate(fail_to_pass: Any, pass_to_pass: Any, *, applied: Any = None) -> MetricResult:
    """Per instance, the post-patch results of its FAIL_TO_PASS and PASS_TO_PASS tests (lists of bools).
    ``applied``: whether the patch applied at all (unapplied instances are unresolved)."""
    f2p, p2p = lists_of(fail_to_pass, "fail_to_pass"), lists_of(pass_to_pass, "pass_to_pass")
    same_length(("fail_to_pass", f2p), ("pass_to_pass", p2p))
    if any(not f for f in f2p):
        raise InputValidationError("Every instance needs at least one FAIL_TO_PASS test.")
    ok = np.array([all(map(bool, f)) and all(map(bool, p)) for f, p in zip(f2p, p2p)])
    extra: dict[str, Any] = {"f2p_only_rate": float(np.mean([all(map(bool, f)) for f in f2p]))}
    if applied is not None:
        a = bools(applied, "applied")
        same_length(("fail_to_pass", f2p), ("applied", a))
        ok &= a
        extra["apply_rate"] = float(a.mean())
    return rate("resolved_rate", "Resolved rate", ok, extra)


@register(
    category=_C,
    task="code-security",
    name="Security vulnerability rate",
    definition="Share of generated programs with at least one security finding at or above a severity level "
    "(from Bandit, CodeQL, Semgrep), with findings per program by severity.",
    formula="programs with a finding ≥ severity / programs",
    range="[0, 1]",
    input_requirements=("findings",),
    references=(_REF_PEARCE,),
    higher_is_better=False,
)
def security_vulnerability_rate(findings: Any, *, min_severity: str = "low") -> MetricResult:
    """``findings``: per program, the list of finding severities (``"low"``, ``"medium"``, ``"high"``,
    ``"critical"``)."""
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if min_severity not in order:
        raise InputValidationError(f"min_severity must be one of {list(order)}.")
    rows = lists_of(findings, "findings")
    counts: Counter[str] = Counter()
    flags = []
    for i, r in enumerate(rows):
        sev = [str(s).lower() for s in r]
        bad = [s for s in sev if s not in order]
        if bad:
            raise InputValidationError(f"findings[{i}] has unknown severities {bad}.")
        counts.update(sev)
        flags.append(any(order[s] >= order[min_severity] for s in sev))
    n = len(rows)
    return rate(
        "security_vulnerability_rate",
        "Security vulnerability rate",
        np.array(flags),
        {"min_severity": min_severity, "findings_per_program": {k: counts.get(k, 0) / n for k in order}},
    )


@register(
    category=_C,
    task="code-efficiency",
    name="Runtime efficiency / memory consumption",
    definition="Execution time and peak memory of generated solutions relative to reference (canonical) "
    "solutions on the same tests: normalised execution time NET and normalised memory usage NMU (EffiBench); "
    "values above 1 mean the generated code is slower or uses more memory.",
    formula="NET = mean_p t_gen / t_ref; NMU = mean_p m_gen / m_ref",
    range="(0, ∞) (1 = as efficient as the reference)",
    input_requirements=("times", "reference_times"),
    references=(_REF_EFFIBENCH,),
    higher_is_better=False,
)
def runtime_efficiency(
    times: Any, reference_times: Any, *, memory: Any = None, reference_memory: Any = None
) -> MetricResult:
    t = floats(times, "times", lo=0)
    r = floats(reference_times, "reference_times", lo=0)
    same_length(("times", t), ("reference_times", r))
    if np.any(r <= 0):
        raise InputValidationError("reference_times must be positive.")
    ratio = t / r
    extra: dict[str, Any] = {
        "median_time_ratio": float(np.median(ratio)),
        "share_slower_than_2x": float((ratio > 2).mean()),
    }
    if memory is not None or reference_memory is not None:
        m = floats(memory, "memory", lo=0)
        rm = floats(reference_memory, "reference_memory", lo=0)
        same_length(("times", t), ("memory", m), ("reference_memory", rm))
        if np.any(rm <= 0):
            raise InputValidationError("reference_memory must be positive.")
        extra["nmu"] = float(np.mean(m / rm))
    return MetricResult("runtime_efficiency", "Normalised execution time (NET)", float(ratio.mean()), extra)


# ---------------------------------------------------------------------------------------------- complexity
class _Complexity(ast.NodeVisitor):
    """Cyclomatic complexity with radon's counting rules (``radon.visitors.ComplexityVisitor``)."""

    def __init__(self, off: bool = True) -> None:
        self.off = off
        self.complexity = 1 if off else 0
        self.functions: list[tuple[str, int]] = []
        self.classes: list[int] = []

    @classmethod
    def run(cls, node: ast.AST, off: bool = True) -> _Complexity:
        v = cls(off)
        v.visit(node)
        return v

    def generic_visit(self, node: ast.AST) -> None:
        name = type(node).__name__
        if name in ("Try", "TryStar"):
            self.complexity += len(node.handlers) + bool(node.orelse)  # type: ignore[attr-defined]
        elif name == "BoolOp":
            self.complexity += len(node.values) - 1  # type: ignore[attr-defined]
        elif name in ("If", "IfExp"):
            self.complexity += 1
        elif name == "Match":
            cases = node.cases  # type: ignore[attr-defined]
            underscore = any(getattr(c.pattern, "pattern", False) is None for c in cases)
            self.complexity += max(0, len(cases) - underscore)
        elif name in ("For", "While", "AsyncFor"):
            self.complexity += bool(node.orelse) + 1  # type: ignore[attr-defined]
        elif name == "comprehension":
            self.complexity += len(node.ifs) + 1  # type: ignore[attr-defined]
        super().generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function(node)

    def _function(self, node: Any) -> None:
        body = 1
        for child in node.body:
            body += _Complexity.run(child, off=False).complexity
        self.functions.append((node.name, body))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        body = 1
        for child in node.body:
            v = _Complexity.run(child, off=False)
            body += v.complexity + sum(c for _, c in v.functions) - len(v.functions) + len(v.functions)
            self.functions_methods = getattr(self, "functions_methods", [])
            self.functions_methods += [(f"{node.name}.{n}", c) for n, c in v.functions]
        self.classes.append(body)

    @property
    def total(self) -> int:
        f = sum(c for _, c in self.functions) - len(self.functions)
        k = sum(self.classes) - len(self.classes)
        return self.complexity + f + k + (not self.off)

    @property
    def blocks(self) -> list[tuple[str, int]]:
        return [*self.functions, *getattr(self, "functions_methods", [])]


def _halstead_volume(tree: ast.AST) -> float:
    """Halstead volume N·log2(η) with radon's operator / operand rules (functions analysed in their own
    context)."""
    ops_seen: set[Any] = set()
    opnds_seen: set[Any] = set()
    counts = [0, 0]

    def operand_key(node: ast.AST, ctx: Any) -> Any:
        if isinstance(node, ast.Name):
            return (ctx, node.id)
        if isinstance(node, ast.Attribute):
            return (ctx, node.attr)
        if isinstance(node, ast.Constant):
            return (ctx, node.value)
        return (ctx, id(node))  # radon keys any other operand node by identity

    def walk(node: ast.AST, ctx: Any) -> None:
        for child in ast.iter_child_nodes(node):
            visit(child, ctx)

    def visit(node: ast.AST, ctx: Any) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.body:
                visit(child, node.name)
            return
        rec: Optional[tuple[int, int, list[str], list[ast.AST]]] = None
        if isinstance(node, ast.BinOp):
            rec = (1, 2, [type(node.op).__name__], [node.left, node.right])
        elif isinstance(node, ast.UnaryOp):
            rec = (1, 1, [type(node.op).__name__], [node.operand])
        elif isinstance(node, ast.BoolOp):
            rec = (1, len(node.values), [type(node.op).__name__], list(node.values))
        elif isinstance(node, ast.AugAssign):
            rec = (1, 2, [type(node.op).__name__], [node.target, node.value])
        elif isinstance(node, ast.Compare):
            rec = (
                len(node.ops),
                len(node.comparators) + 1,
                [type(o).__name__ for o in node.ops],
                [*node.comparators, node.left],
            )
        if rec is not None:
            counts[0] += rec[0]
            counts[1] += rec[1]
            ops_seen.update(rec[2])
            opnds_seen.update(operand_key(o, ctx) for o in rec[3])
        walk(node, ctx)

    for child in ast.iter_child_nodes(tree):
        visit(child, None)
    n, eta = counts[0] + counts[1], len(ops_seen) + len(opnds_seen)
    return n * math.log2(eta) if eta > 0 else 0.0


def _line_tokens(line: str, lines: Any) -> tuple[list[Any], list[str]]:
    """Shortest tokenisable run of stripped lines starting at ``line`` (multi-line strings and statements)."""
    buffer, used = line, [line]
    while True:
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(buffer).readline))
        except tokenize.TokenError:
            pass
        else:
            if not any(t[0] == tokenize.ERRORTOKEN for t in tokens):
                return tokens, used
        nxt = next(lines)
        buffer += "\n" + nxt
        used.append(nxt)


def _logical_lines(tokens: list[Any]) -> int:
    """Logical lines in one physical run: ``if x: return y`` counts 2 (radon's rule)."""
    groups: list[list[Any]] = [[]]
    for t in tokens:
        if t[0] == tokenize.OP and t[1] == ";":
            groups.append([])
        else:
            groups[-1].append(t)
    total = 0
    for g in groups:
        kept = [t for t in g if t[0] not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE)]
        colon = [i for i, t in enumerate(kept) if t[0] == tokenize.OP and t[1] == ":"]
        if colon:
            total += 2 - (colon[-1] == len(kept) - 2)
        elif [t for t in kept if t[0] not in (tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER)]:
            total += 1
    return total


def _raw(src: str) -> tuple[int, int, int]:
    """(logical lines, source lines, comment + multi-line-string lines), counted line by line as radon's
    ``raw.analyze`` does."""
    lloc = comments = multi = sloc = 0
    lines = (ln.strip() for ln in src.splitlines())
    for line in lines:
        try:
            tokens, used = _line_tokens(line, lines)
        except StopIteration:  # pragma: no cover - the source already parsed
            break
        comments += sum(1 for t in tokens if t[0] == tokenize.COMMENT)
        rest = [t[0] for t in tokens[1:]]
        only = all(k in (tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE) for k in rest)
        if tokens[0][0] == tokenize.COMMENT and only:
            pass
        elif tokens[0][0] == tokenize.STRING and only:
            if tokens[0][3][0] != tokens[0][2][0]:
                multi += sum(1 for ln in used if ln)
        else:
            sloc += sum(1 for ln in used if ln)
        lloc += _logical_lines(tokens)
    return lloc, sloc, comments + multi


@register(
    category=_C,
    task="code-quality",
    name="Cyclomatic complexity / maintainability index",
    definition="McCabe cyclomatic complexity of each function (decision points + 1, counted as radon does) and "
    "the maintainability index MI = max(0, 100·(171 − 5.2 ln V − 0.23 G − 16.2 ln L + 50 sin √(2.46·rad(C))) / "
    "171) from Halstead volume V, total complexity G, logical lines L and comment percentage C. Python source.",
    formula="CC = 1 + decisions; MI as above",
    range="CC ≥ 1; MI in [0, 100]",
    input_requirements=("code",),
    references=(_REF_MCCABE, _REF_MI),
    higher_is_better=False,
)
def code_complexity(code: Any) -> MetricResult:
    """Mean cyclomatic complexity per function over all programs (the value), with the maximum, the mean
    maintainability index and per-program results. Programs that do not parse are rejected."""
    progs = _codes(code, "code")
    funcs: list[int] = []
    mi: list[float] = []
    per = []
    for i, src in enumerate(progs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(src)
        except SyntaxError as exc:
            raise InputValidationError(f"code[{i}] is not valid Python: {exc.msg} (line {exc.lineno}).") from exc
        visitor = _Complexity.run(tree)
        blocks = visitor.blocks
        funcs += [c for _, c in blocks]
        v = _halstead_volume(tree)
        g = visitor.total
        lloc, sloc, comments = _raw(src)
        c_pct = comments / sloc * 100 if sloc else 0.0
        if v <= 0 or lloc <= 0:
            m = 100.0
        else:
            nn = (
                171
                - 5.2 * math.log(v)
                - 0.23 * g
                - 16.2 * math.log(lloc)
                + 50 * math.sin(math.sqrt(2.46 * math.radians(c_pct)))
            )
            m = min(max(0.0, nn * 100 / 171.0), 100.0)
        mi.append(m)
        per.append({"functions": dict(blocks), "total_complexity": g, "maintainability_index": m})
    value = float(np.mean(funcs)) if funcs else 1.0
    return MetricResult(
        "code_complexity",
        "Mean cyclomatic complexity",
        value,
        {
            "max_complexity": max(funcs) if funcs else 1,
            "maintainability_index": float(np.mean(mi)),
            "per_program": per,
        },
    )


# ---------------------------------------------------------------------------------------------- CodeBLEU
PYTHON_KEYWORDS = [
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    "match",
    "case",
    "type",
]


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _closest(ref_lens: Sequence[int], hyp_len: int) -> int:
    return min(ref_lens, key=lambda r: (abs(r - hyp_len), r))


def _bp(r: int, c: int) -> float:
    if c > r:
        return 1.0
    if c == 0:
        return 0.0
    return math.exp(1 - r / c)


def _corpus_bleu(refs: list[list[list[str]]], hyps: list[list[str]], weighted: Optional[set[str]]) -> float:
    """BLEU-4 as in the CodeBLEU reference implementation (NLTK conventions, smoothing method 1, ε = 0.1).
    ``weighted`` (the keyword set) switches to the keyword-weighted n-gram match, whose unigram term is a
    recall over reference tokens weighted 1 for keywords and 0.2 otherwise."""
    num: dict[int, float] = {n: 0.0 for n in range(1, 5)}
    den: dict[int, float] = {n: 0.0 for n in range(1, 5)}
    hyp_len = ref_len = 0
    for rs, h in zip(refs, hyps):
        for n in range(1, 5):
            hc = _ngrams(h, n)
            if weighted is None:
                maxc: Counter[tuple[str, ...]] = Counter()
                for r in rs:
                    maxc |= _ngrams(r, n)
                num[n] += sum(min(c, maxc[g]) for g, c in hc.items())
                den[n] += max(1, sum(hc.values()))
            else:
                for r in rs:
                    rc = _ngrams(r, n)
                    clipped = {g: min(c, hc[g]) for g, c in rc.items()}
                    if n == 1:
                        w = {t: 1.0 if t in weighted else 0.2 for t in r}
                        num[n] += sum(c * w.get(g[0], 1) for g, c in clipped.items())
                        den[n] += max(1, sum(c * w.get(g[0], 1) for g, c in rc.items()))
                    else:
                        num[n] += sum(clipped.values())
                        den[n] += max(1, sum(rc.values()))
        hyp_len += len(h)
        ref_len += _closest([len(r) for r in rs], len(h))
    if num[1] == 0:
        return 0.0
    bp = _bp(ref_len, hyp_len)
    logs = [math.log((num[n] if num[n] else 0.1) / den[n]) for n in range(1, 5)]
    return bp * math.exp(math.fsum(0.25 * x for x in logs))


def _strip(src: str) -> str:
    """Remove comments and docstrings (as the CodeBLEU reference does before parsing)."""
    out, prev_type, last_line, last_col = [], tokenize.INDENT, -1, 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            ttype, text, (sl, sc), (el, ec), _ = tok
            if sl > last_line:
                last_col = 0
            if sc > last_col:
                out.append(" " * (sc - last_col))
            if ttype == tokenize.COMMENT or (
                ttype == tokenize.STRING and prev_type in (tokenize.INDENT, tokenize.NEWLINE)
            ):
                pass
            else:
                out.append(text)
            prev_type = ttype
            last_col, last_line = ec, el
    except (tokenize.TokenError, IndentationError):
        return src
    return "\n".join(ln for ln in "".join(out).split("\n") if ln.strip())


def _parse(src: str) -> Optional[ast.AST]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(_strip(src))
    except SyntaxError:
        return None


def _sexp(node: ast.AST, memo: dict[int, str]) -> str:
    key = id(node)
    if key in memo:
        return memo[key]
    kids = [
        c
        for c in ast.iter_child_nodes(node)
        if not isinstance(c, (ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop))
    ]
    tag = type(node).__name__
    if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.AugAssign)):
        tag += ":" + type(node.op).__name__
    s = f"({tag}{''.join(' ' + _sexp(c, memo) for c in kids)})"
    memo[key] = s
    return s


def _subtrees(tree: ast.AST) -> list[str]:
    memo: dict[int, str] = {}
    out = []
    for node in ast.walk(tree):
        kids = [
            c
            for c in ast.iter_child_nodes(node)
            if not isinstance(c, (ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop))
        ]
        if kids or node is tree:
            out.append(_sexp(node, memo))
    return out


def _dataflow(tree: ast.AST) -> list[tuple[str, str, tuple[str, ...]]]:
    """Def-use edges in source order: ``(var, 'computedFrom', sources)`` for assignments and
    ``(var, 'comesFrom', (var,))`` for uses of an earlier definition, as in CodeBLEU's data-flow graph."""
    edges: list[tuple[str, str, tuple[str, ...]]] = []
    defined: set[str] = set()

    def names(node: Optional[ast.AST]) -> list[str]:
        if node is None:
            return []
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]

    def targets(node: ast.AST) -> list[str]:
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]

    def uses(node: Optional[ast.AST]) -> None:
        if node is None:
            return
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in defined:
                edges.append((n.id, "comesFrom", (n.id,)))

    def stmt(s: ast.AST) -> None:
        if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            value = s.value
            uses(value)
            src = tuple(sorted(set(names(value))))
            tg = s.targets if isinstance(s, ast.Assign) else [s.target]
            for t in tg:
                for v in targets(t):
                    edges.append((v, "computedFrom", src + ((v,) if isinstance(s, ast.AugAssign) else ())))
                    defined.add(v)
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            uses(s.iter)
            for v in targets(s.target):
                edges.append((v, "computedFrom", tuple(sorted(set(names(s.iter))))))
                defined.add(v)
            for b in s.body + s.orelse:
                stmt(b)
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in s.args.posonlyargs + s.args.args + s.args.kwonlyargs:
                defined.add(a.arg)
            for b in s.body:
                stmt(b)
        else:
            body_fields = [
                f for f in ("body", "orelse", "finalbody", "handlers") if isinstance(getattr(s, f, None), list)
            ]
            for f, v in ast.iter_fields(s):
                if f not in body_fields and isinstance(v, ast.AST):
                    uses(v)
                elif f not in body_fields and isinstance(v, list):
                    for x in v:
                        if isinstance(x, ast.AST):
                            uses(x)
            for f in body_fields:
                for b in getattr(s, f):
                    if isinstance(b, ast.ExceptHandler):
                        if b.name:
                            defined.add(b.name)
                        for x in b.body:
                            stmt(x)
                    elif isinstance(b, ast.stmt):
                        stmt(b)

    for s in tree.body:  # type: ignore[attr-defined]
        stmt(s)
    return edges


def _normalize_flow(flow: list[tuple[str, str, tuple[str, ...]]]) -> list[tuple[str, str, tuple[str, ...]]]:
    names: dict[str, str] = {}
    out = []
    for var, rel, src in flow:
        for s in src:
            names.setdefault(s, f"var_{len(names)}")
        names.setdefault(var, f"var_{len(names)}")
        out.append((names[var], rel, tuple(names[s] for s in src)))
    return out


@register(
    category=_C,
    task="code-generation",
    name="CodeBLEU",
    definition="Weighted sum of BLEU, keyword-weighted n-gram match, AST subtree match and data-flow match "
    "between generated and reference code (Ren et al.). The two n-gram terms follow the reference "
    "implementation exactly; the syntax and data-flow terms use Python's own ``ast`` instead of tree-sitter, so "
    "they are defined for Python source.",
    formula="CodeBLEU = α·BLEU + β·BLEU_weight + γ·Match_ast + δ·Match_df",
    range="[0, 1]",
    input_requirements=("references", "predictions"),
    references=(_REF_CODEBLEU,),
)
def codebleu(
    references: Any,
    predictions: Any,
    *,
    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    keywords: Optional[Sequence[str]] = None,
    tokenizer: Optional[Callable[[str], list[str]]] = None,
) -> MetricResult:
    """``references``: one reference program (or a list of references) per prediction. Components are in
    ``params``. When no reference has any data flow the data-flow term counts as 1, as in the reference
    implementation."""
    preds = _codes(predictions, "predictions")
    refs_in = seq(references, "references")
    same_length(("references", refs_in), ("predictions", preds))
    refs = []
    for i, r in enumerate(refs_in):
        rs = [r] if isinstance(r, str) else list(r)
        if not rs or not all(isinstance(x, str) for x in rs):
            raise InputValidationError(f"references[{i}] must be a string or a list of strings.")
        refs.append([x.strip() for x in rs])
    preds = [p.strip() for p in preds]
    if len(weights) != 4 or any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
        raise InputValidationError("weights must be four non-negative numbers summing to 1.")
    tok = tokenizer or str.split
    kw = set(PYTHON_KEYWORDS if keywords is None else keywords)
    t_refs = [[tok(x) for x in rs] for rs in refs]
    t_hyps = [tok(p) for p in preds]
    ngram = _corpus_bleu(t_refs, t_hyps, None)
    weighted = _corpus_bleu(t_refs, t_hyps, kw)
    match = total = df_match = df_total = 0
    for rs, p in zip(refs, preds):
        cand = _parse(p)
        cand_sub = set(_subtrees(cand)) if cand is not None else set()
        cand_flow = _normalize_flow(_dataflow(cand)) if cand is not None else []
        for r in rs:
            ref = _parse(r)
            if ref is None:
                continue
            ref_sub = _subtrees(ref)
            match += sum(s in cand_sub for s in ref_sub)
            total += len(ref_sub)
            ref_flow = _normalize_flow(_dataflow(ref))
            pool = list(cand_flow)
            df_total += len(ref_flow)
            for e in ref_flow:
                if e in pool:
                    df_match += 1
                    pool.remove(e)
    syntax = match / total if total else 0.0
    dataflow = df_match / df_total if df_total else 0.0
    a, b, g, d = weights
    value = a * ngram + b * weighted + g * syntax + d * (dataflow or 1.0)
    return MetricResult(
        "codebleu",
        "CodeBLEU",
        value,
        {
            "ngram_match": ngram,
            "weighted_ngram_match": weighted,
            "syntax_match": syntax,
            "dataflow_match": dataflow,
            "weights": tuple(weights),
        },
    )
