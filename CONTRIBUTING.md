# Contributing to EvalSuite

Thank you for helping. EvalSuite values correctness and clarity over breadth.

## Development setup

EvalSuite has a compiled core written in Rust (`rust/`), built into `evalsuite._core` by
[maturin](https://www.maturin.rs). Install a Rust toolchain ([rustup](https://rustup.rs)), then:

```bash
pip install -e ".[dev]"      # builds the core; rerun after changing anything under rust/
python -c "from evalsuite import _native; print(_native.backend())"   # rust
```

Every function with a compiled kernel keeps its Python implementation as the reference and fallback. Set
`EVALSUITE_PURE_PYTHON=1` to run without the core; `tests/test_native.py` checks that both give identical
results, and must keep passing.

## Adding or changing a compiled kernel

1. Port the Python reference operation for operation (same counts, tie-breaks and floating-point order) into
   `rust/evalsuite-core` (pure Rust, no Python), and expose it in `rust/evalsuite-py`.
2. Keep the Python path and call the core from it only when `evalsuite._native.core` is available.
3. Add a parity test to `tests/test_native.py` comparing both paths, then run the checks below.

## Adding or changing a metric

1. Implement it with `@register(...)`: definition, formula, range, input requirements and at least one
   reference.
2. Validate inputs through `evalsuite.core.validation`; raise `InputValidationError` for bad inputs and
   `MetricInputError` for values outside the metric's domain, with a message saying how to fix it.
3. Test it: against an established implementation where definitions coincide (scikit-learn, SciPy), plus
   hand-computed cases, edge cases and, where useful, property-based tests.
4. Document conventions explicitly (averaging, label order, zero division) and update `CHANGELOG.md`.

## Checks

```bash
pytest --cov=evalsuite      # coverage must stay at or above 95% for core modules
EVALSUITE_PURE_PYTHON=1 pytest -q   # the same suite without the compiled core
ruff check . && ruff format --check .
mypy                        # strict
cargo fmt --manifest-path rust/Cargo.toml --all --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml -p evalsuite-core
```

## Reporting issues

Include the EvalSuite, Python and NumPy versions, a minimal example, the expected result (with a reference)
and the actual result.
