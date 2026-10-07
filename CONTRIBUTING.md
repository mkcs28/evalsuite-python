# Contributing to EvalSuite

Thank you for helping. EvalSuite values correctness and clarity over breadth.

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
ruff check . && ruff format --check .
mypy                        # strict
```

## Reporting issues

Include the EvalSuite, Python and NumPy versions, a minimal example, the expected result (with a reference)
and the actual result.
