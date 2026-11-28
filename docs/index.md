# EvalSuite

Unified, reproducible evaluation for machine learning and research — every metric with a confidence interval.

```bash
pip install evalsuite-python
```

```python
import evalsuite as es

result = es.evaluate(y_true, y_pred, y_prob=y_prob)
print(result.summary())
```

- **Guides** live in the [README](https://github.com/mkcs28/evalsuite-python#readme).
- **API reference**: see the sections in the navigation, generated from the docstrings of this version.
- **Stability**: see the [deprecation policy](deprecation.md).
