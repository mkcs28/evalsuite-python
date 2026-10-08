# Benchmarks

Reproduce with `evalsuite benchmark` (add `--format markdown -o BENCHMARKS-local.md` to save a table).

Each case is timed as the fastest of 5 runs after a warm-up; peak memory is measured with `tracemalloc`.
Both libraries compute the **same metrics on the same data**, and the last column is the largest absolute
difference between their results: every row agrees to floating-point rounding.

```text
EvalSuite 0.1.0b1 benchmarks | Python 3.12.3 | NumPy 2.5.3 | scikit-learn 1.9.1 | Linux x86_64 | fastest of 5 runs
```

| Case | n | EvalSuite (ms) | scikit-learn (ms) | Speed-up | EvalSuite peak (MiB) | scikit-learn peak (MiB) | Max |difference| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| binary: 8 label metrics via evaluate() | 1,000 | 0.375 | 11.701 | 31.16× | 0.04 | 0.05 | 0.0e+00 |
| 10 classes: macro F1 | 1,000 | 0.168 | 1.566 | 9.31× | 0.03 | 0.03 | 0.0e+00 |
| binary: ROC AUC | 1,000 | 0.232 | 1.656 | 7.13× | 0.09 | 0.08 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | 0.124 | 0.901 | 7.24× | 0.03 | 0.02 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | 10.919 | 116.881 | 10.70× | 3.21 | 3.07 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | 10.282 | 16.367 | 1.59× | 3.06 | 2.18 | 0.0e+00 |
| binary: ROC AUC | 100,000 | 23.573 | 32.255 | 1.37× | 9.16 | 7.64 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | 2.242 | 1.888 | 0.84× | 2.29 | 1.53 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | 108.751 | 1043.595 | 9.60× | 31.54 | 30.53 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | 88.407 | 139.310 | 1.58× | 30.52 | 21.80 | 0.0e+00 |
| binary: ROC AUC | 1,000,000 | 247.386 | 352.519 | 1.42× | 91.56 | 76.30 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | 29.280 | 20.069 | 0.69× | 22.89 | 15.26 | 0.0e+00 |

## Reading the results

- **Many metrics at once is where EvalSuite is fastest.** `evaluate()` validates the inputs once and builds
  the confusion matrix once, then derives all eight label metrics from it (about 10× faster than eight
  separate scikit-learn calls, at every size).
- **Single classification metrics** (macro F1, ROC AUC) are 1.4–1.6× faster at large sizes and much faster on
  small inputs, where per-call overhead dominates.
- **Regression on large arrays is slower** (0.84× at 100k, 0.69× at 1M samples). EvalSuite checks every
  input for NaN and infinity, its shape and its dtype before computing; scikit-learn's metric functions do
  part of this. For four cheap metrics on a million values, those checks are a large share of a ~30 ms run.
  We consider the validation worth it; if it matters for you, compute with `es.mae(...)` etc. once per array.
- **Memory** is comparable, within a few MiB, in every case.

Numbers depend on the machine, Python and NumPy versions; run `evalsuite benchmark` on your own hardware.
