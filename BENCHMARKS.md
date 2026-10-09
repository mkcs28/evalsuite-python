# Benchmarks

Reproduce with `evalsuite benchmark` (all cases), `evalsuite benchmark --suite core` (classification and
regression) or `evalsuite benchmark --suite clinical` (the v0.2.0 clinical, calibration and statistics
functions). Add `--format markdown -o BENCHMARKS-local.md` to save a table.

Each case is timed as the fastest of 5 runs after a warm-up; peak memory is measured with `tracemalloc`.
EvalSuite and a **reference implementation compute the same quantities on the same data**, and the last
column is the largest absolute difference between their results: every row agrees to floating-point
rounding.

| Cases | Reference |
| --- | --- |
| classification, regression, sensitivity/specificity/likelihood ratios | scikit-learn |
| diagnostic report intervals, calibration slope and intercept, Hochberg correction | statsmodels |
| t-test, Mann–Whitney, Cramér's V | SciPy |
| decision curve | the textbook NumPy loop over thresholds |

```text
EvalSuite 0.2.0 benchmarks | Python 3.12.3 | NumPy 2.5.3 | scikit-learn 1.9.1 | statsmodels 0.15.0 | SciPy 1.18.1 | Linux x86_64 | fastest of 5 runs
```

| Case | n | Reference | EvalSuite (ms) | Reference (ms) | Speed-up | EvalSuite peak (MiB) | Reference peak (MiB) | Max |difference| |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| binary: 8 label metrics via evaluate() | 1,000 | scikit-learn | 0.276 | 8.409 | 30.47× | 0.05 | 0.05 | 0.0e+00 |
| 10 classes: macro F1 | 1,000 | scikit-learn | 0.103 | 1.166 | 11.32× | 0.05 | 0.03 | 0.0e+00 |
| binary: ROC AUC | 1,000 | scikit-learn | 0.161 | 1.425 | 8.86× | 0.09 | 0.08 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000 | scikit-learn | 0.093 | 0.606 | 6.53× | 0.03 | 0.02 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000 | scikit-learn | 0.343 | 3.682 | 10.72× | 0.05 | 0.03 | 4.4e-16 |
| clinical: diagnostic report (7 CIs) | 1,000 | statsmodels | 0.147 | 0.508 | 3.45× | 0.05 | 0.01 | 1.1e-14 |
| calibration: slope and intercept | 1,000 | statsmodels | 0.459 | 2.339 | 5.10× | 0.10 | 0.61 | 2.8e-16 |
| decision curve: 99 thresholds | 1,000 | NumPy loop | 0.152 | 0.842 | 5.53× | 0.07 | 0.01 | 5.6e-17 |
| statistics: Welch t-test | 1,000 | SciPy | 0.687 | 0.542 | 0.79× | 0.04 | 0.02 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000 | SciPy | 0.630 | 0.575 | 0.91× | 0.16 | 0.14 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000 | SciPy | 0.253 | 0.273 | 1.08× | 0.00 | 0.00 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000 | statsmodels | 0.044 | 0.050 | 1.16× | 0.05 | 0.05 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 100,000 | scikit-learn | 4.028 | 104.399 | 25.92× | 3.21 | 3.07 | 0.0e+00 |
| 10 classes: macro F1 | 100,000 | scikit-learn | 4.511 | 16.272 | 3.61× | 3.05 | 2.18 | 0.0e+00 |
| binary: ROC AUC | 100,000 | scikit-learn | 15.178 | 28.997 | 1.91× | 9.16 | 7.64 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 100,000 | scikit-learn | 1.960 | 1.708 | 0.87× | 2.29 | 1.53 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 100,000 | scikit-learn | 10.355 | 43.505 | 4.20× | 3.05 | 2.24 | 8.9e-16 |
| clinical: diagnostic report (7 CIs) | 100,000 | statsmodels | 2.537 | 0.870 | 0.34× | 3.05 | 0.29 | 0.0e+00 |
| calibration: slope and intercept | 100,000 | statsmodels | 17.503 | 95.133 | 5.44× | 8.46 | 58.00 | 2.2e-16 |
| decision curve: 99 thresholds | 100,000 | NumPy loop | 12.774 | 13.534 | 1.06× | 6.87 | 0.29 | 5.6e-17 |
| statistics: Welch t-test | 100,000 | SciPy | 1.966 | 1.140 | 0.58× | 3.06 | 1.53 | 0.0e+00 |
| statistics: Mann–Whitney U | 100,000 | SciPy | 28.748 | 30.390 | 1.06× | 15.45 | 13.93 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 100,000 | SciPy | 0.225 | 0.217 | 0.97× | 0.00 | 0.00 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 100,000 | statsmodels | 3.689 | 3.951 | 1.07× | 4.58 | 3.97 | 0.0e+00 |
| binary: 8 label metrics via evaluate() | 1,000,000 | scikit-learn | 30.225 | 1020.196 | 33.75× | 31.54 | 30.53 | 0.0e+00 |
| 10 classes: macro F1 | 1,000,000 | scikit-learn | 22.373 | 128.679 | 5.75× | 30.52 | 21.79 | 0.0e+00 |
| binary: ROC AUC | 1,000,000 | scikit-learn | 173.236 | 300.584 | 1.74× | 91.56 | 76.30 | 1.1e-16 |
| regression: MAE, MSE, RMSE, R² via evaluate() | 1,000,000 | scikit-learn | 19.067 | 9.739 | 0.51× | 22.89 | 15.26 | 0.0e+00 |
| clinical: sensitivity, specificity, LR+, LR− | 1,000,000 | scikit-learn | 66.196 | 392.770 | 5.93× | 30.52 | 22.33 | 5.6e-17 |
| clinical: diagnostic report (7 CIs) | 1,000,000 | statsmodels | 16.607 | 4.568 | 0.28× | 30.52 | 1.91 | 1.4e-14 |
| calibration: slope and intercept | 1,000,000 | statsmodels | 177.987 | 1014.801 | 5.70× | 83.99 | 579.85 | 1.3e-15 |
| decision curve: 99 thresholds | 1,000,000 | NumPy loop | 155.238 | 174.335 | 1.12× | 68.67 | 1.97 | 5.6e-17 |
| statistics: Welch t-test | 1,000,000 | SciPy | 14.458 | 7.323 | 0.51× | 30.52 | 15.26 | 0.0e+00 |
| statistics: Mann–Whitney U | 1,000,000 | SciPy | 365.904 | 364.548 | 1.00× | 154.50 | 139.24 | 0.0e+00 |
| statistics: Cramér's V (5×5 table) | 1,000,000 | SciPy | 0.493 | 0.421 | 0.85× | 0.00 | 0.00 | 0.0e+00 |
| multiple testing: Hochberg (n p-values) | 1,000,000 | statsmodels | 75.358 | 81.921 | 1.09× | 45.78 | 39.17 | 0.0e+00 |

## Reading the results

- **Many metrics at once is where EvalSuite is fastest.** `evaluate()` validates the inputs once and builds
  the confusion matrix once, then derives all eight label metrics from it: 26–34× faster than eight
  separate scikit-learn calls. The same applies to sensitivity, specificity, LR+ and LR− (4–11×).
- **Calibration slope and intercept are 5–6× faster than statsmodels' GLM and use far less memory**
  (84 MiB vs 580 MiB at 1M samples): EvalSuite fits the two small logistic models with a dedicated
  Newton–Raphson solver.
- **Decision curves** sort the risks once and read every threshold from cumulative sums, so all 99
  thresholds cost about the same as the plain loop at large n and are 5× faster at small n.
- **Hypothesis tests call SciPy** for the statistic and p-value, so they cannot be faster than SciPy. At
  large n the Welch t-test is about 0.5× SciPy's speed because EvalSuite also validates the inputs and
  computes the confidence interval and Cohen's d; rank tests and Cramér's V are on par.
- **The diagnostic report is slower than the reference at large n (0.3×).** The reference here only
  computes seven intervals from counts taken directly with NumPy; EvalSuite also validates the labels, checks
  for NaN, resolves the positive class, and reports ten measures (including likelihood ratios and Youden's
  J). 17 ms for a million observations is the price of those checks.
- **Regression on large arrays is slower** (0.87× at 100k, 0.51× at 1M). EvalSuite checks every input for
  NaN and infinity, its shape and its dtype before computing; for four cheap metrics on a million values
  those checks dominate a ~20 ms run.
- **Memory** is comparable in every case except the decision curve, which keeps a sorted copy of the data.

Since 0.2.1, integer class labels are found with one marking pass instead of a sort, which is why the
classification rows are faster than in 0.1.x (for example macro F1 at 1M: 5.8× vs 1.6× before).

Numbers depend on the machine, Python and NumPy versions; run `evalsuite benchmark` on your own hardware.
