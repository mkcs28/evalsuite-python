# evalsuite (npm)

The EvalSuite metric engine for JavaScript, compiled from Rust to WebAssembly: the same results as the
`evalsuite-python` package and the `evalsuite` crate.

```js
// Node
const es = require("evalsuite");
es.evaluate("f1", { y_true: [0, 1, 1, 0], y_pred: [0, 1, 0, 0] }); // { metric: "f1", value: 0.6666666666666666 }

// Browser / bundlers
import init, { evaluate } from "evalsuite/web";
await init();
evaluate("bleu", { references: ["the cat sat on the mat"], predictions: ["the cat sat on a mat"] });
```

Metrics (`es.metrics()`): classification (accuracy, balanced_accuracy, precision, recall, f1, specificity, mcc,
roc_auc, average_precision, log_loss, brier_score), clinical (sensitivity, ppv, npv, lr_positive, lr_negative,
diagnostic_odds_ratio, youden_j, net_benefit), calibration (calibration_slope, calibration_intercept,
expected_calibration_error, maximum_calibration_error, hosmer_lemeshow), regression (mae, mse, rmse, r2, mape,
median_absolute_error, max_error), text (bleu, chrf, ter, rouge_1, rouge_2, rouge_l), retrieval (ndcg, mrr, map,
precision_at_k, recall_at_k, hit_rate), segmentation (dice, iou) and drift (psi, ks).

Arguments: `y_true`, `y_pred`, `y_prob`, `average`, `sample_weight`, `pos_label`, `threshold`; `references` /
`predictions` for text; `relevant` (lists of ids or `{id: grade}`), `retrieved` and `k` for retrieval; flat
`y_true` / `y_pred` masks with `shape` for segmentation; `reference` / `current` for drift.

Confidence intervals: add `ci: "analytic"` (Wilson for proportions, DeLong for ROC AUC, bootstrap otherwise) or
`ci: "bootstrap"` (BCa; `n_resamples`, `seed`, `level`):

```js
es.evaluate("roc_auc", { y_true, y_prob, ci: "analytic" });
// { metric: "roc_auc", value: 0.83, ci_low: 0.71, ci_high: 0.95, ci_level: 0.95, ci_method: "delong" }
```
