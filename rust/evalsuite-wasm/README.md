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

Metrics: `es.metrics()`: accuracy, balanced_accuracy, precision, recall, f1, specificity, mcc, roc_auc,
average_precision, log_loss, brier_score, mae, mse, rmse, r2, mape, median_absolute_error, max_error, bleu, chrf,
ter, rouge_1, rouge_2, rouge_l, psi, ks. Arguments: `y_true`, `y_pred`, `y_prob`, `average`, `sample_weight`;
`references` / `predictions` for text; `reference` / `current` for drift.
