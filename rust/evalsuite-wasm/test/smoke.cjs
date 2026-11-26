const assert = require("node:assert");
const es = require("../pkg-node/evalsuite.js");

const r = es.evaluate("f1", { y_true: [0, 1, 1, 0], y_pred: [0, 1, 0, 0] });
assert.strictEqual(r.value, 0.6666666666666666);
assert.deepStrictEqual(es.evaluate("recall", { y_true: [0, 1, 2, 2], y_pred: [0, 1, 2, 1], average: "none" }).values, [1, 1, 0.5]);
assert.ok(es.evaluate("bleu", { references: ["the cat sat on the mat"], predictions: ["the cat sat on the mat"] }).value > 99.9);
assert.throws(() => es.evaluate("nope", {}), /unknown metric/);
assert.ok(es.metrics().includes("roc_auc") && es.metrics().includes("ndcg"));
const ci = es.evaluate("roc_auc", { y_true: [0, 0, 1, 1, 0, 1], y_prob: [0.1, 0.4, 0.35, 0.8, 0.2, 0.9], ci: "analytic" });
assert.strictEqual(ci.ci_method, "delong");
assert.ok(ci.ci_low <= ci.value && ci.value <= ci.ci_high);
assert.strictEqual(es.evaluate("mrr", { relevant: [["a"], ["x"]], retrieved: [["b", "a"], ["x"]] }).value, 0.75);
console.log("ok", es.version());
