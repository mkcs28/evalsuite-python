const assert = require("node:assert");
const es = require("../pkg-node/evalsuite.js");

const r = es.evaluate("f1", { y_true: [0, 1, 1, 0], y_pred: [0, 1, 0, 0] });
assert.strictEqual(r.value, 0.6666666666666666);
assert.deepStrictEqual(es.evaluate("recall", { y_true: [0, 1, 2, 2], y_pred: [0, 1, 2, 1], average: "none" }).values, [1, 1, 0.5]);
assert.ok(es.evaluate("bleu", { references: ["the cat sat on the mat"], predictions: ["the cat sat on the mat"] }).value > 99.9);
assert.throws(() => es.evaluate("nope", {}), /unknown metric/);
assert.ok(es.metrics().includes("roc_auc"));
console.log("ok", es.version());
