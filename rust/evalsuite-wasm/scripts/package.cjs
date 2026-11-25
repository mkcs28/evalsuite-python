// Assemble the npm package in ./npm: the Node build as the main entry and the web build under ./web.
const fs = require("node:fs");
const path = require("node:path");
const root = path.join(__dirname, "..");
const out = path.join(root, "npm");
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(path.join(out, "web"), { recursive: true });
for (const f of fs.readdirSync(path.join(root, "pkg-node"))) {
  if (f.startsWith("evalsuite")) fs.copyFileSync(path.join(root, "pkg-node", f), path.join(out, f));
}
for (const f of fs.readdirSync(path.join(root, "pkg-web"))) {
  if (f.startsWith("evalsuite")) fs.copyFileSync(path.join(root, "pkg-web", f), path.join(out, "web", f));
}
const cargo = fs.readFileSync(path.join(root, "..", "Cargo.toml"), "utf8");
const version = /^version = "(.+)"/m.exec(cargo)[1];
const pkg = {
  name: "evalsuite",
  version,
  description: "Evaluation metrics (classification, regression, text, drift) from the EvalSuite Rust engine, for Node and the browser via WebAssembly.",
  license: "MIT OR Apache-2.0",
  author: "Manoj Kumar C S <cs.manojkumar18@gmail.com>",
  repository: { type: "git", url: "https://github.com/mkcs28/evalsuite-python" },
  homepage: "https://github.com/mkcs28/evalsuite-python",
  keywords: ["metrics", "evaluation", "machine-learning", "wasm", "bleu", "f1", "roc-auc"],
  main: "evalsuite.js",
  types: "evalsuite.d.ts",
  exports: {
    ".": { types: "./evalsuite.d.ts", browser: "./web/evalsuite.js", default: "./evalsuite.js" },
    "./web": "./web/evalsuite.js",
  },
  files: ["evalsuite*", "web/", "README.md"],
};
fs.writeFileSync(path.join(out, "package.json"), JSON.stringify(pkg, null, 2) + "\n");
fs.copyFileSync(path.join(root, "README.md"), path.join(out, "README.md"));
console.log("npm package", version, "in", out);
