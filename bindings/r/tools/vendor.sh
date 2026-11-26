#!/usr/bin/env bash
# Make the R package self-contained (for R-universe, CRAN-style source tarballs and offline installs): copy the
# Rust crates it needs into src/rust and vendor their dependencies, so `R CMD INSTALL` builds without the
# repository or network access.  Run from bindings/r:  tools/vendor.sh
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$here/../.." && pwd)
dst="$here/src/rust"
rm -rf "$dst"
mkdir -p "$dst"
for c in evalsuite-core evalsuite evalsuite-capi; do
  mkdir -p "$dst/$c"
  cp -R "$repo/rust/$c/Cargo.toml" "$repo/rust/$c/src" "$dst/$c/"
  [ -d "$repo/rust/$c/include" ] && cp -R "$repo/rust/$c/include" "$dst/$c/"
  [ -f "$repo/rust/$c/build.rs" ] && cp "$repo/rust/$c/build.rs" "$dst/$c/"
  [ -f "$repo/rust/$c/README.md" ] && cp "$repo/rust/$c/README.md" "$dst/$c/"
done
# A workspace with only these crates, same package metadata and release profile.
python3 - "$repo/rust/Cargo.toml" "$dst/Cargo.toml" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
src = re.sub(r'members = \[[^\]]*\]', 'members = ["evalsuite-core", "evalsuite", "evalsuite-capi"]', src)
open(sys.argv[2], "w").write(src)
PY
cp "$repo/rust/Cargo.lock" "$dst/Cargo.lock"
(cd "$dst" && cargo vendor v > vendor-config.toml)
mkdir -p "$dst/.cargo"
mv "$dst/vendor-config.toml" "$dst/.cargo/config.toml"
# Drop test fixtures and benches from the copied crates (not needed to build, and large), and their targets.
rm -rf "$dst"/*/tests "$dst"/*/benches "$dst"/*/examples
python3 - "$dst" <<'PY'
import glob, re, sys
for p in glob.glob(sys.argv[1] + "/evalsuite*/Cargo.toml"):
    s = open(p).read()
    s = re.sub(r"\n\[\[(bench|test|example)\]\][^\[]*", "\n", s)
    open(p, "w").write(s)
PY
# Trim vendored crates to what a build needs (tests, benches, examples, docs carry very long paths that R's tar
# format rejects) and clear the file list in their checksums so cargo accepts the trimmed sources.
python3 - "$dst/v" <<'PY'
import json, os, shutil, sys
root = sys.argv[1]
for crate in os.listdir(root):
    d = os.path.join(root, crate)
    for sub in ("tests", "benches", "examples", "docs", "doc", ".github", "ci", "src/bin_util", "src/snapshots", "src/output_tests", "src/test_util"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    ck = os.path.join(d, ".cargo-checksum.json")
    if os.path.exists(ck):
        data = json.load(open(ck))
        data["files"] = {}
        json.dump(data, open(ck, "w"))
# windows-sys: keep only the API modules some crate enables (module path = feature name); the rest are never
# compiled and include the longest paths.
try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None
if tomllib:
    wanted = {}
    for crate in os.listdir(root):
        try:
            man = tomllib.load(open(os.path.join(root, crate, "Cargo.toml"), "rb"))
        except Exception:
            continue
        tables = [man.get("dependencies", {})]
        tables += [t.get("dependencies", {}) for t in man.get("target", {}).values()]
        for deps in tables:
            for name, spec in deps.items():
                pkg = spec.get("package", name) if isinstance(spec, dict) else name
                if pkg == "windows-sys" and isinstance(spec, dict):
                    wanted.setdefault("windows-sys", set()).update(spec.get("features", []))
    for crate in os.listdir(root):
        if not (crate == "windows-sys" or crate.startswith("windows-sys-")):
            continue
        d = os.path.join(root, crate)
        feats = tomllib.load(open(os.path.join(d, "Cargo.toml"), "rb")).get("features", {})
        on, todo = set(), list(wanted.get("windows-sys", ()))
        while todo:
            f = todo.pop()
            if f not in on:
                on.add(f)
                todo += [x for x in feats.get(f, []) if not x.startswith("dep:")]
        base = os.path.join(d, "src", "Windows")
        for p, dirs, _ in os.walk(base, topdown=True):
            rel = os.path.relpath(p, base)
            for sub in list(dirs):
                feat = (sub if rel == "." else rel.replace(os.sep, "_") + "_" + sub)
                if feat not in on:
                    shutil.rmtree(os.path.join(p, sub))
                    dirs.remove(sub)
long = [p for p, _, fs in os.walk(root) for f in fs if len(os.path.relpath(os.path.join(p, f), root)) > 78]
if long:
    sys.exit("paths too long for an R source package:\n" + "\n".join(long[:20]))
PY
echo "Vendored Rust sources into $dst"
