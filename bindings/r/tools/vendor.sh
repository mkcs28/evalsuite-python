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
(cd "$dst" && cargo vendor --versioned-dirs vendor > vendor-config.toml)
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
echo "Vendored Rust sources into $dst"
