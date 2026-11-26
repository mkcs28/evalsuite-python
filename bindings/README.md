# Language bindings

Every binding calls the same Rust engine through `evalsuite::bridge` (JSON in, JSON out), so a metric gives the
same value in every language.

## R

* **From a release:** `install.packages("https://github.com/mkcs28/evalsuite-python/releases/download/v1.1.0/evalsuite_1.1.0.tar.gz", repos = NULL)`
  (needs Rust's `cargo`; the tarball carries all Rust sources and builds offline).
* **R-universe:** the `r-package` branch is updated on every release. To publish, create the repository
  `mkcs28/mkcs28.r-universe.dev` with this `packages.json`:

  ```json
  [{"package": "evalsuite", "url": "https://github.com/mkcs28/evalsuite-python", "branch": "r-package"}]
  ```

  Users then run `install.packages("evalsuite", repos = c("https://mkcs28.r-universe.dev", "https://cloud.r-project.org"))`.
* **From the repository:** `R CMD INSTALL bindings/r`.

## Julia

* **Now:** build the library (`cargo build --release -p evalsuite-capi` in `rust/`), set
  `ENV["EVALSUITE_LIB"]` to `libevalsuite_c`, and `] dev bindings/julia`.
* **Registry:** submit `julia/yggdrasil/build_tarballs.jl` to Yggdrasil (fill in the release commit) to get
  `EvalSuite_jll` with prebuilt libraries for every platform; then register `EvalSuite.jl` in the General registry
  with JuliaRegistrator.

## C

`rust/evalsuite-capi/include/evalsuite.h`; link with `-levalsuite_c`.

## JavaScript

`npm install evalsuite` (WebAssembly, Node and browsers).
