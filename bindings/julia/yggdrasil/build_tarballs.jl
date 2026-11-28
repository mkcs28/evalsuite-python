# BinaryBuilder recipe for EvalSuite_jll (the EvalSuite C library for every Julia platform).
# Submit as E/EvalSuite/build_tarballs.jl in a pull request to https://github.com/JuliaPackaging/Yggdrasil;
# once merged, EvalSuite.jl can depend on EvalSuite_jll instead of a locally built library.
using BinaryBuilder

name = "EvalSuite"
version = v"1.4.0"

sources = [
    GitSource("https://github.com/mkcs28/evalsuite-python.git", "COMMIT_SHA_OF_v1.4.0"),
]

script = raw"""
cd ${WORKSPACE}/srcdir/evalsuite-python/rust
CARGO_PROFILE_RELEASE_PANIC=unwind cargo build --release -p evalsuite-capi
install -Dvm 755 "target/${rust_target}/release/"*evalsuite_c.${dlext} "${libdir}/libevalsuite_c.${dlext}"
install -Dvm 644 evalsuite-capi/include/evalsuite.h "${includedir}/evalsuite.h"
"""

platforms = supported_platforms()
# Rust targets BinaryBuilder cannot build.
filter!(p -> !(Sys.iswindows(p) && arch(p) == "i686"), platforms)
filter!(p -> arch(p) != "riscv64", platforms)

products = [
    LibraryProduct("libevalsuite_c", :libevalsuite_c),
]

dependencies = Dependency[]

build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies;
               compilers = [:c, :rust], julia_compat = "1.6", preferred_gcc_version = v"10")
