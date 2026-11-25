"""
    EvalSuite

Julia interface to the EvalSuite Rust engine (the C library `libevalsuite_c`, built with
`cargo build --release -p evalsuite-capi`; set `ENV["EVALSUITE_LIB"]` to its path).

```julia
using EvalSuite
metric("f1"; y_true = [0, 1, 1, 0], y_pred = [0, 1, 0, 0])   # 0.6666666666666666
```
"""
module EvalSuite

using JSON3
using Libdl

export metric, version

const HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function sym(name::Symbol)
    if HANDLE[] == C_NULL
        path = get(ENV, "EVALSUITE_LIB",
                   joinpath(@__DIR__, "..", "..", "..", "rust", "target", "release", "libevalsuite_c." * Libdl.dlext))
        HANDLE[] = Libdl.dlopen(path)
    end
    Libdl.dlsym(HANDLE[], name)
end

"""
    metric(name; kwargs...)

Compute an EvalSuite metric. Keywords: `y_true`, `y_pred`, `y_prob`, `average`, `sample_weight`;
`references` / `predictions` for text metrics; `reference` / `current` for drift (`"psi"`, `"ks"`).
"""
function metric(name::AbstractString; kwargs...)
    args = JSON3.write(Dict(String(k) => v for (k, v) in kwargs))
    ptr = ccall(sym(:evalsuite_call), Ptr{UInt8}, (Cstring, Cstring), name, args)
    if ptr == C_NULL
        err = ccall(sym(:evalsuite_last_error), Ptr{UInt8}, ())
        error(err == C_NULL ? "evalsuite error" : unsafe_string(err))
    end
    out = unsafe_string(ptr)
    ccall(sym(:evalsuite_free), Cvoid, (Ptr{UInt8},), ptr)
    res = JSON3.read(out)
    haskey(res, :values) ? collect(res.values) : res.value
end

"Engine version."
version() = unsafe_string(ccall(sym(:evalsuite_version), Ptr{UInt8}, ()))

end
