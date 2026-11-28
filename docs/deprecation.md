# Stability and deprecation policy

EvalSuite follows [semantic versioning](https://semver.org/) from 2.0.0 on.

- **Patch releases (2.0.x)** fix bugs. They never change a public name or signature, and they never change a
  result unless the old result was wrong (the changelog says so under *Fixed*).
- **Minor releases (2.x.0)** add features. Existing code keeps working and keeps producing the same numbers.
- **Major releases (3.0.0)** may remove or change public API, but only after a deprecation period.

**Public API** means every name in `evalsuite.__all__`, the command-line interface, and the documented keyword
arguments. Names starting with `_` are private and may change at any time.

**Deprecation.** A feature to be removed first emits a `DeprecationWarning` (category `EvalSuiteDeprecationWarning`)
naming its replacement, for at least two minor releases and at least six months, and is listed under
*Deprecated* in the changelog. It is removed only in the next major release.

**Results.** Default settings (resample counts, interval methods, random-number streams) are part of the
contract: changing one so that results move is treated as a breaking change, unless it fixes a bug. Seeded
results (`random_state=`) are reproducible across platforms and between the compiled core and pure Python.

**Supported versions.** The latest minor release receives fixes; security fixes go to the latest two minor
releases. Python versions follow NumPy's support window.
