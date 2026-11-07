# Security policy

## Supported versions

Security fixes are released for the latest minor version of `evalsuite-python` (currently 0.5.x).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: **Security → Report a vulnerability** on
https://github.com/mkcs28/evalsuite-python (private vulnerability reporting). Do not open a public issue.

Include the affected version, a minimal example and the impact you expect. You will get an acknowledgement
within 7 days and a fix or mitigation plan within 30 days for confirmed issues; credit is given in the
changelog unless you prefer otherwise.

## How the project is protected

- Releases are published with PyPI Trusted Publishing (OpenID Connect); no API tokens are stored.
- Each release must be approved in the protected `pypi` environment, and the tag must match the version.
- Workflows run with read-only permissions by default; only the release steps get write access.
- CI audits runtime dependencies with `pip-audit` and scans the repository for committed credentials.
- The package never unpickles data (`np.load(..., allow_pickle=False)`), never runs shell commands and
  never evaluates user-supplied code. HTML reports escape all text and contain no scripts.
- Workflow actions are pinned to full commit SHAs and kept current by Dependabot.
- Untrusted text is handled defensively: XML with a DTD or entities is rejected before parsing, deeply
  nested JSON/XML/Python source is reported as invalid instead of crashing the interpreter, and the
  built-in PII detectors run in linear time on adversarial input.

## Patterns you supply

`format_compliance` (and any other function that takes a regular expression) compiles the pattern you pass
with Python's `re` module, which backtracks. A pattern such as `(a+)+$` can take exponential time on some
outputs. Only pass patterns you trust, and prefer patterns without nested quantifiers when scoring
untrusted model outputs.
