# Security policy

## Supported versions

Security fixes are released for the latest minor version of `evalsuite-python` (currently 0.3.x).

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
