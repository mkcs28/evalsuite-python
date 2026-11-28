"""Data quality before evaluation: leakage between splits, likely label errors, and a summary card of a data set.

>>> import evalsuite as es
>>> es.leakage_report(train_df, test_df, target="label", id_column="patient_id")
>>> es.label_issues(y, pred_probs)            # likely mislabelled examples (confident learning)
>>> es.dataset_card(test_df, target="label", groups=["sex", "site"])

``leakage_report`` finds test rows that also appear in training (exact duplicates after normalising numbers and
text), near-duplicates (numeric rows within ``near_tol`` standardised units of a training row; texts whose
word-shingle Jaccard similarity is at least ``text_threshold``, found with MinHash), shared identifiers (the same
patient or document in both splits) and features that predict the target almost perfectly (a sign that the
answer leaked into the inputs). ``label_issues`` implements confident learning (Northcutt, Jiang & Chuang 2021):
an example is flagged when its out-of-sample probability for another class clears that class's average
self-confidence threshold. ``dataset_card`` summarises size, missing values, class balance, feature types,
subgroup counts and duplicates.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core.exceptions import InputValidationError

__all__ = ["LeakageReport", "leakage_report", "LabelIssues", "label_issues", "DatasetCard", "dataset_card"]


def _table(data: Any, what: str) -> dict[str, np.ndarray]:
    if hasattr(data, "column_names") and hasattr(data, "column"):  # pyarrow
        cols = {str(c): np.asarray(data.column(c).to_numpy(zero_copy_only=False)) for c in data.column_names}
    elif hasattr(data, "columns") and hasattr(data, "__getitem__") and not isinstance(data, np.ndarray):
        cols = {str(c): np.asarray(data[c]) for c in data.columns}
    elif isinstance(data, Mapping):
        cols = {str(k): np.asarray(v) for k, v in data.items()}
    else:
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise InputValidationError(f"{what} must be a table (DataFrame, dict of columns or 2-D array).")
        cols = {f"x{i}": arr[:, i] for i in range(arr.shape[1])}
    lengths = {v.shape[0] for v in cols.values()}
    if len(lengths) != 1:
        raise InputValidationError(f"All columns of {what} must have the same length.")
    if not cols:
        raise InputValidationError(f"{what} has no columns.")
    return cols


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC of any real scores by the Mann–Whitney rank sum (ties count one half)."""
    from scipy.stats import rankdata

    npos = int(y.sum())
    nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return math.nan
    return float((rankdata(s)[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _is_numeric(v: np.ndarray) -> bool:
    return v.dtype.kind in "iufb"


def _missing(v: np.ndarray) -> np.ndarray:
    if v.dtype.kind in "fc":
        return np.asarray(~np.isfinite(v))
    if v.dtype.kind == "O":
        return np.array([x is None or (isinstance(x, float) and math.isnan(x)) for x in v], dtype=bool)
    return np.zeros(v.shape[0], dtype=bool)


def _norm_cell(x: Any) -> str:
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)):
            return "nan"
        return repr(round(float(x), 9))
    if isinstance(x, (int, np.integer, bool, np.bool_)):
        return repr(float(x))
    if x is None:
        return "nan"
    return re.sub(r"\s+", " ", str(x).strip().lower())


def _row_hashes(cols: Mapping[str, np.ndarray], names: Sequence[str]) -> list[str]:
    n = len(next(iter(cols.values())))
    out = []
    for i in range(n):
        key = "\x1f".join(_norm_cell(cols[c][i]) for c in names)
        out.append(hashlib.blake2b(key.encode(), digest_size=16).hexdigest())
    return out


def _shingles(text: str, k: int) -> set[str]:
    words = re.findall(r"\w+", str(text).lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(sets: list[set[str]], n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prime = (1 << 61) - 1
    a = rng.integers(1, prime, n_perm, dtype=np.uint64).astype(object)
    b = rng.integers(0, prime, n_perm, dtype=np.uint64).astype(object)
    sig = np.full((len(sets), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(sets):
        if not s:
            continue
        hs = [int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "little") for x in s]
        for j in range(n_perm):
            aj, bj = int(a[j]), int(b[j])
            sig[i, j] = min((aj * h + bj) % prime for h in hs)
    return sig


def _near_text(train: np.ndarray, test: np.ndarray, threshold: float, k: int) -> list[tuple[int, int, float]]:
    """(test index, train index, Jaccard) for test texts with a training text of similarity ≥ threshold."""
    tr_sets = [_shingles(t, k) for t in train]
    te_sets = [_shingles(t, k) for t in test]
    n_perm, bands = 64, 16
    rows = n_perm // bands
    sig_tr = _minhash(tr_sets, n_perm, 0)
    sig_te = _minhash(te_sets, n_perm, 0)
    buckets: dict[tuple[int, bytes], list[int]] = {}
    for i in range(len(tr_sets)):
        if not tr_sets[i]:
            continue
        for bnd in range(bands):
            buckets.setdefault((bnd, sig_tr[i, bnd * rows : (bnd + 1) * rows].tobytes()), []).append(i)
    out = []
    for i in range(len(te_sets)):
        if not te_sets[i]:
            continue
        cands: set[int] = set()
        for bnd in range(bands):
            cands.update(buckets.get((bnd, sig_te[i, bnd * rows : (bnd + 1) * rows].tobytes()), ()))
        best, best_j = 0.0, -1
        for j in cands:
            inter = len(te_sets[i] & tr_sets[j])
            jac = inter / len(te_sets[i] | tr_sets[j])
            if jac > best:
                best, best_j = jac, j
        if best >= threshold:
            out.append((i, best_j, best))
    return out


@dataclass(frozen=True)
class LeakageReport:
    """Overlap between a training and a test set. ``findings`` lists each check with the affected test rows."""

    findings: tuple[dict[str, Any], ...]
    settings: Mapping[str, Any]

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    @property
    def leaked_rows(self) -> list[int]:
        """Test rows affected by any duplicate or shared-identifier finding (sorted)."""
        rows: set[int] = set()
        for f in self.findings:
            if f["kind"] in ("exact duplicate", "near duplicate", "near-duplicate text", "shared identifier"):
                rows.update(f["test_rows"])
        return sorted(rows)

    @property
    def clean(self) -> bool:
        return not any(f["count"] for f in self.findings)

    def summary(self) -> str:
        s = self.settings
        head = f"Leakage report: train n={s['n_train']}, test n={s['n_test']}"
        if self.clean:
            return head + " — no overlap or target leakage found."
        lines = [
            head
            + f" — {len(self.leaked_rows)} test rows ({len(self.leaked_rows) / s['n_test']:.1%}) overlap training"
        ]
        for f in self.findings:
            if f["count"]:
                lines.append(f"  {f['kind']:<22} {f['count']:>6}  {f['detail']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {"findings": list(self.findings), "settings": dict(self.settings), "leaked_rows": self.leaked_rows}
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        rows = ["| Check | Count | Detail |", "| --- | ---: | --- |"]
        rows += [f"| {f['kind']} | {f['count']} | {f['detail']} |" for f in self.findings]
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def leakage_report(
    train: Any,
    test: Any,
    *,
    target: Optional[str] = None,
    id_column: Optional[str] = None,
    text_columns: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
    near_tol: float = 1e-3,
    text_threshold: float = 0.8,
    shingle: int = 3,
    target_auc: float = 0.99,
) -> LeakageReport:
    """Check a train / test split for leakage (see the module docstring). ``columns`` restricts the feature
    columns compared (default: all shared columns except ``target`` and ``id_column``)."""
    tr, te = _table(train, "train"), _table(test, "test")
    n_tr, n_te = len(next(iter(tr.values()))), len(next(iter(te.values())))
    shared = [c for c in tr if c in te]
    feats = [c for c in (columns or shared) if c in shared and c not in (target, id_column)]
    if columns is not None and len(feats) != len([c for c in columns if c not in (target, id_column)]):
        missing = [c for c in columns if c not in shared]
        raise InputValidationError(f"Columns not in both sets: {', '.join(missing[:5])}.")
    texts = list(text_columns or [])
    findings: list[dict[str, Any]] = []
    if feats:
        h_tr = _row_hashes(tr, feats)
        h_te = _row_hashes(te, feats)
        index: dict[str, int] = {}
        for i, h in enumerate(h_tr):
            index.setdefault(h, i)
        exact = [(i, index[h]) for i, h in enumerate(h_te) if h in index]
        within = n_te - len(set(h_te))
        findings.append(
            {
                "kind": "exact duplicate",
                "count": len(exact),
                "detail": f"test rows identical to a training row on {len(feats)} feature(s)",
                "test_rows": [i for i, _ in exact],
                "pairs": exact[:1000],
            }
        )
        findings.append(
            {
                "kind": "duplicates within test",
                "count": within,
                "detail": "repeated rows inside the test set (inflate its effective size)",
                "test_rows": [],
            }
        )
        num = [c for c in feats if c not in texts and _is_numeric(tr[c]) and _is_numeric(te[c])]
        exact_rows = {i for i, _ in exact}
        if num and n_tr and n_te:
            from scipy.spatial import cKDTree

            xtr = np.column_stack([tr[c].astype(float) for c in num])
            xte = np.column_stack([te[c].astype(float) for c in num])
            ok_tr = np.all(np.isfinite(xtr), axis=1)
            ok_te = np.all(np.isfinite(xte), axis=1)
            mu = xtr[ok_tr].mean(axis=0) if ok_tr.any() else np.zeros(len(num))
            sd = xtr[ok_tr].std(axis=0) if ok_tr.any() else np.ones(len(num))
            sd = np.where(sd > 0, sd, 1.0)
            near: list[tuple[int, int, float]] = []
            if ok_tr.any() and ok_te.any():
                tree = cKDTree((xtr[ok_tr] - mu) / sd)
                dist, idx = tree.query((xte[ok_te] - mu) / sd, k=1)
                tr_index = np.flatnonzero(ok_tr)
                for ti, dd, j in zip(np.flatnonzero(ok_te).tolist(), dist, idx):
                    if dd <= near_tol * math.sqrt(len(num)) and ti not in exact_rows:
                        near.append((ti, int(tr_index[j]), float(dd)))
            findings.append(
                {
                    "kind": "near duplicate",
                    "count": len(near),
                    "detail": f"numeric rows within {near_tol:g} standardised units per feature of a training row",
                    "test_rows": [i for i, _, _ in near],
                    "pairs": near[:1000],
                }
            )
    for c in texts:
        if c not in shared:
            raise InputValidationError(f"Text column {c!r} is not in both sets.")
        hits = _near_text(tr[c], te[c], text_threshold, shingle)
        findings.append(
            {
                "kind": "near-duplicate text",
                "count": len(hits),
                "detail": f"{c!r}: Jaccard ≥ {text_threshold:g} on {shingle}-word shingles",
                "test_rows": [i for i, _, _ in hits],
                "pairs": hits[:1000],
            }
        )
    if id_column is not None:
        if id_column not in tr or id_column not in te:
            raise InputValidationError(f"id_column {id_column!r} must be in both sets.")
        ids_tr = {_norm_cell(x) for x in tr[id_column]}
        rows = [i for i, x in enumerate(te[id_column]) if _norm_cell(x) in ids_tr]
        n_ids = len({_norm_cell(te[id_column][i]) for i in rows})
        findings.append(
            {
                "kind": "shared identifier",
                "count": len(rows),
                "detail": f"{n_ids} {id_column!r} value(s) appear in both sets",
                "test_rows": rows,
            }
        )
    if target is not None:
        if target not in tr:
            raise InputValidationError(f"target {target!r} is not a column of train.")
        y = tr[target]
        suspects = []
        labels = np.unique(y[~_missing(y)]) if y.dtype.kind != "O" else np.unique([v for v in y if v is not None])
        if labels.size == 2:
            pos = y == labels.max() if y.dtype.kind != "O" else y == labels[-1]
            for c in feats:
                v = tr[c]
                if not _is_numeric(v):
                    continue
                v = v.astype(float)
                ok = np.isfinite(v)
                if ok.sum() < 10 or np.ptp(v[ok]) == 0:
                    continue
                a = _auc(pos[ok], v[ok])
                if np.isfinite(a) and max(a, 1 - a) >= target_auc:
                    suspects.append((c, max(a, 1 - a)))
        elif labels.size > 2 and _is_numeric(y):
            yt = y.astype(float)
            for c in feats:
                v = tr[c]
                if not _is_numeric(v):
                    continue
                v = v.astype(float)
                ok = np.isfinite(v) & np.isfinite(yt)
                if ok.sum() < 10 or np.ptp(v[ok]) == 0 or np.ptp(yt[ok]) == 0:
                    continue
                r = abs(float(np.corrcoef(v[ok], yt[ok])[0, 1]))
                if r >= target_auc:
                    suspects.append((c, r))
        findings.append(
            {
                "kind": "target leakage",
                "count": len(suspects),
                "detail": ", ".join(f"{c} ({s:.3f})" for c, s in suspects)
                or f"no feature separates {target!r} almost perfectly",
                "test_rows": [],
                "features": [c for c, _ in suspects],
            }
        )
    settings = {"n_train": n_tr, "n_test": n_te, "features": feats, "near_tol": near_tol}
    return LeakageReport(tuple(findings), settings)


# ------------------------------------------------------------------------------------------------ label issues


@dataclass(frozen=True)
class LabelIssues:
    """Examples whose given label looks wrong, ranked by how unlikely the label is under the model."""

    indices: np.ndarray
    given: np.ndarray
    suggested: np.ndarray
    self_confidence: np.ndarray
    classes: tuple[Any, ...]
    noise_rate: float
    confident_joint: np.ndarray
    n: int

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def summary(self) -> str:
        lines = [
            f"Label issues (confident learning): {self.indices.size} of {self.n} examples "
            f"({self.indices.size / self.n:.1%}); estimated label noise {self.noise_rate:.1%}"
        ]
        for i, g, s, c in list(zip(self.indices, self.given, self.suggested, self.self_confidence))[:10]:
            lines.append(f"  row {int(i):<7} labelled {g!s:<10} looks like {s!s:<10} (p(label)={c:.3f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "indices": self.indices.tolist(),
                "given": self.given.tolist(),
                "suggested": self.suggested.tolist(),
                "self_confidence": self.self_confidence.tolist(),
                "classes": list(self.classes),
                "noise_rate": self.noise_rate,
                "confident_joint": self.confident_joint.tolist(),
                "n": self.n,
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        rows = ["| Row | Given | Suggested | p(given label) |", "| ---: | --- | --- | ---: |"]
        rows += [
            f"| {int(i)} | {g} | {s} | {c:.{digits}f} |"
            for i, g, s, c in zip(self.indices, self.given, self.suggested, self.self_confidence)
        ]
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def label_issues(y: Any, pred_probs: Any, *, classes: Any = None) -> LabelIssues:
    """Likely label errors by confident learning (Northcutt et al. 2021). ``pred_probs`` must be out-of-sample
    (e.g. cross-validated) class probabilities, n × k; columns are integer labels, or ``classes=`` gives the label
    of each column. Returned examples are sorted from the most to the least suspicious."""
    from .conformal import _label_index, _probs

    p = _probs(pred_probs, "pred_probs")
    yi, order = _label_index(y, classes, p.shape[1], "y")
    if yi.size != p.shape[0]:
        raise InputValidationError(f"y has {yi.size} labels for {p.shape[0]} probability rows.")
    k = p.shape[1]
    thresholds = np.array([p[yi == j, j].mean() if np.any(yi == j) else np.inf for j in range(k)])
    # as cleanlab: confident in a class when p >= its threshold - 1e-6; with two or more confident classes the
    # guess is the overall argmax; an example whose overall argmax is its given label is never flagged
    above = p >= thresholds[None, :] - 1e-6
    n_conf = above.sum(axis=1)
    has = n_conf > 0
    argmax = np.argmax(p, axis=1)
    pred = np.where(n_conf > 1, argmax, np.argmax(above, axis=1))
    cj = np.zeros((k, k), dtype=np.int64)
    np.add.at(cj, (yi[has], pred[has]), 1)
    flagged = np.flatnonzero(has & (pred != yi) & (argmax != yi))
    sc = p[np.arange(yi.size), yi]
    order_idx = flagged[np.argsort(sc[flagged], kind="stable")]
    noise = float(cj.sum() - np.trace(cj)) / max(1, int(cj.sum()))
    return LabelIssues(
        indices=order_idx,
        given=np.array([order[j] for j in yi[order_idx]], dtype=object),
        suggested=np.array([order[j] for j in pred[order_idx]], dtype=object),
        self_confidence=sc[order_idx],
        classes=order,
        noise_rate=noise,
        confident_joint=cj,
        n=int(yi.size),
    )


# -------------------------------------------------------------------------------------------------- data cards


@dataclass(frozen=True)
class DatasetCard:
    """A summary of a data set for a paper's methods section or a model card."""

    n: int
    columns: tuple[dict[str, Any], ...]
    target: Optional[dict[str, Any]]
    groups: dict[str, dict[str, int]]
    duplicates: int

    def _repr_html_(self) -> str:
        """Rich display in Jupyter and other notebooks."""
        return str(self.to_html())

    def summary(self) -> str:
        lines = [f"Data set: {self.n} rows, {len(self.columns)} columns, {self.duplicates} duplicate rows"]
        if self.target:
            t = self.target
            dist = ", ".join(f"{k}: {v} ({v / self.n:.1%})" for k, v in list(t["counts"].items())[:10])
            lines.append(f"  Target {t['name']!r} ({t['type']}): {dist}")
        for g, counts in self.groups.items():
            lines.append(f"  {g}: " + ", ".join(f"{k} {v}" for k, v in list(counts.items())[:10]))
        for c in self.columns:
            extra = f"mean {c['mean']:.3g}, SD {c['sd']:.3g}" if "mean" in c else f"{c['unique']} distinct"
            lines.append(f"  {c['name']:<20} {c['type']:<11} missing {c['missing']:.1%}  {extra}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def to_dict(self) -> dict[str, Any]:
        from .core.result import _json_safe

        out: dict[str, Any] = _json_safe(
            {
                "n": self.n,
                "columns": list(self.columns),
                "target": self.target,
                "groups": self.groups,
                "duplicates": self.duplicates,
            }
        )
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self, *, digits: int = 3) -> str:
        rows = ["| Column | Type | Missing | Summary |", "| --- | --- | ---: | --- |"]
        for c in self.columns:
            extra = f"mean {c['mean']:.3g}, SD {c['sd']:.3g}" if "mean" in c else f"{c['unique']} distinct"
            rows.append(f"| {c['name']} | {c['type']} | {c['missing']:.1%} | {extra} |")
        if self.target:
            rows += ["", "| Target class | Count | Share |", "| --- | ---: | ---: |"]
            rows += [f"| {k} | {v} | {v / self.n:.1%} |" for k, v in self.target["counts"].items()]
        return "\n".join(rows) + "\n"

    def to_html(self, *, digits: int = 3) -> str:
        from .html_report import _markdown_tables

        return _markdown_tables(self.to_markdown(digits=digits))


def dataset_card(
    data: Any, *, target: Optional[str] = None, groups: Optional[Sequence[str]] = None
) -> DatasetCard:
    """Size, column types, missing values, numeric summaries, class balance of ``target``, counts per subgroup
    (``groups``) and duplicate rows."""
    cols = _table(data, "data")
    n = len(next(iter(cols.values())))
    info = []
    for name, v in cols.items():
        miss = _missing(v)
        row: dict[str, Any] = {"name": name, "missing": float(miss.mean()) if n else 0.0}
        if _is_numeric(v) and v.dtype.kind != "b":
            x = v[~miss].astype(float)
            row["type"] = "numeric"
            row.update(
                mean=float(x.mean()) if x.size else math.nan,
                sd=float(x.std(ddof=1)) if x.size > 1 else math.nan,
                min=float(x.min()) if x.size else math.nan,
                max=float(x.max()) if x.size else math.nan,
            )
        else:
            vals = [str(x) for x, m in zip(v, miss) if not m]
            row["type"] = "boolean" if v.dtype.kind == "b" else "categorical"
            row["unique"] = len(set(vals))
        info.append(row)
    tgt = None
    if target is not None:
        if target not in cols:
            raise InputValidationError(f"target {target!r} is not a column.")
        v = cols[target]
        tvals, tcounts = np.unique([str(x) for x in v], return_counts=True)
        order = np.argsort(-tcounts, kind="stable")
        tgt = {
            "name": target,
            "type": "numeric" if _is_numeric(v) and tvals.size > 20 else "classes",
            "counts": {str(tvals[i]): int(tcounts[i]) for i in order[:50]},
        }
    gtab: dict[str, dict[str, int]] = {}
    for g in groups or ():
        if g not in cols:
            raise InputValidationError(f"Group column {g!r} is not a column.")
        gvals, gcounts = np.unique([str(x) for x in cols[g]], return_counts=True)
        gtab[g] = {str(a): int(b) for a, b in zip(gvals, gcounts)}
    hashes = _row_hashes(cols, list(cols))
    return DatasetCard(n, tuple(info), tgt, gtab, n - len(set(hashes)))
