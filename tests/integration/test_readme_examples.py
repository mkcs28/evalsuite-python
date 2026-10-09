"""Every Python example in README.md runs against the package.

Placeholder lines (code containing ``...``) are skipped; the variables the examples assume are provided by
fixtures matched to the README section the example sits in.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import evalsuite as es

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")

README = Path(__file__).resolve().parents[2] / "README.md"


def _blocks() -> list[tuple[str, str]]:
    text = README.read_text(encoding="utf-8")
    out, heading = [], ""
    for match in re.finditer(r"^(#+ [^\n]*)$|^```python\n(.*?)^```", text, flags=re.M | re.S):
        if match.group(1):
            heading = match.group(1).lstrip("# ").strip()
        else:
            out.append((heading, match.group(2)))
    return out


def _runnable(code: str) -> str:
    lines = []
    for line in code.splitlines():
        if "..." in line.split("#", 1)[0]:
            continue  # placeholder such as [x1, y1, ...] or es.compare(...)
        lines.append(line)
    return "\n".join(lines)


def _namespace(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)

    def prob(strength: float) -> np.ndarray:
        return 1 / (1 + np.exp(-(strength * (y - 0.5) + rng.normal(0, 1, 200))))

    probs = {k: prob(s) for k, s in {"lr": 2.0, "rf": 1.6, "gb": 1.2, "a": 2.0, "b": 1.0}.items()}
    preds = {k: (v >= 0.5).astype(int) for k, v in probs.items()}
    masks = rng.integers(0, 2, (3, 24, 24))
    boxes = [{"boxes": [[0, 0, 10, 10], [20, 20, 40, 40]], "labels": [1, 2]}]
    dets = [{"boxes": [[1, 0, 10, 11], [19, 20, 40, 41]], "labels": [1, 2], "scores": [0.9, 0.8]}]
    coco_gt = {
        "images": [{"id": 1, "width": 50, "height": 50}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}
        ],
        "categories": [{"id": 1, "name": "cell"}],
    }
    (tmp_path / "instances_val.json").write_text(json.dumps(coco_gt))
    (tmp_path / "detections.json").write_text(
        json.dumps([{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}])
    )
    g = [rng.normal(m, 1, 30) for m in (0, 0.3, 0.6)]
    return {
        "es": es,
        "y_true": y,
        "y_pred": preds["lr"],
        "y_prob": probs["lr"],
        "prob": probs["lr"],
        **{f"pred_{k}": v for k, v in preds.items()},
        **{f"prob_{k}": v for k, v in probs.items()},
        "scores_a": g[0],
        "scores_b": g[1],
        "scores_c": g[2],
        "fold_scores_a": g[0],
        "fold_scores_b": g[1],
        "a": g[0],
        "b": g[1],
        "g1": g[0],
        "g2": g[1],
        "g3": g[2],
        "residuals": g[0],
        "table": np.array([[20, 10], [8, 25]]),
        "p_values": [0.01, 0.04, 0.03, 0.2],
        "y_reg": g[0],
        "pred_reg": g[0] + rng.normal(0, 0.2, 30),
        "masks": masks,
        "image": rng.random((24, 24)),
        "boxes_a": [[0, 0, 10, 10]],
        "boxes_b": [[0, 0, 5, 5]],
        "_vision": {"masks": masks, "boxes": boxes, "dets": dets},
    }


EXAMPLES = _blocks()


@pytest.mark.parametrize(("heading", "code"), EXAMPLES, ids=[f"{i}-{h[:30]}" for i, (h, _) in enumerate(EXAMPLES)])
def test_readme_example_runs(heading: str, code: str, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    ns = _namespace(tmp_path)
    vision = ns.pop("_vision")
    if heading.lower().startswith("segmentation"):
        ns["y_true"], ns["y_pred"] = vision["masks"], np.roll(vision["masks"], 1, axis=2)
    if heading.lower().startswith("object detection"):
        ns["y_true"], ns["y_pred"] = vision["boxes"], vision["dets"]
    if heading.lower().startswith("llm evaluation"):
        rng = np.random.default_rng(1)
        refs = ["the cat sat on the mat", "a dog ran in the park", "hello world"] * 4
        preds = ["the cat sat on a mat", "dog ran in park", "hello there world"] * 4
        ns.update(
            references=refs,
            predictions=preds,
            refs=refs,
            preds=preds,
            preds_a=preds,
            preds_b=[p + " today" for p in preds],
            ref_token_embs=[rng.normal(size=(5, 8)) for _ in refs],
            pred_token_embs=[rng.normal(size=(4, 8)) for _ in refs],
            comet_fn=lambda r, p: [
                len(set(a.split()) & set(b.split())) / len(set(a.split())) for a, b in zip(r, p)
            ],
            claim_verdicts=[["supported", "unsupported"], [True, True]],
            comparisons=[("a", "b", "win"), ("b", "c", "win"), ("c", "a", "win"), ("a", "c", "tie")],
            relevant=[{1, 2}, {3}],
            retrieved=[[1, 5, 2], [4, 3]],
            outputs=['{"x": 1}', "not json"],
            schema={"type": "object", "required": ["x"]},
            n_samples=[10, 10],
            n_correct=[3, 0],
        )
    if heading.lower().startswith("classification report") or heading.lower().startswith("plots"):
        ns["y_pred"] = (ns["y_prob"] >= 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(_runnable(code), f"README.md [{heading}]", "exec"), ns)  # noqa: S102


def test_readme_has_examples() -> None:
    assert len(EXAMPLES) >= 8
