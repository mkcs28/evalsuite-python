"""Framework integrations: real scikit-learn, and stand-in modules for the optional frameworks."""

import sys
import types

import numpy as np
import pytest

import evalsuite as es
from evalsuite import integrations as integ


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    return x, y


def test_sklearn_scorer_matches_evalsuite():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    x, y = data()
    m = LogisticRegression().fit(x, y)
    assert integ.sklearn_scorer("mcc")(m, x, y) == pytest.approx(float(es.mcc(y, m.predict(x))))
    assert integ.sklearn_scorer("roc_auc")(m, x, y) == pytest.approx(
        float(es.roc_auc(y, y_prob=m.predict_proba(x)[:, 1]))
    )
    # lower is better: negated, as scikit-learn expects
    assert integ.sklearn_scorer("log_loss")(m, x, y) == pytest.approx(
        -float(es.log_loss(y, y_prob=m.predict_proba(x)[:, 1]))
    )
    scores = cross_val_score(LogisticRegression(), x, y, cv=3, scoring=integ.sklearn_scorer("f1"))
    assert scores.shape == (3,) and np.all(scores > 0.5)


def test_hf_compute_metrics():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 100)
    logits = np.stack([-labels + rng.normal(0, 0.8, 100), labels + rng.normal(0, 0.8, 100)], axis=1)
    out = integ.hf_compute_metrics(["accuracy", "roc_auc"], ci="analytic")(
        types.SimpleNamespace(predictions=logits, label_ids=labels)
    )
    assert out["accuracy"] == pytest.approx(float(es.accuracy(labels, logits.argmax(1))))
    assert out["accuracy_ci_low"] < out["accuracy"] < out["accuracy_ci_high"]
    assert "roc_auc_ci_low" in out
    three = integ.hf_compute_metrics(["accuracy"])((rng.normal(size=(30, 3)), rng.integers(0, 3, 30)))
    assert set(three) == {"accuracy"}


def test_trackers_with_stand_in_modules(monkeypatch):
    logged = {}
    mlflow = types.ModuleType("mlflow")
    mlflow.log_metrics = lambda d, step=None: logged.setdefault("mlflow", d)
    mlflow.log_dict = lambda d, name: logged.setdefault("artifact", (name, d))
    wandb = types.ModuleType("wandb")
    wandb.Table = lambda columns, data: ("table", columns, data)
    wandb.log = lambda d: logged.setdefault("wandb", d)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    _, y = data(80)
    r = es.evaluate(y, y, baseline_pred=np.roll(y, 1))
    integ.log_mlflow(r, prefix="test/")
    assert logged["mlflow"]["test/accuracy"] == 1.0 and "test/accuracy_vs_baseline" in logged["mlflow"]
    assert logged["artifact"][0] == "evalsuite.json"
    integ.log_wandb(r)
    assert logged["wandb"]["evalsuite"][0] == "table" and "accuracy_ci_low" in logged["wandb"]


def test_callbacks_with_stand_in_frameworks(monkeypatch):
    keras = types.ModuleType("keras")
    keras.callbacks = types.SimpleNamespace(Callback=type("Callback", (), {}))
    monkeypatch.setitem(sys.modules, "keras", keras)
    x, y = data(60)

    class Model:
        def predict(self, x, verbose=0):
            return 1 / (1 + np.exp(-x[:, :1] * 3))

    cb = integ.keras_callback(x, y, metrics=["accuracy", "roc_auc"])
    cb.model = Model()
    logs = {}
    cb.on_epoch_end(0, logs)
    assert "val_accuracy" in logs and "val_roc_auc_ci_high" in logs

    pl = types.ModuleType("lightning.pytorch")
    pl.Callback = type("Callback", (), {})
    monkeypatch.setitem(sys.modules, "lightning", types.ModuleType("lightning"))
    monkeypatch.setitem(sys.modules, "lightning.pytorch", pl)
    seen = {}
    module = types.SimpleNamespace(log_dict=lambda d: seen.update(d))
    lc = integ.lightning_callback(lambda m: (y, (x[:, 0] > 0).astype(int), None), metrics=["accuracy"])
    lc.on_validation_epoch_end(None, module)
    assert "val_accuracy" in seen and lc.last_result["accuracy"] == seen["val_accuracy"]


def test_missing_framework_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(es.OptionalDependencyError):
        integ.log_mlflow(es.evaluate([0, 1], [0, 1]))
