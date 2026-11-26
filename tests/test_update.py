import json
import sys
import time

import pytest

import evalsuite as es
from evalsuite import _update
from evalsuite.cli.main import main


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path / "evalsuite" / "update-check.json"


def test_version_ordering():
    k = _update._key
    assert k("1.1.2") > k("1.1.1")
    assert k("1.10.0") > k("1.9.9")
    assert k("2.0.0rc1") < k("2.0.0")
    assert k("2.0.0rc1") > k("1.9.9")


def test_check_for_update_and_cache(cache, monkeypatch):
    monkeypatch.setattr(_update, "_fetch_latest", lambda timeout=3.0: "99.0.0")
    info = es.check_for_update()
    assert info.update_available and info.latest == "99.0.0"
    assert json.loads(cache.read_text())["latest"] == "99.0.0"
    monkeypatch.setattr(_update, "_fetch_latest", lambda timeout=3.0: None)
    assert not es.check_for_update().update_available  # offline: no claim either way


def test_notice_is_printed_at_most_once_a_day(cache, monkeypatch, capsys):
    monkeypatch.setattr(_update, "_disabled", lambda: False)
    monkeypatch.setattr(_update.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: None})())
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"checked": time.time(), "latest": "99.0.0"}))
    _update.notify_on_import()
    assert "99.0.0 is available" in capsys.readouterr().err
    _update.notify_on_import()
    assert capsys.readouterr().err == ""


def test_no_notice_when_current_or_disabled(cache, monkeypatch, capsys):
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"checked": time.time(), "latest": "0.0.1"}))
    monkeypatch.setattr(_update, "_disabled", lambda: False)
    _update.notify_on_import()
    assert capsys.readouterr().err == ""
    cache.write_text(json.dumps({"checked": time.time(), "latest": "99.0.0"}))
    monkeypatch.setattr(_update, "_disabled", lambda: True)
    _update.notify_on_import()
    assert capsys.readouterr().err == ""


def test_upgrade_runs_this_interpreters_pip(monkeypatch):
    calls = []
    import subprocess

    monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert es.upgrade() == 0
    assert calls[0][:5] == [sys.executable, "-m", "pip", "install", "--upgrade"]
    assert calls[0][5] == "evalsuite-python"


def test_cli_upgrade_and_version(cache, monkeypatch, capsys):
    monkeypatch.setattr(_update, "_fetch_latest", lambda timeout=3.0: es.__version__)
    assert main(["upgrade"]) == 0
    assert "latest release" in capsys.readouterr().out
    monkeypatch.setattr(_update, "_fetch_latest", lambda timeout=3.0: "99.0.0")
    called = []
    monkeypatch.setattr(_update, "upgrade", lambda quiet=False: called.append(1) or 0)
    assert main(["upgrade", "--check"]) == 0 and not called
    assert main(["upgrade"]) == 0 and called
    assert main(["version", "--check"]) == 0
    assert "latest: 99.0.0" in capsys.readouterr().out
