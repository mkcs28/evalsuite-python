"""1.4.0: the cache folder, clearing it by hand and automatically."""

import os
import time

import pytest

import evalsuite as es
from evalsuite import cache


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALSUITE_CACHE", str(tmp_path / "c"))
    for var in ("EVALSUITE_CACHE_AUTO_CLEAN", "EVALSUITE_CACHE_MAX_AGE", "EVALSUITE_CACHE_MAX_SIZE"):
        monkeypatch.delenv(var, raising=False)
    return cache.cache_dir(create=True)


def _entry(root, name, size, days_ago=0.0):
    path = root / "datasets" / name
    (path / "sub").mkdir(parents=True)
    f = path / "sub" / "data.arrow"
    f.write_bytes(b"x" * size)
    t = time.time() - days_ago * 86400
    os.utime(f, (t, t))
    return path


def test_location_and_marker(root, monkeypatch, tmp_path):
    assert root == tmp_path / "c" and (root / cache.MARKER).is_file()
    monkeypatch.delenv("EVALSUITE_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache.cache_dir() == tmp_path / "xdg" / "evalsuite"


def test_info_and_clear(root):
    _entry(root, "old", 5000, days_ago=200)
    _entry(root, "new", 3000)
    info = es.cache_info()
    assert info.size_bytes == 8000 and len(info.entries) == 2 and "datasets/old" in info.summary()
    dry = es.clear_cache(older_than="30d", dry_run=True)
    assert dry["removed"] == ["datasets/old"] and (root / "datasets" / "old").exists()
    done = es.clear_cache(older_than="30d")
    assert done["freed_bytes"] == 5000 and not (root / "datasets" / "old").exists()
    assert es.clear_cache(max_size="1KB")["removed"] == ["datasets/new"]
    assert es.cache_info().size_bytes == 0 and not list(root.glob(".trash-*"))
    assert (root / cache.MARKER).is_file()  # the marker survives


def test_size_limit_removes_least_recently_used_first(root):
    _entry(root, "a", 4000, days_ago=3)
    _entry(root, "b", 4000, days_ago=2)
    _entry(root, "c", 4000, days_ago=1)
    assert es.clear_cache(max_size="9000B")["removed"] == ["datasets/a"]


def test_automatic_clearing(root, monkeypatch):
    _entry(root, "stale", 100, days_ago=120)
    _entry(root, "fresh", 100)
    out = cache.auto_clean()
    assert out["removed"] == ["datasets/stale"]
    assert cache.auto_clean() is None  # at most once a day
    _entry(root, "stale2", 100, days_ago=120)
    monkeypatch.setenv("EVALSUITE_CACHE_AUTO_CLEAN", "0")
    assert cache.auto_clean() is None and (root / "datasets" / "stale2").exists()
    monkeypatch.setenv("EVALSUITE_CACHE_AUTO_CLEAN", "1")
    monkeypatch.setenv("EVALSUITE_CACHE_MAX_AGE", "400d")
    assert cache.auto_clean(force=True)["removed"] == []


def test_only_folders_evalsuite_created_are_touched(tmp_path, monkeypatch):
    foreign = tmp_path / "someone-elses"
    foreign.mkdir()
    (foreign / "precious.txt").write_text("keep")
    monkeypatch.setenv("EVALSUITE_CACHE", str(foreign))
    with pytest.raises(es.InputValidationError, match="not created by EvalSuite"):
        es.clear_cache()
    assert cache.auto_clean(force=True) is None and (foreign / "precious.txt").exists()


def test_parsing_and_cli(root, capsys, monkeypatch):
    assert cache._parse_age("2w") == 14 * 86400 and cache._parse_age(5) == 5.0
    assert cache._parse_size("1.5GB") == int(1.5 * 1024**3) and cache._parse_size(10) == 10
    with pytest.raises(es.InputValidationError):
        cache._parse_age("soon")
    with pytest.raises(es.InputValidationError):
        cache._parse_size("big")
    from evalsuite.cli.main import main

    _entry(root, "d", 2048)
    assert main(["cache", "info"]) == 0 and "datasets/d" in capsys.readouterr().out
    assert main(["cache", "clear", "--dry-run"]) == 0 and "Would free" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["cache", "clear"]) == 1 and (root / "datasets" / "d").exists()
    assert main(["cache", "clear", "-y"]) == 0 and "Freed 2.0 KB (1 entry)" in capsys.readouterr().out
    assert main(["cache", "clear"]) == 0 and "already empty" in capsys.readouterr().out
