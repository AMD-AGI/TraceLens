###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The parsed-trace cache is immutable, and tests of one file share an xdist group."""

import gzip
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from TraceLens.parsed_trace_cache import (
    PARSE_CACHE_MAX_SIZE,
    _cache,
    _enabled,
    cache_enabled,
    freeze,
    get_frozen,
)
from TraceLens.util import DataLoader
from conftest import assign_trace_groups, trace_group_id, trace_path_from_params


def _write_gz(path, payload):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _count_gzip_open(fn):
    calls = {"n": 0}
    real_open = gzip.open

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_open(*args, **kwargs)

    with patch("gzip.open", counting):
        fn()
    return calls["n"]


class _Item:
    def __init__(self, params, markers=()):
        self.callspec = SimpleNamespace(params=params)
        self._markers = list(markers)
        self.added = []

    def get_closest_marker(self, name):
        for marker in self._markers:
            if getattr(marker, "name", None) == name:
                return marker
        return None

    def add_marker(self, marker):
        self.added.append(marker)


def test_parsed_trace_cache_is_enabled_under_pytest():
    assert cache_enabled()


def test_frozen_parse_rejects_inplace_writes():
    frozen = freeze(
        {"traceEvents": [{"name": "kernel", "args": {"x": 1}}], "meta": {"rank": 0}}
    )
    with pytest.raises(TypeError, match="immutable"):
        frozen["meta"] = {}
    with pytest.raises(TypeError, match="immutable"):
        frozen["traceEvents"][0]["name"] = "mutated"
    with pytest.raises(TypeError, match="immutable"):
        frozen["traceEvents"].append({"name": "extra"})
    with pytest.raises(TypeError, match="immutable"):
        frozen.update({"extra": 1})
    with pytest.raises(TypeError, match="immutable"):
        frozen["traceEvents"].pop()
    with pytest.raises(TypeError, match="immutable"):
        frozen.clear()


def test_loads_are_isolated_and_do_not_reread(tmp_path):
    payload = {"traceEvents": [{"name": "kernel", "args": {"x": 1}}]}
    path = tmp_path / "model.json.gz"
    _write_gz(path, payload)
    path_str = str(path)

    def load_twice():
        first = DataLoader.load_data(path_str)
        second = DataLoader.load_data(path_str)
        return first, second

    reads = _count_gzip_open(lambda: load_twice())
    # The counter wraps both loads; repeat them so the assertions see the objects.
    first, second = load_twice()
    assert reads == 1
    assert first == payload
    assert second == payload
    assert first is not second
    assert first["traceEvents"] is not second["traceEvents"]

    first["traceEvents"][0]["name"] = "mutated"
    first["traceEvents"].append({"name": "extra"})
    third = DataLoader.load_data(path_str)
    assert third == payload
    frozen = get_frozen(path_str)
    assert frozen["traceEvents"][0]["name"] == "kernel"
    with pytest.raises(TypeError, match="immutable"):
        frozen["traceEvents"][0]["args"]["x"] = 2


def test_rewritten_file_is_a_cache_miss(tmp_path):
    path = tmp_path / "model.json.gz"
    _write_gz(path, {"traceEvents": [{"name": "first"}]})
    path_str = str(path)
    DataLoader.load_data(path_str)

    _write_gz(path, {"traceEvents": [{"name": "second"}]})
    stat = path.stat()
    # Keep the key distinct even if the rewrite lands in the same nanosecond.
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))

    loaded = {}

    def _load():
        loaded["value"] = DataLoader.load_data(path_str)

    reads = _count_gzip_open(_load)
    assert reads == 1
    assert loaded["value"] == {"traceEvents": [{"name": "second"}]}


def test_cache_disabled_reads_every_time(tmp_path):
    path = tmp_path / "model.json.gz"
    _write_gz(path, {"traceEvents": []})
    path_str = str(path)
    token = _enabled.set(False)
    try:
        assert not cache_enabled()
        reads = _count_gzip_open(
            lambda: (DataLoader.load_data(path_str), DataLoader.load_data(path_str))
        )
        assert reads == 2
    finally:
        _enabled.reset(token)
    assert cache_enabled()


def test_cache_evicts_least_recently_used(tmp_path):
    saved = list(_cache.items())
    _cache.clear()
    try:
        paths = []
        for index in range(PARSE_CACHE_MAX_SIZE + 1):
            path = tmp_path / f"trace_{index}.json.gz"
            _write_gz(path, {"traceEvents": [{"name": str(index)}]})
            paths.append(str(path))
            DataLoader.load_data(paths[-1])
        assert len(_cache) == PARSE_CACHE_MAX_SIZE
        with pytest.raises(KeyError):
            get_frozen(paths[0])
        assert get_frozen(paths[-1])["traceEvents"][0]["name"] == str(
            PARSE_CACHE_MAX_SIZE
        )
    finally:
        _cache.clear()
        _cache.update(saved)


def test_same_trace_file_shares_xdist_group(tmp_path):
    path = tmp_path / "model.json.gz"
    path.write_bytes(b"")
    by_path = _Item({"trace_path": str(path)})
    by_dir_gz = _Item({"dirpath": str(tmp_path), "gz": "model.json.gz"})
    by_dir_trace_gz = _Item({"dirpath": str(tmp_path), "trace_gz": "model.json.gz"})
    other = _Item({"trace_path": str(tmp_path / "other.json.gz")})
    unmarked = _Item({"report_name": "not-a-trace"})
    already = _Item(
        {"trace_path": str(path)}, markers=[SimpleNamespace(name="xdist_group")]
    )

    assign_trace_groups([by_path, by_dir_gz, by_dir_trace_gz, other, unmarked, already])

    group = trace_group_id(str(path))
    assert by_path.added[0].args == (group,)
    assert by_path.added[0].mark.name == "xdist_group"
    assert by_dir_gz.added[0].args == (group,)
    assert by_dir_trace_gz.added[0].args == (group,)
    assert other.added[0].args == (trace_group_id(str(tmp_path / "other.json.gz")),)
    assert other.added[0].args != (group,)
    assert unmarked.added == []
    assert already.added == []
    assert trace_path_from_params({"trace_path": str(path)}) == str(path)
    assert trace_path_from_params(
        {"dirpath": str(tmp_path), "gz": "model.json.gz"}
    ) == str(path)
