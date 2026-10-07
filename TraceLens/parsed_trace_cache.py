###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Process-local cache of immutable parsed traces.

Enabled only under pytest. Each checkout returns a plain mutable copy so
callers can build and annotate a tree without changing the cached parse.
"""

from __future__ import annotations

import contextvars
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable

PARSE_CACHE_MAX_SIZE = 8

_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tracelens_parse_cache_enabled", default=False
)
_cache: OrderedDict[tuple, Any] = OrderedDict()


def enable_parse_cache():
    """Turn the cache on for the current context. Returns a reset token."""
    return _enabled.set(True)


def reset_parse_cache(token) -> None:
    _enabled.reset(token)


def cache_enabled() -> bool:
    return _enabled.get()


def clear_parse_cache() -> None:
    _cache.clear()


class _Immutable:
    def _reject(self, *_args, **_kwargs):
        raise TypeError("parsed trace is immutable")


class FrozenDict(_Immutable, dict):
    __setitem__ = _Immutable._reject
    __delitem__ = _Immutable._reject
    clear = _Immutable._reject
    pop = _Immutable._reject
    popitem = _Immutable._reject
    setdefault = _Immutable._reject
    update = _Immutable._reject

    def __deepcopy__(self, memo):
        copied = {}
        memo[id(self)] = copied
        for key, value in self.items():
            copied[deepcopy(key, memo)] = deepcopy(value, memo)
        return copied


class FrozenList(_Immutable, list):
    __setitem__ = _Immutable._reject
    __delitem__ = _Immutable._reject
    append = _Immutable._reject
    clear = _Immutable._reject
    extend = _Immutable._reject
    insert = _Immutable._reject
    pop = _Immutable._reject
    remove = _Immutable._reject
    reverse = _Immutable._reject
    sort = _Immutable._reject

    def __deepcopy__(self, memo):
        copied = []
        memo[id(self)] = copied
        copied.extend(deepcopy(value, memo) for value in self)
        return copied


def freeze(obj: Any) -> Any:
    """Recursively wrap dicts and lists so in-place writes raise."""
    if isinstance(obj, dict) and not isinstance(obj, FrozenDict):
        return FrozenDict((key, freeze(value)) for key, value in obj.items())
    if isinstance(obj, list) and not isinstance(obj, FrozenList):
        return FrozenList(freeze(value) for value in obj)
    if isinstance(obj, tuple):
        return tuple(freeze(value) for value in obj)
    return obj


def thaw(obj: Any) -> Any:
    """Return a plain mutable structure. The frozen cache object stays put."""
    return deepcopy(obj)


def cache_key(filename_path: str) -> tuple:
    stat = os.stat(filename_path)
    return (os.path.realpath(filename_path), stat.st_size, stat.st_mtime_ns)


def get_frozen(filename_path: str) -> Any:
    """Return the cached frozen parse for *filename_path* (tests and diagnostics)."""
    return _cache[cache_key(filename_path)]


def load_cached(filename_path: str, parse: Callable[[], Any]) -> Any:
    """Return a thawed parse, reading the file only on a cache miss.

    When the cache is disabled, or the path is not a file, ``parse`` runs
    every time and nothing is stored.
    """
    if not cache_enabled() or not os.path.isfile(filename_path):
        return parse()

    key = cache_key(filename_path)
    frozen = _cache.get(key)
    if frozen is None:
        frozen = freeze(parse())
        _cache[key] = frozen
        while len(_cache) > PARSE_CACHE_MAX_SIZE:
            _cache.popitem(last=False)
    _cache.move_to_end(key)
    return thaw(frozen)
