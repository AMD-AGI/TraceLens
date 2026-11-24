###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pair a replayed graph's kernels with the launches a model issues, in order.

A launch whose source names its device kernels takes only one of those; a
launch naming none takes one no named launch would claim. A kernel name is
matched to a launched symbol as a whole identifier, demangled or mangled.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - annotations only
    # ``graph_launches.Launch``; read at type-checking time only, as
    # graph_launches itself pairs its launches through this module.
    Launch = Any


def _identifier(symbol: str) -> re.Pattern:
    # Demangled, the name is one identifier; Itanium-mangled (``_ZN5aiter37
    # dynamic_..._kernelI...``), it is its length followed by the name.
    plain = rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])"
    mangled = rf"(?<![0-9]){len(symbol)}{re.escape(symbol)}"
    return re.compile(rf"{plain}|{mangled}")


_SYMBOL_PATTERNS: dict[str, re.Pattern] = {}


def kernel_is(kernel_name: str, symbols: tuple[str, ...]) -> bool:
    """Whether ``kernel_name`` is one of the device functions ``symbols`` names.

    A device symbol carries its function name as one identifier
    (``void ns::name<...>(...)``, a Triton function's own name), so a symbol
    matches only as a whole identifier, never as a substring of a longer one.
    A mangled symbol spells each name as its length and the name.
    """
    for symbol in symbols:
        pattern = _SYMBOL_PATTERNS.get(symbol)
        if pattern is None:
            pattern = _SYMBOL_PATTERNS[symbol] = _identifier(symbol)
        if pattern.search(kernel_name):
            return True
    return False


# Alignment scores. A kernel the launch's source names outweighs any number of
# launches whose source names no kernel, so an anchored kernel is never given
# up to place unanchored launches around it.
_NAMED = 1000
_UNNAMED = 1
_REPEAT = 1
# A launch naming no kernel may run a second one of the same family as its
# first (a library's split reduction: ``Cijk_...`` then ``Cijk_..._PostGSU``).
_UNNAMED_REPEAT = 0.5
# Never more than the difference one pairing makes, summed over a graph.
_AFFINITY = 1e-4
_MANGLED_SCOPE = re.compile(r"^_ZN?(\d+)")


def kernel_family(name: str) -> str:
    """The library a kernel name comes from: its first scope or prefix."""
    name = name.removeprefix("void ").strip()
    mangled = _MANGLED_SCOPE.match(name)
    if mangled:
        start = mangled.end()
        return name[start : start + int(mangled.group(1))]
    return re.split(r"[_:<(\s]", name, maxsplit=1)[0]


def align_kernels(
    kernel_names: list[str],
    launches: list[Launch],
    op_families: dict[str, set[str]] | None = None,
) -> list[Launch | None]:
    """Pair replayed kernels with predicted launches (see :func:`_align_kernels`).

    A trace replays the same graph many times: each distinct pairing is
    worked out once.
    """
    key = (
        tuple(kernel_names),
        tuple(launches),
        frozenset(
            (op, frozenset(families)) for op, families in (op_families or {}).items()
        ),
    )
    found = _ALIGNED.get(key)
    if found is None:
        if len(_ALIGNED) >= _ALIGNED_KEPT:
            _ALIGNED.pop(next(iter(_ALIGNED)))
        found = _ALIGNED[key] = tuple(
            _align_kernels(kernel_names, launches, op_families)
        )
    return list(found)


# Pairings worked out, by their inputs; the oldest go first past a bound.
_ALIGNED: dict = {}
_ALIGNED_KEPT = 256


def _align_kernels(
    kernel_names: list[str],
    launches: list[Launch],
    op_families: dict[str, set[str]] | None = None,
) -> list[Launch | None]:
    """Pair replayed kernels with predicted launches, both in launch order.

    A launch whose source names its kernels takes only a kernel that is one
    of them, and may take several in a row (a split reduction, a second
    pass). A launch whose source names none takes one kernel that no named
    launch would claim, and a further one of the same family. Where the
    trace shows an op launching kernels outside any graph, ``op_families``
    gives the families it launched, and its launches take only those.
    Launches and kernels may both go unpaired; the pairing kept is the one
    that pairs the most, named kernels first.
    """
    count_k = len(kernel_names)
    symbol_sets = {launch.kernels for launch in launches if launch.kernels}
    distinct = set(kernel_names)
    families = [kernel_family(name) for name in kernel_names]
    is_symbol = {
        (name, symbols): kernel_is(name, symbols)
        for name in distinct
        for symbols in symbol_sets
    }
    # Symbols the replay never ran name a branch it did not take, or miss the
    # kernel a library launches indirectly: such a launch pairs as unnamed.
    ran = {
        symbols
        for symbols in symbol_sets
        if any(is_symbol[(name, symbols)] for name in distinct)
    }
    launches = [
        launch if not launch.kernels or launch.kernels in ran else _unnamed(launch)
        for launch in launches
    ]
    count_l = len(launches)
    symbol_sets = ran
    claimed = {
        name: any(is_symbol[(name, symbols)] for symbols in symbol_sets)
        for name in distinct
    }

    family_of = {name: kernel_family(name) for name in distinct}
    op_families = op_families or {}

    def pairs(launch: Launch, name: str) -> bool:
        if launch.kernels:
            return is_symbol[(name, launch.kernels)]
        if claimed[name]:
            return False
        seen = op_families.get(launch.op_name)
        return seen is None or family_of[name] in seen

    lowered = {name: name.lower() for name in distinct}
    op_words: dict[str, list[str]] = {}

    def affinity(launch: Launch, name: str) -> float:
        """A tie-breaker: an unnamed launch's op word in the kernel's name."""
        if launch.kernels:
            return 0.0
        words = op_words.get(launch.op_name)
        if words is None:
            base = launch.op_name.rsplit("::", 1)[-1].rsplit(".", 1)[-1].lower()
            words = op_words[launch.op_name] = [
                word for word in re.split(r"[^a-z0-9]+|_", base) if len(word) >= 3
            ]
        return _AFFINITY if any(word in lowered[name] for word in words) else 0.0

    neg = float("-inf")
    width = count_k + 1
    # Back-pointers, one byte per cell: bits 0-1 the move out of ``best``
    # (0 = pair, 1 = skip launch, 2 = skip kernel), bit 2 set when the pair
    # extends the launch's previous kernel.
    moves = bytearray((count_l + 1) * width)
    best_prev = [0.0] * width
    for j in range(1, width):
        moves[j] = 2
    for i in range(1, count_l + 1):
        launch = launches[i - 1]
        gain = _NAMED if launch.kernels else _UNNAMED
        best_row = [0.0] * width
        open_row = [neg] * width
        best_row[0] = best_prev[0]
        moves[i * width] = 1
        for j in range(1, width):
            cell = i * width + j
            flags = 0
            if pairs(launch, kernel_names[j - 1]):
                first = best_prev[j - 1] + gain + affinity(launch, kernel_names[j - 1])
                if launch.kernels:
                    again = open_row[j - 1] + _REPEAT
                elif j > 1 and families[j - 1] == families[j - 2]:
                    again = open_row[j - 1] + _UNNAMED_REPEAT
                else:
                    again = neg
                if again > first:
                    open_row[j] = again
                    flags = 4
                else:
                    open_row[j] = first
            # On a tie, leave the later kernel unpaired: walking back from the
            # end, that pairs each launch with the earliest kernel it fits.
            choice, move = best_row[j - 1], 2
            if open_row[j] > choice:
                choice, move = open_row[j], 0
            if best_prev[j] > choice:
                choice, move = best_prev[j], 1
            best_row[j] = choice
            moves[cell] = move | flags
        best_prev = best_row
    assigned: list[Launch | None] = [None] * count_k
    i, j = count_l, count_k
    in_pair = False
    while i > 0 and j > 0:
        cell = moves[i * width + j]
        if not in_pair:
            move = cell & 3
            if move == 1:
                i -= 1
                continue
            if move == 2:
                j -= 1
                continue
        assigned[j - 1] = launches[i - 1]
        if cell & 4:
            in_pair = True
            j -= 1
        else:
            in_pair = False
            i, j = i - 1, j - 1
    return assigned


def _unnamed(launch: Launch) -> Launch:
    from dataclasses import replace

    return replace(launch, kernels=())
