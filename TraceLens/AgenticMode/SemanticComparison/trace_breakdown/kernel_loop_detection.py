###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Operation:
    name: str
    ts: float
    dur: float


@dataclass(slots=True)
class GPUOperation(Operation):
    external_id: Optional[int] = None
    correlation: Optional[int] = None
    cat: str = ""
    stream: Optional[int] = None

    @property
    def end_ts(self) -> float:
        return self.ts + self.dur


# ---------------------------------------------------------------------------
# Patterns – shared pattern vocabulary
# ---------------------------------------------------------------------------


class Patterns:
    __slots__ = ("patterns", "pattern_operation_names", "operation_name_to_idx")

    def __init__(
        self,
        patterns: Optional[List[List[int]]] = None,
        pattern_operation_names: Optional[Dict[int, str]] = None,
        operation_name_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.patterns: List[List[int]] = patterns if patterns is not None else []
        self.pattern_operation_names: Dict[int, str] = (
            pattern_operation_names if pattern_operation_names is not None else {}
        )
        self.operation_name_to_idx: Dict[str, int] = (
            operation_name_to_idx if operation_name_to_idx is not None else {}
        )
        if self.pattern_operation_names and not self.operation_name_to_idx:
            self.operation_name_to_idx = {
                name: idx for idx, name in self.pattern_operation_names.items()
            }
        elif self.operation_name_to_idx and not self.pattern_operation_names:
            self.pattern_operation_names = {
                idx: name for name, idx in self.operation_name_to_idx.items()
            }

    def get_pattern_name(self, pattern_idx: int) -> str:
        if pattern_idx < 0 or pattern_idx >= len(self.patterns):
            return "?"
        if pattern_idx < 26:
            return chr(ord("A") + pattern_idx)
        offset = pattern_idx - 26
        letter = chr(ord("A") + (offset % 26))
        suffix = (offset // 26) + 1
        return f"{letter}{suffix}"

    def _validate_pattern_indices_mapped(self) -> None:
        if not self.patterns:
            return
        for op_name, op_idx in self.operation_name_to_idx.items():
            if op_idx not in self.pattern_operation_names:
                self.pattern_operation_names[op_idx] = op_name
        used_indices = {op_idx for pattern in self.patterns for op_idx in pattern}
        missing = sorted(
            idx for idx in used_indices if idx not in self.pattern_operation_names
        )
        if missing:
            raise ValueError(
                "All pattern indices must map to operation names. "
                f"Missing mappings for indices: {missing}"
            )


# ---------------------------------------------------------------------------
# LoopStructures – per-stream detection results
# ---------------------------------------------------------------------------


class LoopStructures:

    def __init__(
        self,
        operations: List[Operation],
        _patterns_obj: Optional[Patterns] = None,
        stream_id: Optional[int] = None,
    ):
        self.operations = operations
        self.stream_id = stream_id
        self.sequences: List[List[List[int]]] = []
        self._patterns_obj: Patterns

        if _patterns_obj is not None:
            self._patterns_obj = _patterns_obj
        else:
            self._patterns_obj = Patterns()
            if operations:
                unique_names: List[str] = []
                seen: set = set()
                for op in operations:
                    if op.name not in seen:
                        unique_names.append(op.name)
                        seen.add(op.name)
                self._patterns_obj.pattern_operation_names = {
                    i: name for i, name in enumerate(unique_names)
                }
                self._patterns_obj.operation_name_to_idx = {
                    name: idx
                    for idx, name in self._patterns_obj.pattern_operation_names.items()
                }

        if self.operations:
            for op in self.operations:
                if op.name not in self._patterns_obj.operation_name_to_idx:
                    new_idx = len(self._patterns_obj.operation_name_to_idx)
                    self._patterns_obj.operation_name_to_idx[op.name] = new_idx
                    self._patterns_obj.pattern_operation_names[new_idx] = op.name

        self._patterns_obj._validate_pattern_indices_mapped()

    # -- Property wrappers ---------------------------------------------------

    @property
    def patterns(self) -> List[List[int]]:
        return self._patterns_obj.patterns

    @property
    def pattern_operation_names(self) -> Dict[int, str]:
        return self._patterns_obj.pattern_operation_names

    @property
    def operation_name_to_idx(self) -> Dict[str, int]:
        return self._patterns_obj.operation_name_to_idx

    def _get_pattern_name(self, pattern_idx: int) -> str:
        return self._patterns_obj.get_pattern_name(pattern_idx)

    def get_pattern_name(self, pattern_idx: int) -> str:
        return self._patterns_obj.get_pattern_name(pattern_idx)

    # -- Coverage helpers ----------------------------------------------------

    def get_covered_indices(
        self,
        pattern_indices: Optional[Union[int, List[int]]] = None,
    ) -> set:
        covered: set = set()
        if pattern_indices is None:
            patterns_to_include = range(len(self.patterns))
        elif isinstance(pattern_indices, int):
            patterns_to_include = [pattern_indices]
        else:
            patterns_to_include = pattern_indices
        for idx in patterns_to_include:
            if idx < len(self.sequences):
                for seq in self.sequences[idx]:
                    covered.update(seq)
        return covered

    def _get_pattern_coverage(self, pattern_idx: Optional[int] = None) -> float:
        if not self.operations:
            return 0.0
        covered_indices = self.get_covered_indices(pattern_idx)
        return len(covered_indices) / len(self.operations)

    def get_pattern_coverage(self) -> float:
        return self._get_pattern_coverage()


# ===========================================================================
# Algorithm – iterative pattern finding with extension, splitting, rotation
# ===========================================================================


def get_operation_frequencies(loop_structures: LoopStructures) -> Counter:
    covered = loop_structures.get_covered_indices()
    unused_op_indices = [
        loop_structures.operation_name_to_idx[op.name]
        for i, op in enumerate(loop_structures.operations)
        if i not in covered
    ]
    return Counter(unused_op_indices)


# ---------------------------------------------------------------------------
# Pattern extension
# ---------------------------------------------------------------------------


def extend_pattern(
    loop_structures: LoopStructures,
    pattern_idx: int,
    min_loop_count: int,
    min_pattern_length: int,
    snippets: bool = False,
) -> int:
    covered = loop_structures.get_covered_indices(
        [i for i in range(len(loop_structures.patterns)) if i != pattern_idx]
    )
    best_agree = 0

    while True:
        sequences = loop_structures.sequences[pattern_idx]
        names_before = [
            (
                loop_structures.operations[seq[0] - 1].name
                if not (seq[0] == 0 or seq[0] - 1 in covered)
                else None
            )
            for i, seq in enumerate(sequences)
        ]
        names_after = [
            (
                loop_structures.operations[seq[-1] + 1].name
                if not (
                    seq[-1] == len(loop_structures.operations) - 1
                    or seq[-1] + 1 in covered
                )
                else None
            )
            for i, seq in enumerate(sequences)
        ]

        extendable_before = sum(1 for name in names_before if name is not None)
        extendable_after = sum(1 for name in names_after if name is not None)
        if extendable_before < min_loop_count and extendable_after < min_loop_count:
            best_agree = max(extendable_before, extendable_after)
            break

        name_counts_before = Counter(
            n for n in names_before if n is not None
        ).most_common(1)
        name_counts_after = Counter(
            n for n in names_after if n is not None
        ).most_common(1)

        kernel_before, agree_before = (
            name_counts_before[0] if name_counts_before else (None, 0)
        )
        kernel_after, agree_after = (
            name_counts_after[0] if name_counts_after else (None, 0)
        )
        best_agree = max(agree_before, agree_after)

        if agree_before < min_loop_count and agree_after < min_loop_count:
            break

        current_seq_count = len(sequences)
        if (
            snippets
            and len(loop_structures.patterns[pattern_idx]) >= min_pattern_length
            and best_agree < current_seq_count
        ):
            break

        extend_before = agree_before >= agree_after and agree_before >= min_loop_count
        kernel_name = kernel_before if extend_before else kernel_after
        op_idx = loop_structures.operation_name_to_idx[kernel_name]

        if extend_before:
            loop_structures.patterns[pattern_idx].insert(0, op_idx)
        else:
            loop_structures.patterns[pattern_idx].append(op_idx)

        new_sequences = []
        for i, seq in enumerate(sequences):
            if extend_before:
                if names_before[i] == kernel_name:
                    new_start = seq[0] - 1
                    if new_sequences and new_sequences[-1][-1] >= new_start:
                        continue
                    seq.insert(0, new_start)
                    new_sequences.append(seq)
            else:
                if names_after[i] == kernel_name:
                    if new_sequences and new_sequences[-1][-1] >= seq[0]:
                        continue
                    seq.append(seq[-1] + 1)
                    new_sequences.append(seq)

        loop_structures.sequences[pattern_idx] = new_sequences

        if len(new_sequences) >= min_pattern_length and any(
            new_sequences[i - 1][-1] + 1 == new_sequences[i][0]
            and new_sequences[i][-1] + 1 == new_sequences[i + 1][0]
            for i in range(1, len(new_sequences) - 1)
        ):
            break

    return best_agree


# ---------------------------------------------------------------------------
# Pattern splitting
# ---------------------------------------------------------------------------


def _analyze_pattern_for_splits(
    loop_structures: LoopStructures,
    pattern_idx: int,
    min_pattern_length: int,
    split_pattern_length_limit: int,
) -> List[List[int]]:
    pattern = loop_structures.patterns[pattern_idx]
    if len(pattern) <= split_pattern_length_limit:
        return []
    if not loop_structures.sequences[pattern_idx]:
        return []

    mini_operations = [
        GPUOperation(
            name=loop_structures.pattern_operation_names[op_idx],
            ts=0.0,
            dur=0.0,
        )
        for op_idx in pattern
    ]

    mini_ls = find_kernel_loops_single_stream(
        mini_operations,
        min_loop_count=2,
        min_pattern_length=min_pattern_length,
        split_pattern_length_limit=split_pattern_length_limit,
    )

    if not mini_ls.patterns:
        return []

    sub_patterns: List[List[int]] = []
    for sp_idx, mini_seqs in enumerate(mini_ls.sequences):
        all_consecutive = len(mini_seqs) >= 2 and all(
            mini_seqs[i][0] == mini_seqs[i - 1][-1] + 1
            for i in range(1, len(mini_seqs))
        )
        if not all_consecutive:
            continue
        global_pat = [
            loop_structures.operation_name_to_idx[mini_ls.pattern_operation_names[mi]]
            for mi in mini_ls.patterns[sp_idx]
        ]
        sub_patterns.append(global_pat)

    return sub_patterns


def _apply_pattern_split(
    loop_structures: LoopStructures,
    pattern_idx: int,
    sub_patterns: List[List[int]],
) -> None:
    loop_structures.patterns.pop(pattern_idx)
    loop_structures.sequences.pop(pattern_idx)

    affected_indices: Set[int] = set()
    for _sp_idx, pat in enumerate(sub_patterns):
        existing_idx = next(
            (i for i, p in enumerate(loop_structures.patterns) if p == pat),
            None,
        )
        if existing_idx is not None:
            affected_indices.add(existing_idx)
        else:
            loop_structures.patterns.append(pat)
            loop_structures.sequences.append([])
            affected_indices.add(len(loop_structures.patterns) - 1)

    for pidx in sorted(affected_indices):
        all_seqs = find_pattern_occurrences(loop_structures, pidx)
        loop_structures.sequences[pidx] = all_seqs


def split_long_pattern(
    loop_structures: LoopStructures,
    pattern_idx: int,
    min_pattern_length: int,
    split_pattern_length_limit: int,
) -> bool:
    sub_patterns = _analyze_pattern_for_splits(
        loop_structures,
        pattern_idx,
        min_pattern_length,
        split_pattern_length_limit,
    )
    if not sub_patterns:
        return False
    _apply_pattern_split(loop_structures, pattern_idx, sub_patterns)
    return True


# ---------------------------------------------------------------------------
# Pattern rotation
# ---------------------------------------------------------------------------


def rotate_pattern(loop_structures: LoopStructures, pattern_idx: int) -> bool:
    pattern = loop_structures.patterns[pattern_idx]
    sequences = loop_structures.sequences[pattern_idx]
    pattern_len = len(pattern)

    shift_steps = 0
    while shift_steps < pattern_len:
        all_match = all(
            seq[-1] + shift_steps < len(loop_structures.operations)
            and loop_structures.operation_name_to_idx[
                loop_structures.operations[seq[-1] + shift_steps].name
            ]
            == pattern[shift_steps]
            for seq in sequences
        )
        if all_match:
            shift_steps += 1
        else:
            break

    shift_steps -= 1
    if shift_steps < 0:
        return False

    loop_structures.patterns[pattern_idx] = (
        pattern[shift_steps:] + pattern[:shift_steps]
    )
    loop_structures.sequences[pattern_idx] = [
        list(range(seq[0] + shift_steps, seq[-1] + shift_steps + 1))
        for seq in sequences
    ]
    sequences = loop_structures.sequences[pattern_idx]

    covered = loop_structures.get_covered_indices(
        [i for i in range(len(loop_structures.patterns)) if i != pattern_idx]
    )

    new_sequences = []
    for i, seq in enumerate(sequences):
        if i > 0 and sequences[i - 1][-1] + 1 == seq[0]:
            continue
        potential_start = seq[0] - pattern_len
        if potential_start >= 0 and all(
            (potential_start + j) not in covered for j in range(pattern_len)
        ):
            if all(
                loop_structures.operation_name_to_idx[
                    loop_structures.operations[potential_start + j].name
                ]
                == loop_structures.patterns[pattern_idx][j]
                for j in range(pattern_len)
            ):
                new_sequences.append(
                    list(range(potential_start, potential_start + pattern_len))
                )
                covered.update(range(potential_start, potential_start + pattern_len))

    if new_sequences:
        loop_structures.sequences[pattern_idx] = sorted(
            sequences + new_sequences,
            key=lambda seq: seq[0],
        )
    return True


# ---------------------------------------------------------------------------
# Pattern occurrence finding
# ---------------------------------------------------------------------------


def find_pattern_occurrences(
    loop_structures: LoopStructures,
    pattern_idx: int,
) -> List[List[int]]:
    pattern = loop_structures.patterns[pattern_idx]
    pattern_len = len(pattern)

    covered = loop_structures.get_covered_indices(
        [i for i in range(len(loop_structures.patterns)) if i != pattern_idx]
    )

    op_name_indices = [
        loop_structures.operation_name_to_idx[op.name]
        for op in loop_structures.operations
    ]

    first_element = pattern[0]
    max_start = len(loop_structures.operations) - pattern_len

    candidates = []
    for i in range(max_start + 1):
        if op_name_indices[i] != first_element or i in covered:
            continue
        if all((i + j) not in covered for j in range(1, pattern_len)) and all(
            op_name_indices[i + j] == pattern[j] for j in range(1, pattern_len)
        ):
            candidates.append(i)

    sequences = []
    for i in candidates:
        if sequences and i < sequences[-1][-1] + 1:
            continue
        sequences.append(list(range(i, i + pattern_len)))

    return sequences


# ---------------------------------------------------------------------------
# Seed pattern creation
# ---------------------------------------------------------------------------


def get_new_pattern_to_start(
    loop_structures: LoopStructures,
    operations: List[GPUOperation],
    min_loop_count: int,
    failed_ops: Optional[Set[int]] = None,
) -> Optional[Tuple[List[int], List[List[int]]]]:
    if failed_ops is None:
        failed_ops = set()

    covered = loop_structures.get_covered_indices()
    unused_op_indices = [
        loop_structures.operation_name_to_idx[op.name]
        for i, op in enumerate(operations)
        if i not in covered
    ]
    op_frequencies = Counter(unused_op_indices)
    if not op_frequencies:
        return None

    for most_frequent_kernel, freq in op_frequencies.most_common():
        if most_frequent_kernel in failed_ops or freq < min_loop_count:
            continue
        kernel_name = loop_structures.pattern_operation_names[most_frequent_kernel]
        new_pattern = [most_frequent_kernel]
        new_sequences = [
            [i]
            for i, op in enumerate(operations)
            if op.name == kernel_name and i not in covered
        ]
        if len(new_sequences) >= min_loop_count:
            return (new_pattern, new_sequences)

    return None


# ===========================================================================
# Main entry point
# ===========================================================================


def find_kernel_loops_single_stream(
    operations: List[GPUOperation],
    min_loop_count: int = 20,
    min_pattern_length: int = 5,
    split_pattern_length_limit: int = 35,
    loop_structures: Optional[LoopStructures] = None,
    stream_id: Optional[int] = None,
    snippets: bool = False,
) -> LoopStructures:
    """Find repeating kernel patterns in a single-stream operation list."""

    if stream_id is None and operations:
        streams_in_ops = {op.stream for op in operations}
        if len(streams_in_ops) == 1:
            stream_id = next(iter(streams_in_ops))

    if loop_structures is not None:
        if loop_structures.operations != operations:
            raise ValueError("loop_structures must have the same operations list")
    else:
        loop_structures = LoopStructures(operations=operations, stream_id=stream_id)

    mpl_schedule = list(range(max(20, min_pattern_length), min_pattern_length - 1, -1))

    if min_loop_count > 20:
        pass_schedule: List[Tuple[int, int]] = [
            (mpl, min_loop_count) for mpl in mpl_schedule
        ]
    else:
        mlc_start = max(20, min_loop_count)
        pass_schedule = [(mpl, mlc_start) for mpl in mpl_schedule]
        if min_loop_count < 20:
            _mlc_relaxed = list(range(19, min_loop_count - 1, -1))
            pass_schedule += [(min_pattern_length, mlc) for mlc in _mlc_relaxed]

    max_iterations_per_pass = 50

    extension_cache: Dict[int, Tuple[int, int, int]] = {}
    split_analysis_cache: Dict[Tuple[int, ...], List[List[int]]] = {}

    for current_min_pattern_length, current_min_loop_count in pass_schedule:
        if loop_structures._get_pattern_coverage() >= 1.0:
            break

        failed_ops = {
            op
            for op, (alen, _, bagree) in extension_cache.items()
            if alen < current_min_pattern_length and bagree < current_min_loop_count
        }

        for pidx in range(len(loop_structures.patterns) - 1, -1, -1):
            if len(loop_structures.patterns[pidx]) <= split_pattern_length_limit:
                continue
            pattern_key = tuple(loop_structures.patterns[pidx])
            sub_patterns = split_analysis_cache.get(pattern_key)
            if not sub_patterns:
                continue
            min_sub_len = min(len(sp) for sp in sub_patterns)
            if min_sub_len >= current_min_pattern_length:
                _apply_pattern_split(loop_structures, pidx, sub_patterns)

        iteration = 0
        while iteration < max_iterations_per_pass:
            pattern_idx = len(loop_structures.patterns)

            result = get_new_pattern_to_start(
                loop_structures,
                operations,
                current_min_loop_count,
                failed_ops,
            )
            if result is None:
                break

            new_pattern, new_sequences = result
            pattern_init_op = new_pattern[0]
            loop_structures.patterns.append(new_pattern)
            loop_structures.sequences.append(new_sequences)
            pattern_idx = len(loop_structures.patterns) - 1

            if loop_structures._get_pattern_coverage() >= 1.0:
                iteration += 1
                continue

            stop_best_agree = extend_pattern(
                loop_structures,
                pattern_idx,
                current_min_loop_count,
                current_min_pattern_length,
                snippets=snippets,
            )

            pattern_len = len(loop_structures.patterns[pattern_idx])
            seq_count = len(loop_structures.sequences[pattern_idx])

            if pattern_len < min_pattern_length:
                if loop_structures._get_pattern_coverage() < 1.0:
                    loop_structures.patterns.pop()
                    loop_structures.sequences.pop()
                    extension_cache[pattern_init_op] = (
                        pattern_len,
                        seq_count,
                        stop_best_agree,
                    )
                    failed_ops.add(pattern_init_op)

            elif pattern_len > split_pattern_length_limit:
                pattern_key = tuple(loop_structures.patterns[pattern_idx])
                if pattern_key not in split_analysis_cache:
                    sub_patterns = _analyze_pattern_for_splits(
                        loop_structures,
                        pattern_idx,
                        min_pattern_length,
                        split_pattern_length_limit,
                    )
                    split_analysis_cache[pattern_key] = sub_patterns
                else:
                    sub_patterns = split_analysis_cache[pattern_key]

                if sub_patterns:
                    min_sub_len = min(len(sp) for sp in sub_patterns)
                    if min_sub_len >= current_min_pattern_length:
                        _apply_pattern_split(
                            loop_structures,
                            pattern_idx,
                            sub_patterns,
                        )
            else:
                all_seqs = find_pattern_occurrences(loop_structures, pattern_idx)
                loop_structures.sequences[pattern_idx] = all_seqs

            iteration += 1

    return loop_structures
