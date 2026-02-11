###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import re
from typing import Any, Callable, cast, Dict, Optional

import pandas as pd
import json
import os
import re

import TraceLens.util
from TraceLens import TraceToTree
from ..TreePerf import GPUEventAnalyser


class TraceDiff:
    def __init__(self, tree1: TraceToTree, tree2: TraceToTree):
        self.baseline = tree1
        self.variant = tree2
        self.db1 = []
        self.db2 = []
        self.pod1 = set()
        self.pod2 = set()
        self.merged_tree = None  # Will hold the merged tree structure
        self.merged_uid_map = {}  # (tree_num, uid) -> corresponding_uid or -1
        self.diff_stats_df = pd.DataFrame()  # DataFrame for diff stats
        self.diff_stats_summary_df = pd.DataFrame()  # DataFrame for diff stats summary
        self.identical_traces = False
        self.cpu_op_map_trace1 = None
        self.cpu_op_map_trace2 = None
        self.cpu_op_map = None
        # Cache for merged tree mapping only (baseline/variant dicts are already in tree objects)
        self._merged_id_to_event = None

        # Automatically merge trees and initialize UID map
        self.merge_trees()

    def _get_baseline_uid2node(self):
        """Return baseline UID to node mapping from the tree object (already cached there)."""
        return getattr(self.baseline, "events_by_uid", {})

    def _get_variant_uid2node(self):
        """Return variant UID to node mapping from the tree object (already cached there)."""
        return getattr(self.variant, "events_by_uid", {})

    def _get_merged_id_to_event(self):
        """Lazily build and cache merged ID to event mapping."""
        if self._merged_id_to_event is None and self.merged_tree is not None:
            merged_events, _ = self.merged_tree
            self._merged_id_to_event = {
                event["merged_id"]: event for event in merged_events
            }
        return self._merged_id_to_event or {}

    def _get_uid_to_merged_id_maps(self):
        """Build reverse mappings from UIDs to merged_ids for efficient lookup."""
        if not hasattr(self, "_uid1_to_merged_id"):
            self._uid1_to_merged_id = {}
            self._uid2_to_merged_id = {}
            merged_id_to_event = self._get_merged_id_to_event()
            for mid, event in merged_id_to_event.items():
                uid1 = event.get("uid1")
                uid2 = event.get("uid2")
                if uid1 is not None:
                    self._uid1_to_merged_id[uid1] = mid
                if uid2 is not None:
                    self._uid2_to_merged_id[uid2] = mid
        return self._uid1_to_merged_id, self._uid2_to_merged_id

    def _invalidate_merged_cache(self):
        """Invalidate merged tree cache when tree is rebuilt."""
        self._merged_id_to_event = None

    def is_gpu_path(self, node):
        if node is None:
            return False
        return not node.get("non_gpu_path", False)

    def is_kernel(self, node):
        cat = node.get("cat") or node.get("category")
        if cat is None:
            try:
                cat = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Category)
            except Exception:
                pass
        return cat in ("kernel", "gpu_memcpy")

    @staticmethod
    def _normalize_name_for_comparison(name):
        """
        Normalize node names by removing variable parts (hex memory addresses and line numbers)
        to enable comparison of functionally identical nodes across traces.

        Removes:
        - Hex memory addresses: 0x7fe640752310 -> 0xXXXX
        - Line numbers in Python stack: path/file.py(715): func -> path/file.py: func

        Args:
            name: The name string to normalize

        Returns:
            The normalized name, or the original name if it's None
        """
        if name is None:
            return name
        # Remove hex memory addresses but keep the "at 0x" part for context
        normalized = re.sub(r"0x[0-9a-fA-F]+", "0xXXXX", name)
        # Remove line numbers from Python stack frames (filename.py(line_number): function)
        normalized = re.sub(r"\.py\(\d+\):", ".py:", normalized)
        return normalized

    @staticmethod
    def _longest_common_substring(s1: str, s2: str) -> str:
        """Return a longest contiguous common substring of s1 and s2, or '' if either is None/empty."""
        if not s1 or not s2:
            return ""
        m, n = len(s1), len(s2)
        max_len = 0
        end_idx = 0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        end_idx = i
        return s1[end_idx - max_len : end_idx] if max_len > 0 else ""

    def _get_op_name(self, uid, tree_num):
        """
        Unified method to get operation name from UID.
        Replaces 4 duplicate get_op_name() functions throughout the code.

        Args:
            uid: The UID to look up
            tree_num: 1 for baseline, 2 for variant

        Returns:
            str: The operation name or string representation of UID
        """
        if uid is None:
            return None

        tree_uid2node = (
            self._get_baseline_uid2node()
            if tree_num == 1
            else self._get_variant_uid2node()
        )
        node = tree_uid2node.get(uid)

        if node is None:
            return None

        # Try to get name from various possible keys
        name = node.get("name") if "name" in node else node.get("Name")
        if name is None:
            try:
                name = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name)
            except Exception:
                pass

        return name if name else str(uid)

    def get_diff_stats_df(self):
        """
        Return the detailed diff stats DataFrame (diff_stats_df).
        If the DataFrame is empty, print a message to generate reports first.
        """
        if getattr(self, "diff_stats_df", None) is None or self.diff_stats_df.empty:
            print(
                "[TraceDiff] diff_stats_df is empty. Please run generate_tracediff_report() first."
            )
            return None
        return self.diff_stats_df

    def get_diff_stats_summary_df(self):
        """
        Return the summary diff stats DataFrame (diff_stats_summary_df).
        If the DataFrame is empty, print a message to generate reports first.
        """
        if (
            getattr(self, "diff_stats_summary_df", None) is None
            or self.diff_stats_summary_df.empty
        ):
            print(
                "[TraceDiff] diff_stats_summary_df is empty. Please run generate_tracediff_report() first."
            )
            return None
        return self.diff_stats_summary_df

    def add_to_pod(self, node: Dict[str, Any], pod: set, tree: TraceToTree) -> None:
        """
        Iteratively adds the subtree rooted at the given node to the set of points of differences (PODs).
        Uses an iterative approach instead of recursion for better performance on deep trees.

        Args:
            node (Dict[str, Any]): The current node in the trace tree.
            pod (set): The set to which PODs will be added.
            tree (TraceToTree): The trace tree containing the events.
        """
        if not isinstance(node, dict):
            return

        # Iterative approach using a stack instead of recursion
        stack = [node]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue

            # Add current node's UID to pod
            uid = current.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
            if uid is not None:
                pod.add(uid)

            # Add children to stack
            children = tree.get_children_events(current)
            stack.extend(children)

    def _get_top_level_root(self, tree: TraceToTree, start_uid: int) -> int:
        """
        Find the top-level root node by traversing parent pointers upward from a starting UID.
        The root is the node with no parent, which is typically a python_function event at the
        top of the call stack.

        Args:
            tree (TraceToTree): The trace tree to traverse.
            start_uid (int): The UID to start traversal from (typically a CPU root node).

        Returns:
            int: The UID of the top-level root node.
        """
        current = tree.get_UID2event(start_uid)
        while True:
            parent_uid = current.get("parent")
            if parent_uid is None:
                root = current
                while True:
                    children = current.get("children", [])
                    if len(children) == 1:
                        current = tree.get_UID2event(children[0])
                    else:
                        children = current.get("children", [])
                        root["children"] = children
                        for child_uid in children:
                            child_event = tree.get_UID2event(child_uid)
                            child_event["parent"] = root.get(
                                TraceLens.util.TraceEventUtils.TraceKeys.UID
                            )
                        current = root
                        break
                return current.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
            current = tree.get_UID2event(parent_uid)

    def wagner_fischer(self, items1, items2, wf_cache):
        """
        Wagner-Fischer algorithm that works with any items and name lookup functions.

        Args:
            items1: List of items
            items2: List of items
            wf_cache: Dictionary for caching results

        Returns:
            List of operations: [("match", i, j), ("delete", i, None), ("insert", None, j), ...]
        """
        # Pre-compute names for cache key
        names1 = [
            self._normalize_name_for_comparison(self._get_op_name(item, 1))
            for item in items1
        ]
        names2 = [
            self._normalize_name_for_comparison(self._get_op_name(item, 2))
            for item in items2
        ]

        # Check cache
        cache_key = (tuple(items1), tuple(items2))
        if cache_key in wf_cache:
            return wf_cache[cache_key]

        m, n = len(items1), len(items2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if names1[i - 1] == names2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )
        # Backtrack
        i, j = m, n
        ops = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and names1[i - 1] == names2[j - 1]:
                ops.append(("match", i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
                ops.append(("delete", i - 1, None))
                i -= 1
            else:
                ops.append(("insert", None, j - 1))
                j -= 1
        ops.reverse()
        wf_cache[cache_key] = ops
        return ops

    def merge_trees(self):
        """
        Merges the two trees using the PODs from get_diff_boundaries, inspired by merge_tree_from_pod, but returns a flat list of merged event dicts.
        Each merged event has a unique merged_id, children as merged_id references, and root merged_ids. Compatible with TraceToTree format.
        Returns: (merged_events, merged_root_ids)
        """

        print("[TraceDiff] Calculating trace diff and creating merged tree...")

        self._invalidate_merged_cache()

        tree1 = self.baseline
        tree2 = self.variant
        if not tree1.cpu_root_nodes or not tree2.cpu_root_nodes:
            raise ValueError(
                "Both trees must have at least one root node in cpu_root_nodes."
            )

        baseline_uid2node = self._get_baseline_uid2node()
        variant_uid2node = self._get_variant_uid2node()
        wf_cache = {}
        merged_events = []
        merged_id_counter = [0]
        uid_pair_to_merged_id = {}

        def make_event(merged_id, uid1, uid2, merged_type, children, nn_module_stack):
            return {
                "merged_id": merged_id,
                "uid1": uid1,
                "uid2": uid2,
                "merged_type": merged_type,
                "children": children,  # list of merged_id
                "nn_module_stack": nn_module_stack,
            }

        def safe_children(uid2node, uid):
            if uid is None:
                return []
            node = uid2node.get(uid)
            if node is None or not isinstance(node, dict):
                return []
            return node.get("children", [])

        def get_name_uid(uid, tree_num):
            name = self._get_op_name(uid, tree_num)
            return self._normalize_name_for_comparison(name) if name else None

        def get_children_with_missing(uid1, uid2):
            """Get aligned children lists, adding missing-by-name from full child list."""
            children1 = safe_children(baseline_uid2node, uid1)
            children2 = safe_children(variant_uid2node, uid2)
            all_nodes1 = [
                baseline_uid2node[c] for c in children1 if baseline_uid2node.get(c)
            ]
            all_nodes2 = [variant_uid2node[c] for c in children2 if variant_uid2node.get(c)]
            gpu_children1 = [
                n[TraceLens.util.TraceEventUtils.TraceKeys.UID]
                for n in all_nodes1
                if self.is_gpu_path(n)
            ]
            gpu_children2 = [
                n[TraceLens.util.TraceEventUtils.TraceKeys.UID]
                for n in all_nodes2
                if self.is_gpu_path(n)
            ]
            gpu_names1 = {get_name_uid(c, 1) for c in gpu_children1}
            gpu_names2 = {get_name_uid(c, 2) for c in gpu_children2}

            all_names1 = {get_name_uid(c, 1): c for c in children1}
            all_names2 = {get_name_uid(c, 2): c for c in children2}

            names1_only = gpu_names1 - gpu_names2
            names2_only = gpu_names2 - gpu_names1

            for n in names1_only:
                if n in all_names2:
                    gpu_children2.append(all_names2[n])
            for n in names2_only:
                if n in all_names1:
                    gpu_children1.append(all_names1[n])

            def sort_by_ts(uids, uid2node):
                nodes = [(uid2node.get(u), u) for u in uids]
                nodes.sort(key=lambda x: x[0].get("ts", 0))
                return [u for _, u in nodes]

            return sort_by_ts(gpu_children1, baseline_uid2node), sort_by_ts(
                gpu_children2, variant_uid2node
            )

        def traverse_and_merge(uid1, uid2):
            key = (uid1, uid2)
            if key in uid_pair_to_merged_id:
                return uid_pair_to_merged_id[key]

            node1 = baseline_uid2node.get(uid1) if uid1 is not None else None
            node2 = variant_uid2node.get(uid2) if uid2 is not None else None

            # Boundary logic: when both exist, check POD and name mismatch
            if uid1 and uid2:
                if uid1 in self.pod1 or uid2 in self.pod2:
                    pass  # Skip boundary logic, still need merge structure
                else:
                    name1 = get_name_uid(uid1, 1)
                    name2 = get_name_uid(uid2, 2)
                    if name1 != name2:
                        self.db1.append(node1)
                        self.db2.append(node2)
                        self.pod1.add(uid1)
                        self.pod2.add(uid2)

            # POD logic: when only one exists, add to POD
            if uid1 and not uid2:
                self.pod1.add(uid1)
            if uid2 and not uid1:
                self.pod2.add(uid2)

            merged_id = merged_id_counter[0]
            merged_id_counter[0] += 1
            uid_pair_to_merged_id[key] = merged_id

            if uid1 and uid2:
                self.merged_uid_map[(1, uid1)] = uid2
                self.merged_uid_map[(2, uid2)] = uid1
                nn_module_stack = node1.get("nn_module_stack", "")
            elif uid1:
                self.merged_uid_map[(1, uid1)] = -1
                nn_module_stack = node1.get("nn_module_stack", "")
            else:
                self.merged_uid_map[(2, uid2)] = -1
                nn_module_stack = node2.get("nn_module_stack", "")

            children1, children2 = get_children_with_missing(uid1, uid2)
            ops = self.wagner_fischer(children1, children2, wf_cache)

            # Promote delete+insert pairs to match when two levels of children agree.
            delete_indices = [i for op, i, j in ops if op == "delete"]
            insert_indices = [j for op, i, j in ops if op == "insert"]
            candidates = []
            for i in delete_indices:
                for j in insert_indices:
                    uid_d, uid_i = children1[i], children2[j]
                    imm1 = safe_children(baseline_uid2node, uid_d)
                    imm2 = safe_children(variant_uid2node, uid_i)
                    names_imm1 = [get_name_uid(u, 1) for u in imm1]
                    names_imm2 = [get_name_uid(u, 2) for u in imm2]
                    if names_imm1 != names_imm2:
                        continue
                    same_two_levels = True
                    for k in range(len(imm1)):
                        grand1 = safe_children(baseline_uid2node, imm1[k])
                        grand2 = safe_children(variant_uid2node, imm2[k])
                        grand_names1 = [get_name_uid(u, 1) for u in grand1]
                        grand_names2 = [get_name_uid(u, 2) for u in grand2]
                        if grand_names1 != grand_names2:
                            same_two_levels = False
                            break
                    if same_two_levels:
                        candidates.append((i, j))
            used_i, used_j = set(), set()
            promoted = {}  # i -> j for (i, j) promoted to match
            for (i, j) in candidates:
                if i not in used_i and j not in used_j:
                    promoted[i] = j
                    used_i.add(i)
                    used_j.add(j)
            for (i, j) in promoted.items():
                node1 = baseline_uid2node.get(children1[i])
                node2 = variant_uid2node.get(children2[j])
                if node1 is not None and node2 is not None:
                    raw1 = self._get_op_name(children1[i], 1) or ""
                    raw2 = self._get_op_name(children2[j], 2) or ""
                    common = self._longest_common_substring(raw1, raw2)
                    name_to_set = common if common else (raw1 or raw2)
                    node1["name"] = name_to_set
                    node2["name"] = name_to_set
            promoted_j = set(promoted.values())
            new_ops = []
            for op, i, j in ops:
                if op == "delete" and i in promoted:
                    new_ops.append(("match", i, promoted[i]))
                elif op == "insert" and j in promoted_j:
                    continue
                else:
                    new_ops.append((op, i, j))
            ops = new_ops

            child_merged_ids = []
            for op, i, j in ops:
                if op == "match":
                    c1, c2 = children1[i], children2[j]
                    child_merged_ids.append(traverse_and_merge(c1, c2))
                elif op == "delete":
                    c1 = children1[i]
                    child_node = baseline_uid2node.get(c1)
                    if child_node:
                        self.db1.append(child_node)
                    child_merged_ids.append(traverse_and_merge(c1, None))
                elif op == "insert":
                    c2 = children2[j]
                    child_node = variant_uid2node.get(c2)
                    if child_node:
                        self.db2.append(child_node)
                    child_merged_ids.append(traverse_and_merge(None, c2))

            merged_type = (
                "combined" if (uid1 and uid2) else ("trace1" if uid1 else "trace2")
            )
            event = make_event(
                merged_id, uid1, uid2, merged_type, child_merged_ids, nn_module_stack
            )
            merged_events.append(event)
            return merged_id

        root_uid1 = self._get_top_level_root(tree1, tree1.cpu_root_nodes[0])
        root_uid2 = self._get_top_level_root(tree2, tree2.cpu_root_nodes[0])

        merged_root_id = traverse_and_merge(root_uid1, root_uid2)
        merged_root_ids = [merged_root_id]

        self.merged_tree = (merged_events, merged_root_ids)
        return self.merged_tree

    def get_corresponding_uid(self, tree_num, uid):
        """
        Given a tree number (1 or 2) and a UID, return the corresponding UID from the other tree if combined, else -1.
        """
        return self.merged_uid_map.get((tree_num, uid), -1)

    def print_merged_subtree(self, uid_tree1=None, uid_tree2=None):
        if uid_tree1 is None and uid_tree2 is None:
            raise ValueError("At least one of uid_tree1 or uid_tree2 must be provided.")
        if self.merged_tree is None:
            raise ValueError(
                "merged_tree is not initialized. Call merge_trees() first."
            )
        merged_events, merged_root_ids = self.merged_tree
        merged_id_to_event = {event["merged_id"]: event for event in merged_events}

        # Find merged_id corresponding to the given uid
        merged_id_to_event = self._get_merged_id_to_event()
        uid1_to_merged_id, uid2_to_merged_id = self._get_uid_to_merged_id_maps()

        # Efficiently find merged_id using O(1) lookup instead of linear search
        merged_id = None
        if uid_tree1 is not None:
            merged_id = uid1_to_merged_id.get(uid_tree1)
        elif uid_tree2 is not None:
            merged_id = uid2_to_merged_id.get(uid_tree2)

        if merged_id is None:
            raise ValueError("Could not find merged node for the given UID.")

        # Print merged subtree to console
        def print_merged_tree_to_console(merged_id, prefix="", is_last=True):
            node = merged_id_to_event[merged_id]
            merge_type = node["merged_type"]
            name1 = (
                self._get_op_name(node["uid1"], 1) if node["uid1"] is not None else None
            )
            name2 = (
                self._get_op_name(node["uid2"], 2) if node["uid2"] is not None else None
            )
            connector = "└── " if is_last else "├── "
            if merge_type == "combined":
                if name1 == name2 and name1 is not None:
                    line = f"{prefix}{connector}{name1}"
                else:
                    line = f"{prefix}{connector}{merge_type}: {name1} | {name2}"
            elif merge_type == "trace1":
                line = f"{prefix}{connector}>> {merge_type}: {name1}"
            elif merge_type == "trace2":
                line = f"{prefix}{connector}<< {merge_type}: {name2}"
            else:
                line = f"{prefix}{connector}{merge_type}: {name1} | {name2}"
            # Sort children by merge_type order: combined, trace1, trace2
            children = [merged_id_to_event[cid] for cid in node["children"]]
            combined = [
                c["merged_id"] for c in children if c["merged_type"] == "combined"
            ]
            trace1 = [c["merged_id"] for c in children if c["merged_type"] == "trace1"]
            trace2 = [c["merged_id"] for c in children if c["merged_type"] == "trace2"]
            sorted_children = combined + trace1 + trace2
            child_count = len(sorted_children)
            for i, cid in enumerate(sorted_children):
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_merged_tree_to_console(
                    cid, new_prefix, is_last=(i == child_count - 1)
                )

        print_merged_tree_to_console(merged_id, prefix="", is_last=True)

    def print_merged_tree(self, output_file, prune_non_gpu=False):
        if self.merged_tree is None:
            raise ValueError(
                "merged_tree is not initialized. Call merge_trees() first."
            )
        merged_events, merged_root_ids = self.merged_tree
        output_lines = []
        merged_id_to_event = self._get_merged_id_to_event()

        def subtree_has_gpu(merged_id: int) -> bool:
            # Depending on the merge type, get the corresponsonding UIDs in both trees
            node = merged_id_to_event[merged_id]
            uid1 = node["uid1"]
            uid2 = node["uid2"]

            # Check in baseline tree
            node1 = self.baseline.get_UID2event(uid1) if uid1 is not None else None
            node2 = self.variant.get_UID2event(uid2) if uid2 is not None else None

            if node1 and not node1.get("non_gpu_path", False):
                return True
            if node2 and not node2.get("non_gpu_path", False):
                return True

            return False

        def print_merged_tree_to_lines(merged_id, prefix="", is_last=True):
            node = merged_id_to_event[merged_id]
            merge_type = node["merged_type"]
            name1 = (
                self._get_op_name(node["uid1"], 1) if node["uid1"] is not None else None
            )
            name2 = (
                self._get_op_name(node["uid2"], 2) if node["uid2"] is not None else None
            )
            connector = "└── " if is_last else "├── "
            if merge_type == "combined":
                if name1 == name2 and name1 is not None:
                    line = f"{prefix}{connector}{name1}"
                else:
                    line = f"{prefix}{connector}{merge_type}: {name1} | {name2}"
            elif merge_type == "trace1":
                line = f"{prefix}{connector}>> {merge_type}: {name1}"
            elif merge_type == "trace2":
                line = f"{prefix}{connector}<< {merge_type}: {name2}"
            else:
                line = f"{prefix}{connector}{merge_type}: {name1} | {name2}"
            output_lines.append(line)
            # Sort children by merge_type order: combined, trace1, trace2
            children = [merged_id_to_event[cid] for cid in node["children"]]
            combined = [
                c["merged_id"] for c in children if c["merged_type"] == "combined"
            ]
            trace1 = [c["merged_id"] for c in children if c["merged_type"] == "trace1"]
            trace2 = [c["merged_id"] for c in children if c["merged_type"] == "trace2"]
            sorted_children = combined + trace1 + trace2
            child_count = len(sorted_children)
            for i, cid in enumerate(sorted_children):
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_merged_tree_to_lines(
                    cid, new_prefix, is_last=(i == child_count - 1)
                )

        for i, root_id in enumerate(merged_root_ids):
            if prune_non_gpu and not subtree_has_gpu(root_id):
                continue

            print_merged_tree_to_lines(
                root_id, prefix="", is_last=(i == len(merged_root_ids) - 1)
            )

        with open(output_file, "w") as f:
            for line in output_lines:
                f.write(line + "\n")

    def generate_diff_stats(self):
        """
        For combined ops on a GPU path with non-combined children, generate a DataFrame with columns:
        name, input_shape, total_kernel_time_trace1, total_kernel_time_trace2, kernel_names_trace1, kernel_names_trace2
        Stores the DataFrame in self.diff_stats_df and returns it.
        """
        if self.merged_tree is None:
            raise ValueError(
                "merged_tree is not initialized. Call merge_trees() first."
            )
        merged_events, merged_root_ids = self.merged_tree
        merged_id_to_event = self._get_merged_id_to_event()
        baseline_uid2node = self._get_baseline_uid2node()
        variant_uid2node = self._get_variant_uid2node()

        def list_to_tuple(obj):
            if isinstance(obj, list):
                return tuple(list_to_tuple(item) for item in obj)
            return obj

        def get_input_shape(node):
            args = node.get("args", {})
            shape = args.get("Input Dims")
            if shape is not None:
                return list_to_tuple(shape)
            return ""

        def get_concrete_inputs(node):
            args = node.get("args", {})
            val = args.get("Concrete Inputs")
            if val is not None:
                return list_to_tuple(val)
            return ""

        def get_input_strides(node):
            args = node.get("args", {})
            val = args.get("Input Strides")
            if val is not None:
                return list_to_tuple(val)
            return ""

        def get_input_type(node):
            args = node.get("args", {})
            val = args.get("Input type")
            if val is not None:
                return list_to_tuple(val)
            return ""

        def get_duration(node):
            dur = node.get("dur")
            if dur is not None:
                return dur
            try:
                dur = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Duration)
            except Exception:
                pass
            return dur

        def find_last_cpu_op_on_gpu_path(
            merged_child, tree_obj, tree_uid2node, uid_key, all_kernels=None
        ):
            """
            Traverse down from a merged child node to find the lowest (deepest) CPU operation
            that contains ALL the kernels in the branch.

            Args:
                merged_child: The merged tree child node to start from
                tree_obj: The tree object (baseline or variant) to use for event_to_category
                tree_uid2node: The uid2node dictionary (baseline_uid2node or variant_uid2node)
                uid_key: Either "uid1" or "uid2" depending on which trace we're looking at
                all_kernels: Set of all kernel UIDs that should be present (computed on first call)

            Returns:
                The UID of the lowest CPU operation that contains all kernels, or None if not found
            """
            uid = merged_child.get(uid_key)
            if uid is None:
                return None

            # Check current node
            node = tree_uid2node.get(uid)
            if node is None:
                return None

            # If not on GPU path, return None
            if not self.is_gpu_path(node):
                return None

            # On first call, get all kernels from this starting node
            if all_kernels is None:
                all_kernels = set(node.get("gpu_events", []))
                if not all_kernels:
                    return None  # No kernels to find

            # Get kernels in current node's subtree
            current_kernels = set(node.get("gpu_events", []))

            # If this node doesn't contain all kernels, it can't be the answer
            if not all_kernels.issubset(current_kernels):
                return None

            is_cpu_op = tree_obj.event_to_category(node) == "cpu_op"

            # Try to find a deeper CPU op that also contains all kernels
            for child_merged_id in merged_child.get("children", []):
                child_merged = merged_id_to_event.get(child_merged_id)
                if child_merged is None:
                    continue

                # Recursively search this child
                result = find_last_cpu_op_on_gpu_path(
                    child_merged, tree_obj, tree_uid2node, uid_key, all_kernels
                )
                if result is not None:
                    return result  # Return the deeper CPU op that contains all kernels

            # If this node is a CPU op and contains all kernels, and no children do, return this one
            if is_cpu_op and all_kernels.issubset(current_kernels):
                return uid

            return None

        def find_all_last_cpu_ops_on_gpu_path(
            merged_node, tree_obj, tree_uid2node, uid_key
        ):
            """
            Traverse down from a merged node to find ALL lowest (deepest) CPU operations
            that are on GPU paths - the CPU ops closest to actual kernel execution.

            Args:
                merged_node: The merged tree node to start from
                tree_obj: The tree object (baseline or variant) to use for event_to_category
                tree_uid2node: The uid2node dictionary (baseline_uid2node or variant_uid2node)
                uid_key: Either "uid1" or "uid2" depending on which trace we're looking at

            Returns:
                List of UIDs of lowest CPU operations on GPU paths (closest to kernels)
            """
            uid = merged_node.get(uid_key)
            if uid is None:
                return []

            # Check current node
            node = tree_uid2node.get(uid)
            if node is None:
                return []

            # If not on GPU path, return empty
            if not self.is_gpu_path(node):
                return []

            is_cpu_op = tree_obj.event_to_category(node) == "cpu_op"

            # Traverse all children and collect CPU ops from each branch
            cpu_ops = []
            for child_merged_id in merged_node.get("children", []):
                child_merged = merged_id_to_event.get(child_merged_id)
                if child_merged is None:
                    continue

                # Recursively search this child and collect results
                child_cpu_ops = find_all_last_cpu_ops_on_gpu_path(
                    child_merged, tree_obj, tree_uid2node, uid_key
                )
                cpu_ops.extend(child_cpu_ops)

            # If this node is a CPU op and no children returned CPU ops, return this one
            # (it's the lowest/deepest CPU op on this path)
            if is_cpu_op and not cpu_ops:
                return [uid]

            # Otherwise return whatever children found (they're deeper)
            return cpu_ops

        def get_kernel_info_subtree(root_uid, tree_uid2node):
            node = tree_uid2node.get(root_uid)
            gpu_event_uids = node["gpu_events"]
            gpu_events = [tree_uid2node.get(uid) for uid in gpu_event_uids]
            kernel_names = [gpu_event["name"] for gpu_event in gpu_events]
            kernel_time = GPUEventAnalyser(gpu_events).compute_metrics()["busy_time"]
            return kernel_names, kernel_time

        rows = []
        visited_stats_nodes = set()

        def traverse(merged_id, combined_parent_node):
            if merged_id in visited_stats_nodes:
                return
            node = merged_id_to_event[merged_id]
            mt = node["merged_type"]
            if mt == "combined":
                event1 = baseline_uid2node.get(node["uid1"])
                event2 = variant_uid2node.get(node["uid2"])
                if (
                    event1
                    and event2
                    and self.is_gpu_path(event1)
                    and self.is_gpu_path(event2)
                ):
                    children = [merged_id_to_event[cid] for cid in node["children"]]
                    non_combined_children = [
                        c for c in children if c["merged_type"] != "combined"
                    ]
                    non_combined_children_trace1_gpu_paths = [
                        child
                        for child in non_combined_children
                        if self.is_gpu_path(baseline_uid2node.get(child.get("uid1")))
                    ]
                    non_combined_children_trace2_gpu_paths = [
                        child
                        for child in non_combined_children
                        if self.is_gpu_path(variant_uid2node.get(child.get("uid2")))
                    ]
                    if (
                        non_combined_children_trace1_gpu_paths
                        or non_combined_children_trace2_gpu_paths
                    ) or (self.is_kernel(event1) and self.is_kernel(event2)):

                        # Store the LCA name from this combined node
                        lca_name = self._get_op_name(
                            node["uid1"], 1
                        ) or self._get_op_name(node["uid2"], 2)
                        lca_name = re.sub(r"\(\d+\)", "", lca_name)

                        gpu_event_uids1 = []
                        for child in non_combined_children_trace1_gpu_paths:
                            child_node = baseline_uid2node.get(child.get("uid1"))
                            gpu_event_uids1.extend(child_node.get("gpu_events", []))
                        if self.is_kernel(event1):
                            gpu_event_uids1.append(event1["UID"])

                        gpu_event_uids2 = []
                        for child in non_combined_children_trace2_gpu_paths:
                            child_node = variant_uid2node.get(child.get("uid2"))
                            gpu_event_uids2.extend(child_node.get("gpu_events", []))
                        if self.is_kernel(event2):
                            gpu_event_uids2.append(event2["UID"])

                        def add_rows(gpu_event_uids, tree_obj, uid2node, source):
                            tree_num = 1 if source == "trace1" else 2
                            for gpu_uid in gpu_event_uids:
                                gpu_event = uid2node.get(gpu_uid)
                                if gpu_event is None:
                                    continue
                                parent_uid = gpu_event.get("parent")
                                parent_node = uid2node.get(parent_uid)
                                while parent_node is not None:
                                    if (
                                        tree_obj.event_to_category(parent_node)
                                        == "cpu_op"
                                    ):
                                        break
                                    parent_uid = parent_node.get("parent")
                                    parent_node = uid2node.get(parent_uid)

                                if parent_node is None:
                                    continue

                                rows.append(
                                    {
                                        "name": gpu_event["name"],
                                        "cpu_op_name": self._get_op_name(
                                            parent_uid, tree_num
                                        ),
                                        "source": source,
                                        "Input Dims": get_input_shape(parent_node),
                                        "Input Strides": get_input_strides(parent_node),
                                        "Input type": get_input_type(parent_node),
                                        "Concrete Inputs": get_concrete_inputs(
                                            parent_node
                                        ),
                                        "kernel_time": gpu_event.get("dur", 0),
                                        "lowest_common_ancestor_name": lca_name,
                                        "lowest_common_ancestor_id": node["merged_id"],
                                        "nn_module_stack": ";".join(
                                            str(x)
                                            for x in parent_node.get(
                                                "nn_module_stack", []
                                            )
                                        ),
                                        "nn_module_parent": (
                                            parent_node.get("nn_module_stack") or [""]
                                        )[-1],
                                    }
                                )

                        add_rows(
                            gpu_event_uids1, self.baseline, baseline_uid2node, "trace1"
                        )

                        add_rows(
                            gpu_event_uids2, self.variant, variant_uid2node, "trace2"
                        )

                        visited_stats_nodes.add(merged_id)
                        visited_stats_nodes.update(
                            [
                                child.get("merged_id")
                                for child in non_combined_children_trace1_gpu_paths
                                + non_combined_children_trace2_gpu_paths
                            ]
                        )

            elif mt == "trace1":
                event1 = baseline_uid2node.get(node["uid1"])
                if event1 and self.is_gpu_path(event1):
                    # This branch should only be executed if root is not combined and is on GPU path
                    if combined_parent_node is not None:
                        lca_id = combined_parent_node.get("merged_id")
                        lca = merged_id_to_event[lca_id]
                        lca_name = self._get_op_name(lca["uid1"], 1)
                        lca_name = re.sub(r"\(\d+\)", "", lca_name)
                    else:
                        lca_name = None  # Root node has no LCA
                        lca_id = None

                    # Get all GPU kernels from trace1's branch
                    gpu_event_uids = event1.get("gpu_events", [])
                    for gpu_uid in gpu_event_uids:
                        gpu_event = baseline_uid2node.get(gpu_uid)

                        # Find parent CPU operation for this GPU event
                        parent_uid = gpu_event.get("parent")
                        parent_node = baseline_uid2node.get(parent_uid)

                        # Traverse up to find the first CPU op
                        while parent_node is not None:
                            if self.baseline.event_to_category(parent_node) == "cpu_op":
                                break
                            parent_uid = parent_node.get("parent")
                            parent_node = baseline_uid2node.get(parent_uid)

                        if parent_node is None:
                            continue

                        child_name = self._get_op_name(parent_uid, 1)

                        rows.append(
                            {
                                "name": gpu_event["name"],
                                "cpu_op_name": child_name,
                                "source": "trace1",
                                "Input Dims": get_input_shape(parent_node),
                                "Input Strides": get_input_strides(parent_node),
                                "Input type": get_input_type(parent_node),
                                "Concrete Inputs": get_concrete_inputs(parent_node),
                                "kernel_time": gpu_event.get("dur", 0),
                                "lowest_common_ancestor_name": lca_name,
                                "lowest_common_ancestor_id": lca_id,
                                "nn_module_stack": ";".join(
                                    str(x)
                                    for x in parent_node.get("nn_module_stack", [])
                                ),
                                "nn_module_parent": (
                                    parent_node.get("nn_module_stack") or [""]
                                )[-1],
                            }
                        )

                visited_stats_nodes.add(merged_id)
                return
            elif mt == "trace2":
                event2 = variant_uid2node.get(node["uid2"])
                if event2 and self.is_gpu_path(event2):
                    # This branch should only be executed if root is not combined and is on GPU path
                    if combined_parent_node is not None:
                        lca_id = combined_parent_node.get("merged_id")
                        lca = merged_id_to_event[lca_id]
                        lca_name = self._get_op_name(lca["uid2"], 2)
                        lca_name = re.sub(r"\(\d+\)", "", lca_name)
                    else:
                        lca_name = None  # Root node has no LCA
                        lca_id = None

                    # Get all GPU kernels from trace2's branch
                    gpu_event_uids = event2.get("gpu_events", [])
                    for gpu_uid in gpu_event_uids:
                        gpu_event = variant_uid2node.get(gpu_uid)

                        # Find parent CPU operation for this GPU event
                        parent_uid = gpu_event.get("parent")
                        parent_node = variant_uid2node.get(parent_uid)

                        # Traverse up to find the first CPU op
                        while parent_node is not None:
                            if self.variant.event_to_category(parent_node) == "cpu_op":
                                break
                            parent_uid = parent_node.get("parent")
                            parent_node = variant_uid2node.get(parent_uid)

                        if parent_node is None:
                            continue

                        child_name = self._get_op_name(parent_uid, 2)

                        rows.append(
                            {
                                "name": gpu_event["name"],
                                "cpu_op_name": child_name,
                                "source": "trace2",
                                "Input Dims": get_input_shape(parent_node),
                                "Input Strides": get_input_strides(parent_node),
                                "Input type": get_input_type(parent_node),
                                "Concrete Inputs": get_concrete_inputs(parent_node),
                                "kernel_time": gpu_event.get("dur", 0),
                                "lowest_common_ancestor_name": lca_name,
                                "lowest_common_ancestor_id": lca_id,
                                "nn_module_stack": ";".join(
                                    str(x)
                                    for x in parent_node.get("nn_module_stack", [])
                                ),
                                "nn_module_parent": (
                                    parent_node.get("nn_module_stack") or [""]
                                )[-1],
                            }
                        )

                visited_stats_nodes.add(merged_id)
                return

            # Only traverse children if either trace is on a GPU path
            should_traverse_children = False
            if self.is_gpu_path(event1) or self.is_gpu_path(event2):
                should_traverse_children = True

            if should_traverse_children:
                for cid in node["children"]:
                    traverse(cid, combined_parent_node)
            return

        for root_id in merged_root_ids:
            traverse(root_id, None)

        df = pd.DataFrame(rows)

        df_trace1 = df[df["source"] == "trace1"].drop(columns=["source"])
        df_trace2 = df[df["source"] == "trace2"].drop(columns=["source"])
        if df_trace1.reset_index(drop=True).equals(df_trace2.reset_index(drop=True)):
            print("[TraceDiff] Identical traces detected")
            self.identical_traces = True
        else:
            self.identical_traces = False

        self.diff_stats_df = df
        return df

    def get_df_diff_stats_unique_args(
        self, op_name: str | None = None, agg_metrics: list[str] = ["mean"]
    ) -> pd.DataFrame:
        """
        Summarise diff stats across two traces by grouping on all argument columns and
        aggregating timing differences.

        Args:
            df_diff_stats (pd.DataFrame): DataFrame containing diff stats with trace1 and trace2 metrics.
            op_name (str, optional): If provided, only include rows where `name == op_name`.
            agg_metrics (list[str]): List of aggregation functions (e.g. ['mean', 'median']).
                                    'sum' will automatically be included if not in agg_metrics.

        Returns:
            pd.DataFrame: Summarised DataFrame sorted by the total difference column.
        """
        if self.diff_stats_df is None or self.diff_stats_df.empty:
            print(
                "[TraceDiff] diff_stats_df is empty. Please run generate_diff_stats() first."
            )
            return None
        # Avoid unnecessary copies - use views when filtering
        df_filtered = self.diff_stats_df
        if op_name:
            df_filtered = df_filtered[df_filtered["name"] == op_name]
        df_filtered = df_filtered.drop(columns=["lowest_common_ancestor_id"])

        # 3. Identify “argument” columns (everything that isn’t a metric)
        metric_columns = ["kernel_time"]
        grouping_cols_original = [
            c for c in df_filtered.columns if c not in metric_columns
        ]

        # 4. Build aggregation dictionary - ensure sum is always included
        agg_metrics_set = set(agg_metrics) | {"sum"}
        agg_dict = {mcol: list(agg_metrics_set) for mcol in metric_columns}
        for col in grouping_cols_original:
            agg_dict[col] = "first"  # keep first occurrence of each argument column

        # 5. Try groupby directly; fallback to string conversion for unhashable types
        try:
            df_agg = df_filtered.groupby(grouping_cols_original, dropna=False).agg(
                agg_dict
            )
            # Add row_count column (number of grouped rows per unique group)
            df_agg["operation_count"] = df_filtered.groupby(
                grouping_cols_original, dropna=False
            ).size()
        except TypeError:
            # Fallback for unhashable types (lists/dicts): convert to strings
            str_cols = [f"{col}_str_repr" for col in grouping_cols_original]
            df_temp = df_filtered.copy()
            for col, str_col in zip(grouping_cols_original, str_cols):
                df_temp[str_col] = df_temp[col].astype(str)
            df_agg = df_temp.groupby(str_cols, dropna=False).agg(agg_dict)
            # Add row_count column for stringified grouping columns
            df_agg["operation_count"] = df_temp.groupby(str_cols, dropna=False).size()

        # 7. Flatten the multi‑index column labels
        df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
        df_agg = df_agg.reset_index(drop=True)

        # 8. Rename “_first” columns back to the original column names for clarity
        rename_map = {}
        for col in grouping_cols_original:
            col_first = f"{col}_first"
            if col_first in df_agg.columns:
                rename_map[col_first] = col
        df_agg = df_agg.rename(columns=rename_map)

        # 9. Reorder columns: original argument columns first, then aggregated metric columns
        primary_cols = grouping_cols_original
        metric_cols = []
        for metric in metric_columns:
            for agg in agg_metrics + ([] if "sum" in agg_metrics else ["sum"]):
                col_name = f"{metric}_{agg}"
                if col_name in df_agg.columns:
                    metric_cols.append(col_name)
        metric_cols = list(dict.fromkeys(metric_cols))  # remove duplicates
        other_cols = [
            col for col in df_agg.columns if col not in primary_cols + metric_cols
        ]
        df_agg = df_agg[primary_cols + metric_cols + other_cols]
        df_agg = df_agg.rename(columns={"operation_count_": "operation_count"})
        cols = list(df_agg.columns)
        cols.remove("operation_count")
        cols.insert(1, "operation_count")
        df_agg = df_agg[cols]

        # 10. Sort by the trace1 kernel time sum
        sort_col = "kernel_time_sum"
        if sort_col in df_agg.columns:
            df_agg = df_agg.sort_values(by=sort_col, ascending=False, ignore_index=True)

        self.diff_stats_unique_args_summary_df = df_agg
        return df_agg

    def get_cpu_op_to_kernels_json(self) -> tuple[dict, dict]:
        """
        Create a JSON-serializable dict mapping CPU ops to the kernels they call,
        for both traces. Uses 'name' (kernel) and 'cpu_op_name' from diff_stats_unique_args_summary_df.

        Returns:
            Dict with keys "trace1" and "trace2", each mapping cpu_op_name -> list of kernel names.
        """



        def cleanKernelNames(kernel_name: str) -> str:
            # Clean kernel names to remove text between (), <>, []
            kernel_name = re.sub(r"\(.*?\)", "", kernel_name)
            kernel_name = re.sub(r"<.*?>", "", kernel_name)
            kernel_name = re.sub(r"\[.*?\]", "", kernel_name)

            # GEMM kernel. Keeps
            if kernel_name.startswith("Cijk"):
                kernel_name = kernel_name.split("_UserArgs")[0]

            # If kernel name contains 'Cijk', truncate everything after 'SAV_'
            if kernel_name.startswith("_gemm_"):
                kernel_name = kernel_name.split("_BLOCK")[0]

            if kernel_name.startswith("_matmul_"):
                kernel_name = kernel_name.split("_matmul")[0] + "_matmul"
            if kernel_name.startswith("nvjet"):
                kernel_name = "nvjet"

            # Remove more clutter
            kernel_name = kernel_name.replace("void ", "").replace("void", "")
            kernel_name = kernel_name.replace("__amd_rocclr_", "")
            kernel_name = kernel_name.replace("rocprim::detail::", "")
            kernel_name = " ".join(kernel_name.split())
            return kernel_name

        if (
            self.diff_stats_unique_args_summary_df is None
            or self.diff_stats_unique_args_summary_df.empty
        ):
            print(
                "[TraceDiff] diff_stats_unique_args_summary_df is empty. "
                "Run generate_tracediff_report() first."
            )
            return {"trace1": {}, "trace2": {}}

        def get_cpu_op_map(df_agg, df):
            def find_common_name(name1, name2, module_map):
                modules1 = module_map.get(name1, [])
                modules2 = module_map.get(name2, [])

                name1_clean = name1.split("::")[-1]
                name2_clean = name2.split("::")[-1]
                if name1_clean == name2_clean:
                    return name1_clean
                if name1_clean in name2_clean:
                    return name1_clean
                if name2_clean in name1_clean:
                    return name2_clean
                if name1_clean[0:20] == name2_clean[0:20]:
                    return f"{name1_clean}/{name2_clean}"
                if len(modules1) == 1 and len(modules2) == 1:
                    if modules1[0] == modules2[0]:
                        return re.sub(" ", "", modules1[0])
                return None

            def get_rename_map(df):
                result = {
                    str(lca_id): {
                        source: {
                            "name": list(group["cpu_op_name"].unique()),
                            "nn_module_parent": list(
                                group["nn_module_parent"].unique()
                            ),
                        }
                        for source, group in df[
                            df["lowest_common_ancestor_id"] == lca_id
                        ].groupby("source")
                    }
                    for lca_id in df["lowest_common_ancestor_id"].unique()
                }

                module_map = {}
                for cpu_op in df["cpu_op_name"].unique():
                    for source, group in df[df["cpu_op_name"] == cpu_op].groupby(
                        "source"
                    ):
                        module_map[cpu_op] = list(group["nn_module_parent"].unique())
                visited_cpu_op = []
                rename_map = {}
                ##
                for lcaid, mapping in result.items():
                    if "trace1" in mapping and "trace2" in mapping:
                        if all(
                            op in visited_cpu_op for op in mapping["trace1"]["name"]
                        ) and all(
                            op in visited_cpu_op for op in mapping["trace2"]["name"]
                        ):
                            continue
                        visited_cpu_op.extend(mapping["trace1"]["name"])
                        visited_cpu_op.extend(mapping["trace2"]["name"])
                        if len(mapping["trace1"]["name"]) == len(
                            mapping["trace2"]["name"]
                        ):
                            for n1, n2 in zip(
                                mapping["trace1"]["name"], mapping["trace2"]["name"]
                            ):
                                if n1 != n2:
                                    common_name = find_common_name(n1, n2, module_map)
                                    if common_name is not None:
                                        print(f"Renaming: {n1}, {n2} to {common_name}")
                                        rename_map[n2] = common_name
                                        rename_map[n1] = common_name
                                    else:
                                        print(
                                            f"No common name found for {n1} and {n2} under the same LCA, keeping original names."
                                        )
                        else:
                            n1_list = mapping["trace1"]["name"]
                            n1_list_copy = n1_list.copy()
                            n2_list = mapping["trace2"]["name"]
                            for n1 in n1_list:
                                found = 0
                                for n2 in n2_list:
                                    if n1 == n2:
                                        n1_list_copy.remove(n1)
                                        n2_list.remove(n2)
                                        break
                            n1_list = n1_list_copy.copy()
                            for n1 in n1_list_copy:
                                for n2 in n2_list:
                                    common_name = find_common_name(n1, n2, module_map)
                                    if common_name is not None:
                                        print(f"Renaming: {n1}, {n2} to {common_name}")
                                        rename_map[n1] = common_name
                                        rename_map[n2] = common_name
                                        n2_list.remove(n2)
                                        n1_list.remove(n1)
                                        break
                            if len(n1_list) > 0 or len(n2_list) > 0:
                                print(
                                    f"Unmatched for LCA {lcaid}: {n1_list} vs {n2_list}"
                                )
                return rename_map

            def rename_cpu_op(row):
                if row["cpu_op_name"] in rename_map:
                    return rename_map[row["cpu_op_name"]]
                return row["cpu_op_name"]

            def rename_nnmodule(row):
                return re.sub(" ", "", row["nn_module_parent"])

            rename_map = get_rename_map(df)

            df_agg["cpu_op_name"] = df_agg.apply(rename_cpu_op, axis=1)

            df_agg["nn_module_parent"] = df_agg.apply(rename_nnmodule, axis=1)
            ##df_agg['cpu_op_name'] = df_agg['cpu_op_name'].astype(str) + '(' + df_agg['nn_module_parent'].astype(str)+')'
            cpu_op_map = {}
            for cpu_op in df_agg["cpu_op_name"].unique():
                cpu_op_map[cpu_op] = {}
                for source, group in df_agg[df_agg["cpu_op_name"] == cpu_op].groupby(
                    "source"
                ):
                    cpu_op_map[cpu_op][source] = {
                        "kernels": sorted(list(group["name"].unique()))
                    }
                    cpu_op_map[cpu_op][source]["nn_module_parents"] = sorted(
                        list(group["nn_module_parent"].unique())
                    )

            result = {
                kernel_name: {
                    source: {
                        "cpu_op_name": list(group["cpu_op_name"].unique()),
                    }
                    for source, group in df_agg[df_agg["name"] == kernel_name].groupby(
                        "source"
                    )
                }
                for kernel_name in df_agg["name"].unique()
            }
            print("Kernel to CPU op mapping (showing entries with 1:n mapping):")
            for name, mapping in result.items():
                if len(mapping.get("trace1", {}).get("cpu_op_name", [])) > 1:
                    print(
                        name[0:30],
                        "\t",
                        mapping.get("trace1", {}).get("cpu_op_name", []),
                    )
                if len(mapping.get("trace2", {}).get("cpu_op_name", [])) > 1:
                    print(
                        name[0:30],
                        "\t",
                        mapping.get("trace2", {}).get("cpu_op_name", []),
                    )
            return cpu_op_map

        df_agg = self.diff_stats_unique_args_summary_df
        df = self.diff_stats_df



        def _agg_cleaned_kernels(series):
            """Apply cleanKernelNames to each kernel name in the Series."""
            return sorted(
                set(
                    cleanKernelNames(str(k))
                    for k in series.dropna()
                    if k and str(k).strip()
                )
            )

        cpu_op_map_trace1 = (
            df_agg[df_agg["source"] == "trace1"]
            .groupby(["cpu_op_name"])
            .agg({"name": _agg_cleaned_kernels})
            .sort_index()
        )
        cpu_op_map_trace2 = (
            df_agg[df_agg["source"] == "trace2"]
            .groupby(["cpu_op_name"])
            .agg({"name": _agg_cleaned_kernels})
            .sort_index()
        )
        cpu_op_map = get_cpu_op_map(df_agg, df)

        if self.identical_traces:
            for cpu_op, mapping in cpu_op_map.items():
                cpu_op_map[cpu_op] = mapping["trace1"]

        self.cpu_op_map = cpu_op_map
        self.cpu_op_map_trace1 = cpu_op_map_trace1
        self.cpu_op_map_trace2 = cpu_op_map_trace2

    def generate_tracediff_report(self):
        """
        Generate all TraceDiff output DataFrames and update the object variables.
        This does NOT write any files. Use print_tracediff_report_files to save outputs.
        """
        self.generate_diff_stats()
        self.get_df_diff_stats_unique_args()
        self.get_cpu_op_to_kernels_json()

        if self.identical_traces:
            df = self.diff_stats_df
            df = df[~(df["source"] == "trace2")]
            df = df.drop(columns=["source"])
            self.diff_stats_df = df

            df_agg = self.diff_stats_unique_args_summary_df
            df_agg = df_agg[~(df_agg["source"] == "trace2")]
            df_agg = df_agg.drop(columns=["source"])
            self.diff_stats_unique_args_summary_df = df_agg

    def print_tracediff_report_files(
        self, output_folder="rprt_diff", prune_non_gpu=False
    ):
        """
        Write all TraceDiff output reports to files in the specified output folder (default 'rprt_diff').
        Output file names are:
            - merged_tree_output.txt
            - diff_stats.csv
            - diff_stats_summary.csv
            - cpu_op_map_trace1.json
            - cpu_op_map_trace2.json
            - cpu_op_map.json
        """

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        merged_tree_file = os.path.join(output_folder, "merged_tree_output.txt")
        diff_stats_file = os.path.join(output_folder, "diff_stats.csv")
        diff_stats_unique_args_summary_file = os.path.join(
            output_folder, "diff_stats_unique_args_summary.csv"
        )
        self.print_merged_tree(
            output_file=merged_tree_file, prune_non_gpu=prune_non_gpu
        )
        if self.diff_stats_df is not None and not self.diff_stats_df.empty:
            self.diff_stats_df.to_csv(diff_stats_file, index=False)
        else:
            print(
                f"[TraceDiff] diff_stats_df is empty. Run generate_tracediff_report() first."
            )
        if (
            self.diff_stats_unique_args_summary_df is not None
            and not self.diff_stats_unique_args_summary_df.empty
        ):
            self.diff_stats_unique_args_summary_df.to_csv(
                diff_stats_unique_args_summary_file, index=False
            )
        else:
            print(
                f"[TraceDiff] diff_stats_unique_args_summary_df is empty. Run generate_tracediff_report() first."
            )
        if self.cpu_op_map_trace1 is not None:
            with open(
                os.path.join(output_folder, "cpu_op_map_trace1.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.cpu_op_map_trace1.to_dict()["name"],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            print(
                f"[TraceDiff] cpu_op_map_trace1 is empty. Run get_cpu_op_to_kernels_json() first."
            )
        if self.cpu_op_map_trace2 is not None:
            with open(
                os.path.join(output_folder, "cpu_op_map_trace2.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.cpu_op_map_trace2.to_dict()["name"],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        else:
            print(
                f"[TraceDiff] cpu_op_map_trace2 is empty. Run get_cpu_op_to_kernels_json() first."
            )
        if self.cpu_op_map is not None:
            with open(
                os.path.join(output_folder, "cpu_op_map.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    self.cpu_op_map,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
