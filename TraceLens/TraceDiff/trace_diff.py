###############################################################################
# Copyright (c) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import os
import re

import pandas as pd

from ..Trace2Tree.trace_to_tree import TraceToTree
from .cpu_op_map import (
    _build_rename_map,
    _cpu_op_kernel_map,
    _print_one_to_many,
)
from .diff_stats import (
    _DiffStatsCollector,
    _aggregate_unique_args,
    _order_summary_columns,
    _sort_summary_rows,
)
from .merge import (
    _MergeBuilder,
    _descend_single_wrapper_child,
    _reparent_under,
)
from .util import (
    _NAME,
    _UID,
)
from ..util import normalize_name_for_comparison


def _write_json(path, payload):
    """Write one report payload as indented JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
        self._uid1_to_merged_id = None
        self._uid2_to_merged_id = None

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
        if self._uid1_to_merged_id is None:
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
        self._uid1_to_merged_id = None
        self._uid2_to_merged_id = None

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

        name = node.get(_NAME)
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
        root = tree.get_UID2event(start_uid)
        while root.get("parent") is not None:
            root = tree.get_UID2event(root["parent"])

        current = _descend_single_wrapper_child(tree, root)
        if current is not root:
            _reparent_under(tree, root, current.get("children", []))
        return current.get(_UID)

    def wagner_fischer(self, items1, items2, wf_cache, strip_details=False):
        """
        Wagner-Fischer algorithm that works with any items and name lookup functions.

        Args:
            items1: List of items
            items2: List of items
            wf_cache: Dictionary for caching results
            strip_details: Passed to _normalize_name_for_comparison to enable
                aggressive matching that ignores function names in python_function events.

        Returns:
            List of operations: [("match", i, j), ("delete", i, None), ("insert", None, j), ...]
        """
        # Pre-compute names for cache key
        names1 = [
            normalize_name_for_comparison(self._get_op_name(item, 1), strip_details)
            for item in items1
        ]
        names2 = [
            normalize_name_for_comparison(self._get_op_name(item, 2), strip_details)
            for item in items2
        ]

        # Check cache
        cache_key = (tuple(items1), tuple(items2), strip_details)
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

        builder = _MergeBuilder(self)
        roots1 = self._unique_top_level_roots(tree1)
        roots2 = self._unique_top_level_roots(tree2)

        print(
            f"[TraceDiff] Found {len(roots1)} root trees in trace1, "
            f"{len(roots2)} in trace2"
        )

        # Match roots by position: i-th root in trace1 pairs with i-th in trace2
        merged_root_ids = []
        n = max(len(roots1), len(roots2))
        for i in range(n):
            uid1 = roots1[i] if i < len(roots1) else None
            uid2 = roots2[i] if i < len(roots2) else None
            merged_root_ids.append(builder.merge(uid1, uid2))

        self.merged_tree = (builder.events, merged_root_ids)
        return self.merged_tree

    def _unique_top_level_roots(self, tree):
        """Top-level roots of a tree, de-duplicated, in first-seen order."""
        seen, roots = set(), []
        for crn in tree.cpu_root_nodes:
            root = self._get_top_level_root(tree, crn)
            if root not in seen:
                seen.add(root)
                roots.append(root)
        return roots

    def _format_merged_subtree(
        self, merged_id, merged_id_to_event, prefix="", is_last=True
    ):
        """Yield formatted lines for a merged subtree. Shared by print_merged_subtree and print_merged_tree."""
        node = merged_id_to_event[merged_id]
        merge_type = node["merged_type"]
        name1 = self._get_op_name(node["uid1"], 1) if node["uid1"] is not None else None
        name2 = self._get_op_name(node["uid2"], 2) if node["uid2"] is not None else None
        connector = "└── " if is_last else "├── "
        if merge_type == "combined":
            if name1 == name2 and name1 is not None:
                yield f"{prefix}{connector}{name1}"
            else:
                yield f"{prefix}{connector}{merge_type}: {name1} | {name2}"
        elif merge_type == "trace1":
            yield f"{prefix}{connector}>> {merge_type}: {name1}"
        elif merge_type == "trace2":
            yield f"{prefix}{connector}<< {merge_type}: {name2}"
        else:
            yield f"{prefix}{connector}{merge_type}: {name1} | {name2}"
        # Sort children by merge_type order: combined, trace1, trace2
        children = [merged_id_to_event[cid] for cid in node["children"]]
        sorted_children = (
            [c["merged_id"] for c in children if c["merged_type"] == "combined"]
            + [c["merged_id"] for c in children if c["merged_type"] == "trace1"]
            + [c["merged_id"] for c in children if c["merged_type"] == "trace2"]
        )
        child_count = len(sorted_children)
        for i, cid in enumerate(sorted_children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            yield from self._format_merged_subtree(
                cid, merged_id_to_event, new_prefix, is_last=(i == child_count - 1)
            )

    def print_merged_subtree(self, uid_tree1=None, uid_tree2=None):
        if uid_tree1 is None and uid_tree2 is None:
            raise ValueError("At least one of uid_tree1 or uid_tree2 must be provided.")
        if self.merged_tree is None:
            raise ValueError(
                "merged_tree is not initialized. Call merge_trees() first."
            )
        merged_id_to_event = self._get_merged_id_to_event()
        uid1_to_merged_id, uid2_to_merged_id = self._get_uid_to_merged_id_maps()

        merged_id = None
        if uid_tree1 is not None:
            merged_id = uid1_to_merged_id.get(uid_tree1)
        elif uid_tree2 is not None:
            merged_id = uid2_to_merged_id.get(uid_tree2)

        if merged_id is None:
            raise ValueError("Could not find merged node for the given UID.")

        for line in self._format_merged_subtree(merged_id, merged_id_to_event):
            print(line)

    def print_merged_tree(self, output_file, prune_non_gpu=False):
        if self.merged_tree is None:
            raise ValueError(
                "merged_tree is not initialized. Call merge_trees() first."
            )
        _, merged_root_ids = self.merged_tree
        merged_id_to_event = self._get_merged_id_to_event()

        def subtree_has_gpu(merged_id: int) -> bool:
            node = merged_id_to_event[merged_id]
            uid1 = node["uid1"]
            uid2 = node["uid2"]
            node1 = self.baseline.get_UID2event(uid1) if uid1 is not None else None
            node2 = self.variant.get_UID2event(uid2) if uid2 is not None else None
            if node1 and not node1.get("non_gpu_path", False):
                return True
            if node2 and not node2.get("non_gpu_path", False):
                return True
            return False

        output_lines = []
        for i, root_id in enumerate(merged_root_ids):
            if prune_non_gpu and not subtree_has_gpu(root_id):
                continue
            output_lines.extend(
                self._format_merged_subtree(
                    root_id,
                    merged_id_to_event,
                    prefix="",
                    is_last=(i == len(merged_root_ids) - 1),
                )
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
        _, merged_root_ids = self.merged_tree
        rows = _DiffStatsCollector(self).collect(merged_root_ids)

        df = pd.DataFrame(rows)

        if df.empty:
            print("[TraceDiff] No GPU events found in either trace")
            self.identical_traces = True
            self.diff_stats_df = df
            return df

        df["busy_time"] = df["busy_time"].round(3)
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
        df_filtered = df_filtered.drop(
            columns=["lowest_common_ancestor_id", "gpu_op_uid"]
        )

        metric_columns = ["kernel_time"]
        grouping_cols = [c for c in df_filtered.columns if c not in metric_columns]

        agg_dict = {mcol: list(set(agg_metrics) | {"sum"}) for mcol in metric_columns}
        for col in grouping_cols:
            agg_dict[col] = "first"  # keep first occurrence of each argument column

        df_agg = _aggregate_unique_args(df_filtered, grouping_cols, agg_dict)
        df_agg = _order_summary_columns(
            df_agg, grouping_cols, metric_columns, agg_metrics
        )
        df_agg = _sort_summary_rows(df_agg)

        self.diff_stats_unique_args_summary_df = df_agg
        return df_agg

    def get_cpu_op_to_kernels_json(self) -> None:
        """
        Build CPU-op-to-kernel mappings for both traces and store them on this
        object. Uses 'name' (kernel) and 'cpu_op_name' from
        diff_stats_unique_args_summary_df.
        """

        if (
            self.diff_stats_unique_args_summary_df is None
            or self.diff_stats_unique_args_summary_df.empty
        ):
            print(
                "[TraceDiff] diff_stats_unique_args_summary_df is empty. "
                "Run generate_tracediff_report() first."
            )
            return

        df_agg = self.diff_stats_unique_args_summary_df
        df = self.diff_stats_df

        cpu_op_map_trace1 = (
            df_agg[df_agg["source"] == "trace1"]
            .groupby(["cpu_op_name"])
            .agg({"name": lambda x: sorted(set(x))})
            .sort_index()
        )
        cpu_op_map_trace2 = (
            df_agg[df_agg["source"] == "trace2"]
            .groupby(["cpu_op_name"])
            .agg({"name": lambda x: sorted(set(x))})
            .sort_index()
        )

        rename_map = _build_rename_map(df)
        df_agg["cpu_op_name"] = df_agg["cpu_op_name"].map(
            lambda name: rename_map.get(name, name)
        )
        df_agg["nn_module_parent"] = df_agg["nn_module_parent"].map(
            lambda parent: re.sub(" ", "", parent)
        )
        cpu_op_map = _cpu_op_kernel_map(df_agg)
        _print_one_to_many(df_agg)

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

        os.makedirs(output_folder, exist_ok=True)
        self.print_merged_tree(
            output_file=os.path.join(output_folder, "merged_tree_output.txt"),
            prune_non_gpu=prune_non_gpu,
        )

        for filename, attr in (
            ("diff_stats.csv", "diff_stats_df"),
            ("diff_stats_unique_args_summary.csv", "diff_stats_unique_args_summary_df"),
        ):
            df = getattr(self, attr)
            if df is not None and not df.empty:
                df.to_csv(os.path.join(output_folder, filename), index=False)
            else:
                print(
                    f"[TraceDiff] {attr} is empty. Run generate_tracediff_report() first."
                )

        for filename, attr in (
            ("cpu_op_map_trace1.json", "cpu_op_map_trace1"),
            ("cpu_op_map_trace2.json", "cpu_op_map_trace2"),
        ):
            frame = getattr(self, attr)
            if frame is not None:
                _write_json(
                    os.path.join(output_folder, filename), frame.to_dict()["name"]
                )
            else:
                print(
                    f"[TraceDiff] {attr} is empty. Run get_cpu_op_to_kernels_json() first."
                )

        if self.cpu_op_map is not None:
            _write_json(os.path.join(output_folder, "cpu_op_map.json"), self.cpu_op_map)
