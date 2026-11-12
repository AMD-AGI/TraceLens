###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-kernel diff rows collected from the merged tree, and their summary."""

import re

from ..TreePerf.gpu_event_analyser import GPUEventAnalyser
from .util import (
    _DUR,
    _NAME,
    _TRACELENS_DEBUG,
    _UID,
    _get_node_arg,
    _is_gpu_path,
    _is_kernel,
)


def _aggregate_unique_args(df_filtered, grouping_cols, agg_dict):
    """Group by every argument column, falling back to string keys when unhashable."""
    try:
        df_agg = df_filtered.groupby(grouping_cols, dropna=False).agg(agg_dict)
        df_agg["operation_count"] = df_filtered.groupby(
            grouping_cols, dropna=False
        ).size()
        return df_agg
    except TypeError:
        # Lists and dicts are unhashable, so group on their string repr instead.
        str_cols = [f"{col}_str_repr" for col in grouping_cols]
        df_temp = df_filtered.copy()
        for col, str_col in zip(grouping_cols, str_cols):
            df_temp[str_col] = df_temp[col].astype(str)
        df_agg = df_temp.groupby(str_cols, dropna=False).agg(agg_dict)
        df_agg["operation_count"] = df_temp.groupby(str_cols, dropna=False).size()
        return df_agg


def _order_summary_columns(df_agg, grouping_cols, metric_columns, agg_metrics):
    """Argument columns first, then aggregated metrics, with the count second."""
    df_agg.columns = ["_".join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index(drop=True)
    df_agg = df_agg.rename(
        columns={
            f"{col}_first": col
            for col in grouping_cols
            if f"{col}_first" in df_agg.columns
        }
    )

    metric_cols = []
    for metric in metric_columns:
        for agg in agg_metrics + ([] if "sum" in agg_metrics else ["sum"]):
            col_name = f"{metric}_{agg}"
            if col_name in df_agg.columns:
                metric_cols.append(col_name)
    metric_cols = list(dict.fromkeys(metric_cols))  # remove duplicates
    other_cols = [
        col for col in df_agg.columns if col not in grouping_cols + metric_cols
    ]
    df_agg = df_agg[grouping_cols + metric_cols + other_cols]
    df_agg = df_agg.rename(columns={"operation_count_": "operation_count"})
    cols = list(df_agg.columns)
    cols.remove("operation_count")
    cols.insert(1, "operation_count")
    return df_agg[cols]


def _sort_summary_rows(df_agg):
    """Sort by trace1 kernel time, with stable tie-breakers so row order is
    deterministic across Python and pandas versions."""
    sort_col = "kernel_time_sum"
    if sort_col not in df_agg.columns:
        return df_agg
    for col in reversed([c for c in df_agg.columns if c != sort_col]):
        df_agg = df_agg.sort_values(
            by=col,
            kind="mergesort",
            na_position="last",
            key=lambda s: s.astype(str),
        )
    return df_agg.sort_values(
        by=sort_col,
        ascending=False,
        kind="mergesort",
        ignore_index=True,
    )


class _DiffStatsCollector:
    """Collects per-kernel diff rows by walking the merged tree."""

    def __init__(self, diff):
        self.diff = diff
        self.merged_id_to_event = diff._get_merged_id_to_event()
        self.uid2node1 = diff._get_baseline_uid2node()
        self.uid2node2 = diff._get_variant_uid2node()
        self.rows = []
        self.visited = set()

    def collect(self, merged_root_ids):
        """Traverse every root, returning the collected row dicts."""
        for root_index, root_id in enumerate(merged_root_ids):
            self._traverse(root_id, None, root_index)
        return self.rows

    def _side(self, tree_num):
        """(uid key, uid->node map, tree, source label) for one of the two traces."""
        if tree_num == 1:
            return "uid1", self.uid2node1, self.diff.baseline, "trace1"
        return "uid2", self.uid2node2, self.diff.variant, "trace2"

    def _traverse(self, merged_id, combined_parent_node, root_index=0):
        if merged_id in self.visited:
            return
        node = self.merged_id_to_event[merged_id]
        mt = node["merged_type"]
        if mt == "trace1":
            self._single_side(node, merged_id, combined_parent_node, root_index, 1)
            return
        if mt == "trace2":
            self._single_side(node, merged_id, combined_parent_node, root_index, 2)
            return

        event1 = self.uid2node1.get(node["uid1"])
        event2 = self.uid2node2.get(node["uid2"])
        self._combined(node, merged_id, event1, event2, root_index)

        # Only traverse children if either trace is on a GPU path
        if _is_gpu_path(event1) or _is_gpu_path(event2):
            for cid in node["children"]:
                self._traverse(cid, node, root_index)

    def _single_side(self, node, merged_id, combined_parent_node, root_index, tree_num):
        """Emit rows for a node present in only one of the two traces."""
        uid_key, uid2node, tree_obj, source = self._side(tree_num)
        event = uid2node.get(node[uid_key])
        if event and _is_gpu_path(event):
            lca_name, lca_id, lca_children_t1, lca_children_t2 = (
                self._resolve_lca_from_parent(combined_parent_node, uid_key, tree_num)
            )
            gpu_event_uids = event.get("gpu_events", [])
            self.rows.extend(
                self._gpu_rows(
                    gpu_event_uids,
                    tree_obj,
                    uid2node,
                    tree_num,
                    source,
                    lca_name,
                    lca_id,
                    self._busy_time(gpu_event_uids, uid2node),
                    lca_children_t1,
                    lca_children_t2,
                    root_index,
                )
            )
        self.visited.add(merged_id)

    def _combined(self, node, merged_id, event1, event2, root_index):
        """Emit rows for a node present in both traces."""
        if not (event1 and event2 and _is_gpu_path(event1) and _is_gpu_path(event2)):
            return
        children = [self.merged_id_to_event[cid] for cid in node["children"]]
        non_combined = [c for c in children if c["merged_type"] != "combined"]
        gpu1 = [
            c for c in non_combined if _is_gpu_path(self.uid2node1.get(c.get("uid1")))
        ]
        gpu2 = [
            c for c in non_combined if _is_gpu_path(self.uid2node2.get(c.get("uid2")))
        ]
        if not (gpu1 or gpu2) and not (_is_kernel(event1) and _is_kernel(event2)):
            return

        lca_name = self._lca_name(node)
        lca_children_t1 = self._gpu_path_child_names(node["uid1"], self.uid2node1, 1)
        lca_children_t2 = self._gpu_path_child_names(node["uid2"], self.uid2node2, 2)
        per_side = (
            (1, self._gpu_event_uids(gpu1, "uid1", self.uid2node1, event1)),
            (2, self._gpu_event_uids(gpu2, "uid2", self.uid2node2, event2)),
        )
        for tree_num, gpu_event_uids in per_side:
            _, uid2node, tree_obj, source = self._side(tree_num)
            self.rows.extend(
                self._gpu_rows(
                    gpu_event_uids,
                    tree_obj,
                    uid2node,
                    tree_num,
                    source,
                    lca_name,
                    node["merged_id"],
                    self._busy_time(gpu_event_uids, uid2node),
                    lca_children_t1,
                    lca_children_t2,
                    root_index,
                )
            )

        self.visited.add(merged_id)
        self.visited.update(child.get("merged_id") for child in gpu1 + gpu2)

    def _lca_name(self, node):
        """LCA display name; the two traces' names are joined when they differ."""
        name1 = re.sub(r"\(\d+\)", "", self.diff._get_op_name(node["uid1"], 1))
        name2 = re.sub(r"\(\d+\)", "", self.diff._get_op_name(node["uid2"], 2))
        return name1 if name1 == name2 else f"{name1} | {name2}"

    @staticmethod
    def _gpu_event_uids(gpu_children, uid_key, uid2node, event):
        """GPU event UIDs contributed by one side's non-combined children."""
        uids = []
        for child in gpu_children:
            child_node = uid2node.get(child.get(uid_key))
            uids.extend(child_node.get("gpu_events", []))
            if _is_kernel(child_node):
                uids.append(child_node[_UID])
        if _is_kernel(event):
            uids.append(event[_UID])
        return uids

    def _resolve_lca_from_parent(self, combined_parent_node, uid_key, tree_num):
        """Resolve LCA info from a combined parent node for trace-only branches."""
        if combined_parent_node is None:
            return None, None, "", ""
        lca_id = combined_parent_node.get("merged_id")
        lca = self.merged_id_to_event[lca_id]
        lca_name = re.sub(
            r"\(\d+\)", "", self.diff._get_op_name(lca[uid_key], tree_num)
        )
        return (
            lca_name,
            lca_id,
            self._gpu_path_child_names(lca["uid1"], self.uid2node1, 1),
            self._gpu_path_child_names(lca["uid2"], self.uid2node2, 2),
        )

    def _gpu_path_child_names(self, uid, uid2node, tree_num):
        """Return semicolon-joined names of GPU-path children of the given node."""
        if uid is None:
            return ""
        node = uid2node.get(uid)
        if node is None:
            return ""
        names = []
        for cuid in node.get("children", []):
            cn = uid2node.get(cuid)
            if cn and _is_gpu_path(cn):
                names.append(self.diff._get_op_name(cuid, tree_num) or "")
        return "; ".join(names)

    @staticmethod
    def _busy_time(gpu_event_uids, uid2node):
        gpu_events = [
            uid2node.get(uid) for uid in gpu_event_uids if uid2node.get(uid) is not None
        ]
        if not gpu_events:
            return 0.0
        return GPUEventAnalyser(gpu_events).compute_metrics()["busy_time"]

    @staticmethod
    def _callstack(gpu_event, uid2node):
        """Walk from gpu_event up to root, collecting op names root-first."""
        stack = []
        uid = gpu_event.get("parent")
        while uid is not None:
            n = uid2node.get(uid)
            if n is None:
                break
            name = n.get(_NAME, "")
            if name:
                stack.append(name)
            uid = n.get("parent")
        stack.reverse()
        return ", ".join(stack)

    def _gpu_rows(
        self,
        gpu_event_uids,
        tree_obj,
        uid2node,
        tree_num,
        source,
        lca_name,
        lca_id,
        lca_busy_time,
        lca_children_t1,
        lca_children_t2,
        root_index,
    ):
        """Build row dicts for each GPU event. Used by combined and trace-only branches."""
        result = []
        for gpu_uid in gpu_event_uids:
            gpu_event = uid2node.get(gpu_uid)
            if gpu_event is None:
                continue
            parent_uid = gpu_event.get("parent")
            parent_node = uid2node.get(parent_uid)
            while parent_node is not None:
                if tree_obj.event_to_category(parent_node) == "cpu_op":
                    break
                parent_uid = parent_node.get("parent")
                parent_node = uid2node.get(parent_uid)
            if parent_node is None:
                continue
            row = {
                "name": gpu_event[_NAME],
                "cpu_op_name": self.diff._get_op_name(parent_uid, tree_num),
                "cpu_op_uid": parent_uid,
                "source": source,
                "Input Dims": _get_node_arg(parent_node, "Input Dims"),
                "Input Strides": _get_node_arg(parent_node, "Input Strides"),
                "Input type": _get_node_arg(parent_node, "Input type"),
                "Concrete Inputs": _get_node_arg(parent_node, "Concrete Inputs"),
                "kernel_time": gpu_event.get(_DUR, 0),
                "busy_time": lca_busy_time,
                "lowest_common_ancestor_name": lca_name,
                "lowest_common_ancestor_id": lca_id,
                "gpu_op_uid": gpu_uid,
                "nn_module_stack": ";".join(
                    str(x) for x in parent_node.get("nn_module_stack", [])
                ),
                "nn_module_parent": (parent_node.get("nn_module_stack") or [""])[-1],
            }
            if _TRACELENS_DEBUG:
                row["lca_children_t1"] = lca_children_t1
                row["lca_children_t2"] = lca_children_t2
                row["root_index"] = root_index
                row["callstack"] = self._callstack(gpu_event, uid2node)
            result.append(row)
        return result
