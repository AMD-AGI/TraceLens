from typing import Any, Callable, Dict

import pandas as pd

import TraceLens.util
from TraceLens import TraceToTree


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
        # Automatically merge trees and initialize UID map
        self.merge_trees()

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

    def _add_subtree_to_pod_recursive(
        self, node: Dict[str, Any], pod: set, tree: TraceToTree
    ) -> None:
        name = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name, "Unknown")
        cat = tree.event_to_category(node)
        uid = {node[TraceLens.util.TraceEventUtils.TraceKeys.UID]}

        children = tree.get_children_events(node)
        pod.update(uid)

        for i, child in enumerate(children):
            self._add_subtree_to_pod_recursive(child, pod, tree)

    def add_to_pod(self, node: Dict[str, Any], pod: set, tree: TraceToTree) -> None:
        """
        Recursively adds the subtree rooted at the given node to the set of points of differences (PODs).

        Args:
            node (Dict[str, Any]): The current node in the trace tree.
            pod (set): The set to which PODs will be added.
            tree (TraceToTree): The trace tree containing the events.
        """
        if not isinstance(node, dict):
            return

        self._add_subtree_to_pod_recursive(node, pod, tree)

    def calculate_diff_boundaries(self):
        """
        Compare two trees and identify the boundaries of differences between them using recursive Wagner-Fischer and DFS, matching the reference tree.py algorithm.
        Returns:
            - db1 (list[dict]): List of difference boundaries in tree1.
            - db2 (list[dict]): List of difference boundaries in tree2.
            - pod1 (set): Set of points of differences in tree1.
            - pod2 (set): Set of points of differences in tree2.
        """
        tree1 = self.baseline
        tree2 = self.variant

        def get_name(node):
            return node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name)

        def get_children(tree, node):
            return tree.get_children_events(node)

        def add_to_pod(node, pod, tree):
            # Add node and all its descendants to pod
            pod.add(node.get(TraceLens.util.TraceEventUtils.TraceKeys.UID))
            for child in get_children(tree, node):
                add_to_pod(child, pod, tree)

        def wagner_fischer(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if get_name(tree1.get_UID2event(seq1[i - 1])) == get_name(
                        tree2.get_UID2event(seq2[j - 1])
                    ):
                        cost = 0
                    else:
                        cost = 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + cost,
                    )
            # Backtrack to get the operations
            i, j = m, n
            ops = []
            while i > 0 or j > 0:
                if (
                    i > 0
                    and j > 0
                    and get_name(tree1.get_UID2event(seq1[i - 1]))
                    == get_name(tree2.get_UID2event(seq2[j - 1]))
                ):
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
            return ops

        def dfs(node1, node2):
            # If either node is already a POD, skip
            uid1 = node1.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
            uid2 = node2.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
            if uid1 in self.pod1 or uid2 in self.pod2:
                return

            name1 = get_name(node1)
            name2 = get_name(node2)
            if name1 != name2:
                self.db1.append(node1)
                self.db2.append(node2)
                add_to_pod(node1, self.pod1, tree1)
                add_to_pod(node2, self.pod2, tree2)
                return

            children1 = sorted(
                get_children(tree1, node1), key=lambda child: child.get("ts", 0)
            )
            children2 = sorted(
                get_children(tree2, node2), key=lambda child: child.get("ts", 0)
            )
            seq1 = [
                child.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
                for child in children1
            ]
            seq2 = [
                child.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
                for child in children2
            ]
            ops = wagner_fischer(seq1, seq2)
            idx1, idx2 = 0, 0
            for op, i, j in ops:
                if op == "match":
                    child1 = tree1.get_UID2event(seq1[i])
                    child2 = tree2.get_UID2event(seq2[j])
                    dfs(child1, child2)
                    idx1 += 1
                    idx2 += 1
                elif op == "delete":
                    child1 = tree1.get_UID2event(seq1[i])
                    self.db1.append(child1)
                    add_to_pod(child1, self.pod1, tree1)
                    idx1 += 1
                elif op == "insert":
                    child2 = tree2.get_UID2event(seq2[j])
                    self.db2.append(child2)
                    add_to_pod(child2, self.pod2, tree2)
                    idx2 += 1

        # Start DFS from the root nodes
        if not tree1.cpu_root_nodes or not tree2.cpu_root_nodes:
            raise ValueError(
                "Both trees must have at least one root node in cpu_root_nodes."
            )
        roots1 = sorted(
            tree1.cpu_root_nodes, key=lambda uid: tree1.get_UID2event(uid).get("ts", 0)
        )
        roots2 = sorted(
            tree2.cpu_root_nodes, key=lambda uid: tree2.get_UID2event(uid).get("ts", 0)
        )
        ops = wagner_fischer(roots1, roots2)
        for op, i, j in ops:
            if op == "match":
                node1 = tree1.get_UID2event(roots1[i])
                node2 = tree2.get_UID2event(roots2[j])
                dfs(node1, node2)
            elif op == "delete":
                node1 = tree1.get_UID2event(roots1[i])
                self.db1.append(node1)
                add_to_pod(node1, self.pod1, tree1)
            elif op == "insert":
                node2 = tree2.get_UID2event(roots2[j])
                self.db2.append(node2)
                add_to_pod(node2, self.pod2, tree2)
        return self.db1, self.db2, self.pod1, self.pod2

    def merge_trees(self):
        """
        Merges the two trees using the PODs from get_diff_boundaries, inspired by merge_tree_from_pod, but returns a flat list of merged event dicts.
        Each merged event has a unique merged_id, children as merged_id references, and root merged_ids. Compatible with TraceToTree format.
        Returns: (merged_events, merged_root_ids)
        """

        # Set the PODs and diff_boundaries
        self.calculate_diff_boundaries()

        # Helper to create a merged event
        def make_event(merged_id, uid1, uid2, merged_type, children):
            return {
                "merged_id": merged_id,
                "uid1": uid1,
                "uid2": uid2,
                "merged_type": merged_type,
                "children": children,  # list of merged_id
            }

        # Build lookup for UID to node for both trees
        baseline_uid2node = {}
        variant_uid2node = {}
        for node in self.baseline.events:
            if isinstance(node, dict):
                baseline_uid2node[
                    node.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
                ] = node
        for node in self.variant.events:
            if isinstance(node, dict):
                variant_uid2node[
                    node.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
                ] = node

        # Recursive merge using PODs, but build flat event list
        merged_events = []
        merged_id_counter = [0]
        uid_pair_to_merged_id = {}

        def safe_children(tree, uid):
            node = tree.get(uid, None)
            if node is None or not isinstance(node, dict):
                return []
            return node.get("children", [])

        def get_name_by_uid(tree, uid):
            node = tree.get(uid)
            return (
                node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name)
                if node
                else None
            )

        def merge_from_pod(uid1, uid2, parent_merged_id=None):
            key = (uid1, uid2)
            if key in uid_pair_to_merged_id:
                return uid_pair_to_merged_id[key]
            merged_id = merged_id_counter[0]
            merged_id_counter[0] += 1
            uid_pair_to_merged_id[key] = merged_id
            # Build merged_uid_map for combined nodes
            if uid1 and uid2:
                merged_type = "combined"
                # Map both directions
                self.merged_uid_map[(1, uid1)] = uid2
                self.merged_uid_map[(2, uid2)] = uid1
            elif uid1:
                merged_type = "trace1"
                self.merged_uid_map[(1, uid1)] = -1
            else:
                merged_type = "trace2"
                self.merged_uid_map[(2, uid2)] = -1
            children1 = safe_children(baseline_uid2node, uid1)
            children2 = safe_children(variant_uid2node, uid2)
            # Wagner-Fischer to align children
            ops = []
            if children1 or children2:
                m, n = len(children1), len(children2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(m + 1):
                    dp[i][0] = i
                for j in range(n + 1):
                    dp[0][j] = j
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        name1 = get_name_by_uid(baseline_uid2node, children1[i - 1])
                        name2 = get_name_by_uid(variant_uid2node, children2[j - 1])
                        cost = 0 if name1 == name2 else 1
                        dp[i][j] = min(
                            dp[i - 1][j] + 1,
                            dp[i][j - 1] + 1,
                            dp[i - 1][j - 1] + cost,
                        )
                i, j = m, n
                while i > 0 or j > 0:
                    if (
                        i > 0
                        and j > 0
                        and get_name_by_uid(baseline_uid2node, children1[i - 1])
                        == get_name_by_uid(variant_uid2node, children2[j - 1])
                    ):
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
            child_merged_ids = []
            for op, i, j in ops:
                if op == "match":
                    child_uid1 = children1[i]
                    child_uid2 = children2[j]
                    child_merged_id = merge_from_pod(child_uid1, child_uid2, merged_id)
                    child_merged_ids.append(child_merged_id)
                elif op == "delete":
                    child_uid1 = children1[i]
                    child_merged_id = merge_from_pod(child_uid1, None, merged_id)
                    child_merged_ids.append(child_merged_id)
                elif op == "insert":
                    child_uid2 = children2[j]
                    child_merged_id = merge_from_pod(None, child_uid2, merged_id)
                    child_merged_ids.append(child_merged_id)
            event = make_event(merged_id, uid1, uid2, merged_type, child_merged_ids)
            merged_events.append(event)
            return merged_id

        # Find root UIDs
        roots1 = list(self.baseline.cpu_root_nodes)
        roots2 = list(self.variant.cpu_root_nodes)
        merged_root_ids = []

        # Sort roots by ts for deterministic order
        roots1 = sorted(
            roots1, key=lambda uid: baseline_uid2node.get(uid, {}).get("ts", 0)
        )
        roots2 = sorted(
            roots2, key=lambda uid: variant_uid2node.get(uid, {}).get("ts", 0)
        )

        i, j = 0, 0
        while i < len(roots1) and j < len(roots2):
            uid1 = roots1[i]
            uid2 = roots2[j]
            in_pod1 = uid1 in self.pod1
            in_pod2 = uid2 in self.pod2
            if in_pod1 and not in_pod2:
                merged_root_ids.append(merge_from_pod(uid1, None))
                i += 1
            elif in_pod2 and not in_pod1:
                merged_root_ids.append(merge_from_pod(None, uid2))
                j += 1
            elif in_pod1 and in_pod2:
                # Both are PODs, treat as separate
                merged_root_ids.append(merge_from_pod(uid1, None))
                merged_root_ids.append(merge_from_pod(None, uid2))
                i += 1
                j += 1
            else:
                # Both are not PODs, merge as combined
                merged_root_ids.append(merge_from_pod(uid1, uid2))
                i += 1
                j += 1

        # Handle remaining roots in either list
        while i < len(roots1):
            merged_root_ids.append(merge_from_pod(roots1[i], None))
            i += 1
        while j < len(roots2):
            merged_root_ids.append(merge_from_pod(None, roots2[j]))
            j += 1

        self.merged_tree = (merged_events, merged_root_ids)
        return self.merged_tree

    def get_corresponding_uid(self, tree_num, uid):
        """
        Given a tree number (1 or 2) and a UID, return the corresponding UID from the other tree if combined, else -1.
        """
        return self.merged_uid_map.get((tree_num, uid), -1)

    def print_merged_tree(self, output_file):
        if self.merged_tree is None:
            raise ValueError(
                "merged_tree is not initialized. Call merge_trees() first."
            )
        (merged_events, merged_root_ids) = self.merged_tree
        output_lines = []
        merged_id_to_event = {event["merged_id"]: event for event in merged_events}

        def get_op_name(uid, tree_uid2node):
            node = tree_uid2node.get(uid)
            if node is None:
                return None
            name = node.get("name") if "name" in node else node.get("Name")
            if name is None:
                try:
                    import TraceLens.util

                    name = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name)
                except Exception:
                    pass
            return name if name else str(uid)

        baseline_uid2node = {
            event.get("UID"): event
            for event in getattr(self.baseline, "events", [])
            if isinstance(event, dict)
        }
        variant_uid2node = {
            event.get("UID"): event
            for event in getattr(self.variant, "events", [])
            if isinstance(event, dict)
        }

        def print_merged_tree_to_lines(merged_id, prefix="", is_last=True):
            node = merged_id_to_event[merged_id]
            merge_type = node["merged_type"]
            name1 = (
                get_op_name(node["uid1"], baseline_uid2node)
                if node["uid1"] is not None
                else None
            )
            name2 = (
                get_op_name(node["uid2"], variant_uid2node)
                if node["uid2"] is not None
                else None
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
        (merged_events, merged_root_ids) = self.merged_tree
        merged_id_to_event = {event["merged_id"]: event for event in merged_events}
        baseline_uid2node = {
            event.get("UID"): event
            for event in getattr(self.baseline, "events", [])
            if isinstance(event, dict)
        }
        variant_uid2node = {
            event.get("UID"): event
            for event in getattr(self.variant, "events", [])
            if isinstance(event, dict)
        }

        def get_op_name(uid, tree_uid2node):
            node = tree_uid2node.get(uid)
            if node is None:
                return None
            name = node.get("name") if "name" in node else node.get("Name")
            if name is None:
                try:
                    name = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Name)
                except Exception:
                    pass
            return name if name else str(uid)

        def get_input_shape(node):
            args = node.get("args", {})
            shape = args.get("Input Dims")
            if shape is not None:
                return str(shape)
            return ""

        def get_concrete_inputs(node):
            args = node.get("args", {})
            val = args.get("Concrete Inputs")
            if val is not None:
                return str(val)
            return ""

        def get_input_strides(node):
            args = node.get("args", {})
            val = args.get("Input Strides")
            if val is not None:
                return str(val)
            return ""

        def get_input_type(node):
            args = node.get("args", {})
            val = args.get("Input type")
            if val is not None:
                return str(val)
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

        def is_gpu_path(node):
            return not node.get("non_gpu_path", False)

        def is_kernel(node):
            cat = node.get("cat") or node.get("category")
            if cat is None:
                try:
                    cat = node.get(TraceLens.util.TraceEventUtils.TraceKeys.Category)
                except Exception:
                    pass
            return cat in ("kernel", "gpu_memcpy")

        def get_kernel_info_subtree(root_uid, tree_uid2node):
            kernel_names = []
            total_time = 0.0
            stack = [root_uid]
            visited = set()
            while stack:
                uid = stack.pop()
                if uid in visited:
                    continue
                visited.add(uid)
                node = tree_uid2node.get(uid)
                if node is None:
                    continue
                if is_kernel(node):
                    kname = node.get("name") or node.get("Name")
                    if kname is None:
                        try:
                            kname = node.get(
                                TraceLens.util.TraceEventUtils.TraceKeys.Name
                            )
                        except Exception:
                            pass
                    # Shorten kernel name if too long
                    if kname and len(kname) > 40:
                        kname = kname[:37] + "..."
                    kernel_names.append(kname)
                    dur = get_duration(node)
                    if dur:
                        try:
                            total_time += float(dur)
                        except Exception:
                            pass
                # Traverse children
                for child_uid in node.get("children", []):
                    stack.append(child_uid)
            return kernel_names, total_time

        rows = []
        visited_stats_nodes = set()

        def traverse(merged_id):
            if merged_id in visited_stats_nodes:
                return
            node = merged_id_to_event[merged_id]
            mt = node["merged_type"]
            if mt == "combined":
                event1 = baseline_uid2node.get(node["uid1"])
                event2 = variant_uid2node.get(node["uid2"])
                if event1 and event2 and is_gpu_path(event1) and is_gpu_path(event2):
                    children = [merged_id_to_event[cid] for cid in node["children"]]
                    has_non_combined_child = any(
                        c["merged_type"] != "combined" for c in children
                    )
                    if has_non_combined_child:
                        name = get_op_name(
                            node["uid1"], baseline_uid2node
                        ) or get_op_name(node["uid2"], variant_uid2node)
                        input_shape1 = get_input_shape(event1)
                        input_shape2 = get_input_shape(event2)
                        concrete_inputs1 = get_concrete_inputs(event1)
                        concrete_inputs2 = get_concrete_inputs(event2)
                        input_strides1 = get_input_strides(event1)
                        input_strides2 = get_input_strides(event2)
                        input_type1 = get_input_type(event1)
                        input_type2 = get_input_type(event2)
                        kernel_names1, total_time1 = get_kernel_info_subtree(
                            node["uid1"], baseline_uid2node
                        )
                        kernel_names2, total_time2 = get_kernel_info_subtree(
                            node["uid2"], variant_uid2node
                        )
                        rows.append(
                            {
                                "name": name,
                                "input_shape_trace1": input_shape1,
                                "input_shape_trace2": input_shape2,
                                "concrete_inputs_trace1": concrete_inputs1,
                                "concrete_inputs_trace2": concrete_inputs2,
                                "input_strides_trace1": input_strides1,
                                "input_strides_trace2": input_strides2,
                                "input_type_trace1": input_type1,
                                "input_type_trace2": input_type2,
                                "total_kernel_time_trace1": total_time1,
                                "total_kernel_time_trace2": total_time2,
                                "kernel_names_trace1": ";".join(kernel_names1),
                                "kernel_names_trace2": ";".join(kernel_names2),
                            }
                        )
                        visited_stats_nodes.add(merged_id)
                        return  # Do not traverse children further
            elif mt == "trace1":
                event1 = baseline_uid2node.get(node["uid1"])
                if event1 and is_gpu_path(event1):
                    name = get_op_name(node["uid1"], baseline_uid2node)
                    input_shape1 = get_input_shape(event1)
                    concrete_inputs1 = get_concrete_inputs(event1)
                    input_strides1 = get_input_strides(event1)
                    input_type1 = get_input_type(event1)
                    kernel_names1, total_time1 = get_kernel_info_subtree(
                        node["uid1"], baseline_uid2node
                    )
                    rows.append(
                        {
                            "name": name,
                            "input_shape_trace1": input_shape1,
                            "input_shape_trace2": "",
                            "concrete_inputs_trace1": concrete_inputs1,
                            "concrete_inputs_trace2": "",
                            "input_strides_trace1": input_strides1,
                            "input_strides_trace2": "",
                            "input_type_trace1": input_type1,
                            "input_type_trace2": "",
                            "total_kernel_time_trace1": total_time1,
                            "total_kernel_time_trace2": "",
                            "kernel_names_trace1": ";".join(kernel_names1),
                            "kernel_names_trace2": "",
                        }
                    )
                    visited_stats_nodes.add(merged_id)
                    return
            elif mt == "trace2":
                event2 = variant_uid2node.get(node["uid2"])
                if event2 and is_gpu_path(event2):
                    name = get_op_name(node["uid2"], variant_uid2node)
                    input_shape2 = get_input_shape(event2)
                    concrete_inputs2 = get_concrete_inputs(event2)
                    input_strides2 = get_input_strides(event2)
                    input_type2 = get_input_type(event2)
                    kernel_names2, total_time2 = get_kernel_info_subtree(
                        node["uid2"], variant_uid2node
                    )
                    rows.append(
                        {
                            "name": name,
                            "input_shape_trace1": "",
                            "input_shape_trace2": input_shape2,
                            "concrete_inputs_trace1": "",
                            "concrete_inputs_trace2": concrete_inputs2,
                            "input_strides_trace1": "",
                            "input_strides_trace2": input_strides2,
                            "input_type_trace1": "",
                            "input_type_trace2": input_type2,
                            "total_kernel_time_trace1": "",
                            "total_kernel_time_trace2": total_time2,
                            "kernel_names_trace1": "",
                            "kernel_names_trace2": ";".join(kernel_names2),
                        }
                    )
                    visited_stats_nodes.add(merged_id)
                    return
            for cid in node["children"]:
                traverse(cid)

        for root_id in merged_root_ids:
            traverse(root_id)

        df = pd.DataFrame(rows)
        self.diff_stats_df = df
        return df

    def diff_stats_summary(self, diff_stats_df=None):
        """
        Create a summary DataFrame from the diff_stats_df, grouping by name and input_shape, aggregating kernel times and names.
        For non-combined nodes (i.e., where only trace1 or trace2 fields are filled), ensure the other trace's fields are blank in the summary as well.
        Stores the summary DataFrame in self.diff_stats_summary_df and returns it.
        """
        if diff_stats_df is None:
            diff_stats_df = self.diff_stats_df
        df = diff_stats_df.copy()
        # Ensure all relevant columns exist
        for col in [
            "name",
            "input_shape_trace1",
            "input_shape_trace2",
            "concrete_inputs_trace1",
            "concrete_inputs_trace2",
            "input_strides_trace1",
            "input_strides_trace2",
            "input_type_trace1",
            "input_type_trace2",
            "total_kernel_time_trace1",
            "total_kernel_time_trace2",
            "kernel_names_trace1",
            "kernel_names_trace2",
        ]:
            if col not in df.columns:
                df[col] = ""

        # Convert kernel time columns to numeric, coercing errors to NaN, then fillna(0) and ensure float dtype
        for col in ["total_kernel_time_trace1", "total_kernel_time_trace2"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

        def join_or_blank(x):
            vals = x.dropna().astype(str)
            vals = vals[vals != ""]
            return ";".join(vals) if len(vals) > 0 else ""

        grouped = df.groupby(
            ["name", "input_shape_trace1", "input_shape_trace2"], dropna=False
        )
        summary = grouped.agg(
            {
                "total_kernel_time_trace1": "sum",
                "total_kernel_time_trace2": "sum",
                "kernel_names_trace1": join_or_blank,
                "kernel_names_trace2": join_or_blank,
                "concrete_inputs_trace1": join_or_blank,
                "concrete_inputs_trace2": join_or_blank,
                "input_strides_trace1": join_or_blank,
                "input_strides_trace2": join_or_blank,
                "input_type_trace1": join_or_blank,
                "input_type_trace2": join_or_blank,
            }
        ).reset_index()

        def avg_or_blank(series):
            vals = series.dropna()
            if len(vals) == 0:
                return ""
            return vals.mean()

        summary["avg_kernel_time_trace1"] = (
            grouped["total_kernel_time_trace1"].apply(avg_or_blank).values
        )
        summary["avg_kernel_time_trace2"] = (
            grouped["total_kernel_time_trace2"].apply(avg_or_blank).values
        )
        # Reorder columns
        summary = summary[
            [
                "name",
                "input_shape_trace1",
                "input_shape_trace2",
                "concrete_inputs_trace1",
                "concrete_inputs_trace2",
                "input_strides_trace1",
                "input_strides_trace2",
                "input_type_trace1",
                "input_type_trace2",
                "total_kernel_time_trace1",
                "avg_kernel_time_trace1",
                "total_kernel_time_trace2",
                "avg_kernel_time_trace2",
                "kernel_names_trace1",
                "kernel_names_trace2",
            ]
        ]
        self.diff_stats_summary_df = summary
        return summary

    def generate_tracediff_report(self):
        """
        Generate all TraceDiff output DataFrames and update the object variables.
        This does NOT write any files. Use print_tracediff_report_files to save outputs.
        """
        self.generate_diff_stats()
        self.diff_stats_summary(self.diff_stats_df)

    def print_tracediff_report_files(self, output_folder="rprt_diff"):
        """
        Write all TraceDiff output reports to files in the specified output folder (default 'rprt_diff').
        Output file names are:
            - merged_tree_output.txt
            - diff_stats.csv
            - diff_stats_summary.csv
        """
        import os
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        merged_tree_file = os.path.join(output_folder, "merged_tree_output.txt")
        diff_stats_file = os.path.join(output_folder, "diff_stats.csv")
        diff_stats_summary_file = os.path.join(output_folder, "diff_stats_summary.csv")
        self.print_merged_tree(output_file=merged_tree_file)
        if self.diff_stats_df is not None and not self.diff_stats_df.empty:
            self.diff_stats_df.to_csv(diff_stats_file, index=False)
        else:
            print(f"[TraceDiff] diff_stats_df is empty. Run generate_tracediff_report() first.")
        if self.diff_stats_summary_df is not None and not self.diff_stats_summary_df.empty:
            self.diff_stats_summary_df.to_csv(diff_stats_summary_file, index=False)
        else:
            print(f"[TraceDiff] diff_stats_summary_df is empty. Run generate_tracediff_report() first.")
