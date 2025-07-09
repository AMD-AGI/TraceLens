import json
from pprint import pprint
from typing import Any, Callable, Dict

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

    def get_diff_boundaries(self):
        """
        Compare two trees and identify the boundaries of differences between them.

        This function performs a depth-first traversal of two trees and identifies
        the points of differences (PODs) and difference boundaries (DBs) between them.
        It checks for differences in node names, shapes, and children, and records
        the necessary instructions for insertions, deletions, and replacements.

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

        def get_gpu_children(tree, node):
            return node.get("gpu_events", [])

        def dfs(node1: Dict[str, Any], node2: Dict[str, Any]):
            # If either node is already a POD, skip
            uid1 = node1.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
            uid2 = node2.get(TraceLens.util.TraceEventUtils.TraceKeys.UID)
            if uid1 in self.pod1 or uid2 in self.pod2:
                return

            # Compare node names
            name1 = get_name(node1)
            name2 = get_name(node2)
            if name1 != name2:
                self.db1.append(node1)
                self.db2.append(node2)
                self.add_to_pod(node1, self.pod1, tree1)
                self.add_to_pod(node2, self.pod2, tree2)
                return

            # Get children
            children1 = get_children(tree1, node1)
            children2 = get_children(tree2, node2)

            # If number of children differ
            if len(children1) != len(children2):
                if not (children1 and children2):
                    self.db1.append(node1)
                    self.db2.append(node2)
                    self.add_to_pod(node1, self.pod1, tree1)
                    self.add_to_pod(node2, self.pod2, tree2)
                    return
                names1 = set(get_name(child) for child in children1)
                names2 = set(get_name(child) for child in children2)
                intersection = names1 & names2
                if names1 == names2:
                    pass
                elif intersection:
                    diff1 = names1 - intersection
                    diff2 = names2 - intersection
                    for child in children1:
                        if get_name(child) in diff1:
                            self.add_to_pod(child, self.pod1, tree1)
                    for child in children2:
                        if get_name(child) in diff2:
                            self.add_to_pod(child, self.pod2, tree2)
                else:
                    self.db1.append(node1)
                    self.db2.append(node2)
                    self.add_to_pod(node1, self.pod1, tree1)
                    self.add_to_pod(node2, self.pod2, tree2)
                    return
            # If both are leaf nodes
            elif not children1 and not children2:
                self.db1.append(node1)
                self.db2.append(node2)
                return
            else:
                # GPU children comparison (if available)
                gpu_children1 = get_gpu_children(tree1, node1)
                gpu_children2 = get_gpu_children(tree2, node2)

                if not gpu_children1 and not gpu_children2:
                    pass
                elif len(gpu_children1) == len(gpu_children2):
                    pass
                elif not (gpu_children1 and gpu_children2):
                    self.db1.append(node1)
                    self.db2.append(node2)
                    self.add_to_pod(node1, self.pod1, tree1)
                    self.add_to_pod(node2, self.pod2, tree2)
                    return
                else:
                    names1 = set(
                        get_name(tree1.get_UID2event(uid)) for uid in gpu_children1
                    )
                    names2 = set(
                        get_name(tree2.get_UID2event(uid)) for uid in gpu_children2
                    )
                    intersection = names1 & names2
                    if names1 == names2:
                        pass
                    elif intersection:
                        diff1 = names1 - intersection
                        diff2 = names2 - intersection
                        for uid in gpu_children1:
                            if get_name(tree1.get_UID2event(uid)) in diff1:
                                self.add_to_pod(
                                    tree1.get_UID2event(uid), self.pod1, tree1
                                )
                        for uid in gpu_children2:
                            if get_name(tree2.get_UID2event(uid)) in diff2:
                                self.add_to_pod(
                                    tree2.get_UID2event(uid), self.pod2, tree2
                                )
                    else:
                        self.db1.append(node1)
                        self.db2.append(node2)
                        self.add_to_pod(node1, self.pod1, tree1)
                        self.add_to_pod(node2, self.pod2, tree2)
                        return
            # Recursively traverse the children
            for c1, c2 in zip(children1, children2):
                dfs(c1, c2)

        # Start DFS from the root nodes (assume first cpu_root_nodes[0] is root)
        if not tree1.cpu_root_nodes or not tree2.cpu_root_nodes:
            raise ValueError(
                "Both trees must have at least one root node in cpu_root_nodes."
            )

        # Get the CPU root nodes
        roots1 = tree1.cpu_root_nodes
        roots2 = tree2.cpu_root_nodes

        # Use Wagner-Fischer algorithm (edit distance) to align roots by name
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
                        dp[i - 1][j] + 1,  # deletion
                        dp[i][j - 1] + 1,  # insertion
                        dp[i - 1][j - 1] + cost,  # substitution
                    )
            # Backtrack to get the operations
            i, j = m, n
            ops = []
            while i > 0 or j > 0:
                if (
                    i > 0
                    and j > 0
                    and (
                        get_name(tree1.get_UID2event(seq1[i - 1]))
                        == get_name(tree2.get_UID2event(seq2[j - 1]))
                        or dp[i][j] == dp[i - 1][j - 1] + 1
                    )
                ):
                    if get_name(tree1.get_UID2event(seq1[i - 1])) == get_name(
                        tree2.get_UID2event(seq2[j - 1])
                    ):
                        ops.append(("match", i - 1, j - 1))
                    else:
                        ops.append(("replace", i - 1, j - 1))
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

        # Find differences in roots by name
        ops = wagner_fischer(roots1, roots2)
        for op, i, j in ops:
            if op == "match":
                node1 = tree1.get_UID2event(roots1[i])
                node2 = tree2.get_UID2event(roots2[j])
                dfs(node1, node2)
            elif op == "replace":
                node1 = tree1.get_UID2event(roots1[i])
                node2 = tree2.get_UID2event(roots2[j])
                self.db1.append(node1)
                self.db2.append(node2)
                self.add_to_pod(node1, self.pod1, tree1)
                self.add_to_pod(node2, self.pod2, tree2)
            elif op == "delete":
                node1 = tree1.get_UID2event(roots1[i])
                self.db1.append(node1)
                self.add_to_pod(node1, self.pod1, tree1)
            elif op == "insert":
                node2 = tree2.get_UID2event(roots2[j])
                self.db2.append(node2)
                self.add_to_pod(node2, self.pod2, tree2)

        return self.db1, self.db2, self.pod1, self.pod2
