###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tree merging: Wagner-Fischer alignment, name disambiguation, and the
merged-event builder."""

from .util import (
    _CATEGORY,
    _GRAPH_LAUNCH_NAMES,
    _KERNEL_DISPATCH_CATEGORIES,
    _NAME,
    _ROOT_DESCENT_STOP_CATEGORIES,
    _UID,
    _WRAPPER_SKIP_CATEGORIES,
    _get_name_node,
    _is_gpu_path,
    _sort_by_ts,
)


def _gpu_path_child_names_at_bfs_levels(uid, uid2node, max_depth):
    """
    Return a list of tuples, one per BFS level (1..max_depth), each containing
    the sorted multiset of normalized GPU-path child names at that depth
    relative to `uid`.

    A node is on the GPU path when it does not have non_gpu_path=True.
    """
    levels = []
    current_frontier = [uid]
    for _ in range(max_depth):
        next_frontier = []
        names_this_level = []
        for u in current_frontier:
            node = uid2node.get(u)
            if not node:
                continue
            for child_uid in node.get("children", []):
                child_node = uid2node.get(child_uid)
                if child_node and not child_node.get("non_gpu_path", False):
                    next_frontier.append(child_uid)
                    names_this_level.append(_get_name_node(child_node))
        levels.append(tuple(sorted(names_this_level)))
        current_frontier = next_frontier
        if not current_frontier:
            break
    return levels


def _match_depth(levels_a, levels_b):
    """(full_levels_matched, partial_overlap_at_first_divergence).

    full_levels_matched: number of BFS levels where the sorted child-name
    tuples are identical before the first divergence.
    partial_overlap_at_first_divergence: multiset intersection size at the
    first diverging level (0 if all levels matched).
    """
    max_d = max(len(levels_a), len(levels_b))
    for d in range(max_d):
        a = levels_a[d] if d < len(levels_a) else ()
        b = levels_b[d] if d < len(levels_b) else ()
        if a != b:
            overlap = sum(min(a.count(x), b.count(x)) for x in set(a))
            return (d, overlap)
    return (max_d, 0)


def _name_index(ops, children1, children2, uid2node1, uid2node2):
    """Normalized name -> indices on each side, plus the names WF already matched."""
    all1_by_name = {}  # normalized_name -> [idx in children1]
    all2_by_name = {}  # normalized_name -> [idx in children2]
    matched_names = set()  # names that appear in at least one match op
    for op, i, j in ops:
        if op in ("match", "delete"):
            n = _get_name_node(uid2node1.get(children1[i]))
            all1_by_name.setdefault(n, []).append(i)
            if op == "match":
                matched_names.add(n)
        if op in ("match", "insert"):
            n = _get_name_node(uid2node2.get(children2[j]))
            all2_by_name.setdefault(n, []).append(j)
            if op == "match":
                matched_names.add(n)
    return all1_by_name, all2_by_name, matched_names


def _greedy_name_matches(indices1, indices2, levels1, levels2, orig_match_set):
    """Pair same-name nodes by BFS subtree similarity, highest-scoring pair first."""
    n_to_match = min(len(indices1), len(indices2))
    scored_pairs = sorted(
        (_match_depth(levels1[i], levels2[j]), (i, j) in orig_match_set, i, j)
        for i in indices1
        for j in indices2
    )
    scored_pairs.sort(key=lambda x: (-x[0][0], -x[0][1], not x[1], x[2], x[3]))

    matches = set()
    assigned1, assigned2 = set(), set()
    for _, _, i, j in scored_pairs:
        if i in assigned1 or j in assigned2:
            continue
        matches.add((i, j))
        assigned1.add(i)
        assigned2.add(j)
        if len(assigned1) == n_to_match:
            break
    return matches, set(indices1) - assigned1, set(indices2) - assigned2


def _rebuild_ops(ops, handled1, handled2, matches, deletes, inserts):
    """Keep the ops disambiguation did not touch; replace the ones it did."""
    final_ops = [
        (op, i, j)
        for op, i, j in ops
        if not (
            (op == "match" and (i in handled1 or j in handled2))
            or (op == "delete" and i in handled1)
            or (op == "insert" and j in handled2)
        )
    ]
    final_ops.extend(("match", i, j) for i, j in matches)
    final_ops.extend(("delete", i, None) for i in sorted(deletes))
    final_ops.extend(("insert", None, j) for j in sorted(inserts))
    return final_ops


def _disambiguate_same_name_candidates(
    ops, children1, children2, baseline_uid2node, variant_uid2node, max_depth=4
):
    """
    Post-WF pass: for every normalized name that appears more than once on at
    least one side, re-match using BFS-level GPU-path child name comparison
    (up to *max_depth* levels).

    Candidates include ALL nodes with that name on each side — both matched and
    unmatched — so swapped WF matches can be corrected (Issue 1).

    The side with fewer nodes of a given name (the minority side) is guaranteed
    to be fully matched; only majority-side leftovers become inserts/deletes
    (Issue 2).

    Matching uses a greedy algorithm: score every (i, j) pair by the number of
    BFS levels that agree before the first divergence, then greedily assign
    highest-scoring pairs first.  Ties are broken by preferring pairs that
    already exist in the WF output, then by positional order.

    Does not mutate children1/children2.  Returns a new ops list.
    """
    orig_match_set = {(i, j) for op, i, j in ops if op == "match"}
    all1_by_name, all2_by_name, matched_names = _name_index(
        ops, children1, children2, baseline_uid2node, variant_uid2node
    )

    # Only disambiguate names that: appear on both sides, have >1 on at least
    # one side, AND participate in at least one existing match op.
    ambiguous_names = {
        name
        for name in set(all1_by_name) & set(all2_by_name) & matched_names
        if len(all1_by_name[name]) > 1 or len(all2_by_name[name]) > 1
    }
    if not ambiguous_names:
        return ops

    handled1 = set()  # all trace1 indices in ambiguous name groups
    handled2 = set()  # all trace2 indices in ambiguous name groups
    new_matches, new_deletes, new_inserts = set(), set(), set()
    for name in ambiguous_names:
        indices1 = sorted(all1_by_name[name])
        indices2 = sorted(all2_by_name[name])
        handled1.update(indices1)
        handled2.update(indices2)
        levels1 = {
            i: _gpu_path_child_names_at_bfs_levels(
                children1[i], baseline_uid2node, max_depth
            )
            for i in indices1
        }
        levels2 = {
            j: _gpu_path_child_names_at_bfs_levels(
                children2[j], variant_uid2node, max_depth
            )
            for j in indices2
        }
        matches, deletes, inserts = _greedy_name_matches(
            indices1, indices2, levels1, levels2, orig_match_set
        )
        new_matches |= matches
        new_deletes |= deletes
        new_inserts |= inserts

    # If disambiguation produced the same matches as WF, nothing to do
    if new_matches <= orig_match_set:
        return ops

    return _rebuild_ops(ops, handled1, handled2, new_matches, new_deletes, new_inserts)


def _make_merged_event(merged_id, uid1, uid2, children, nn_module_stack):
    """Build one merged event; children are merged_id references."""
    # UID 0 is valid; do not use truthiness (0 and 0 is falsy).
    if uid1 is not None:
        merged_type = "combined" if uid2 is not None else "trace1"
    else:
        merged_type = "trace2"
    return {
        "merged_id": merged_id,
        "uid1": uid1,
        "uid2": uid2,
        "merged_type": merged_type,
        "children": children,
        "nn_module_stack": nn_module_stack,
    }


def _descend_single_wrapper_child(tree, node):
    """Descend through single-child wrapper nodes to the first real branch point."""
    current = node
    while True:
        children = current.get("children", [])
        if len(children) != 1:
            return current
        child = tree.get_UID2event(children[0])
        if (
            child.get(_CATEGORY) in _ROOT_DESCENT_STOP_CATEGORIES
            and child.get(_NAME) not in _GRAPH_LAUNCH_NAMES
        ):
            return current
        current = child


def _reparent_under(tree, root, children):
    """Hoist children onto root, fixing their parent pointers."""
    root["children"] = children
    root_uid = root.get(_UID)
    for child_uid in children:
        tree.get_UID2event(child_uid)["parent"] = root_uid


def _is_skipped_dispatch(node):
    """Kernels and kernel dispatches are never wrapper candidates; graph launches are."""
    if not node:
        return False
    if node.get(_CATEGORY) not in _WRAPPER_SKIP_CATEGORIES:
        return False
    return node.get(_NAME) not in _GRAPH_LAUNCH_NAMES


def _gpu_path_children(tree, node):
    """Immediate GPU-path children of a node, empty when the node is missing."""
    if not node:
        return []
    return [c for c in tree.get_children_events(node) if _is_gpu_path(c)]


def _expand(children, subs, uid2node):
    """Replace substituted slots with their node lists; resolve the rest by UID."""
    nodes = []
    for idx, uid in enumerate(children):
        if idx in subs:
            nodes.extend(subs[idx])
        else:
            nodes.append(uid2node[uid])
    return nodes


def _subtree_contains_cuda_runtime(node):
    """Return True if this node is a cuda_runtime node. Graph launch events are exempt."""
    if not node:
        return False
    if node.get(_NAME) in _GRAPH_LAUNCH_NAMES:
        return False
    cat = node.get(_CATEGORY)
    return cat in _KERNEL_DISPATCH_CATEGORIES


def _collapse_single_gpu_child(uid, uid2node, tree_obj):
    """
    Descend through single-GPU-path-child wrapper nodes until reaching
    a node with != 1 GPU-path child, a cpu_op with a cuda_runtime
    child, or a dead end.
    Returns (uid, gpu_path_children_uids) of the deepest reachable node.
    """
    current = uid
    gpu_kids = []
    while True:
        node = uid2node.get(current)
        if not node or (
            tree_obj.event_to_category(node) in _KERNEL_DISPATCH_CATEGORIES
            and node.get(_NAME) not in _GRAPH_LAUNCH_NAMES
        ):
            break
        child_nodes = tree_obj.get_children_events(node)
        cat = tree_obj.event_to_category(node)
        if cat == "cpu_op":
            has_cr_child = any(
                tree_obj.event_to_category(c) in _KERNEL_DISPATCH_CATEGORIES
                and c.get(_NAME) not in _GRAPH_LAUNCH_NAMES
                for c in child_nodes
            )
            if has_cr_child:
                break
        gpu_child_nodes = [c for c in child_nodes if _is_gpu_path(c)]
        if len(gpu_child_nodes) != 1:
            gpu_kids = [c[_UID] for c in gpu_child_nodes]
            break
        current = gpu_child_nodes[0][_UID]
    return current, gpu_kids


class _MergeBuilder:
    """Builds the flat merged event list for one TraceDiff pair."""

    def __init__(self, diff):
        self.diff = diff
        self.tree1 = diff.baseline
        self.tree2 = diff.variant
        self.uid2node1 = diff._get_baseline_uid2node()
        self.uid2node2 = diff._get_variant_uid2node()
        self.wf_cache = {}
        self.events = []
        self.next_merged_id = 0
        self.merged_id_by_uid_pair = {}

    def _aligned_wf(self, items1, items2, strip_details=False):
        ops = self.diff.wagner_fischer(items1, items2, self.wf_cache, strip_details)
        return _disambiguate_same_name_candidates(
            ops, items1, items2, self.uid2node1, self.uid2node2
        )

    def _children_with_missing(self, uid1, uid2):
        """Get aligned children lists, adding missing-by-name from full child list."""
        node1 = self.uid2node1.get(uid1)
        node2 = self.uid2node2.get(uid2)
        all_children1 = self.tree1.get_children_events(node1) if node1 else []
        all_children2 = self.tree2.get_children_events(node2) if node2 else []

        gpu_nodes1 = [n for n in all_children1 if _is_gpu_path(n)]
        gpu_nodes2 = [n for n in all_children2 if _is_gpu_path(n)]
        gpu_names1 = {_get_name_node(n) for n in gpu_nodes1}
        gpu_names2 = {_get_name_node(n) for n in gpu_nodes2}

        all_by_name1 = {_get_name_node(n): n for n in all_children1}
        all_by_name2 = {_get_name_node(n): n for n in all_children2}

        for name in gpu_names1 - gpu_names2:
            if name in all_by_name2:
                gpu_nodes2.append(all_by_name2[name])
        for name in gpu_names2 - gpu_names1:
            if name in all_by_name1:
                gpu_nodes1.append(all_by_name1[name])

        return _sort_by_ts(gpu_nodes1), _sort_by_ts(gpu_nodes2)

    def _substitute_wrappers(self, ops, children1, children2):
        """Reduce spurious delete/insert pairs by detecting wrapper nodes.

        For every delete-node (baseline) and insert-node (variant) pair,
        checks three cases (kernels and cuda_runtime are skipped):

        1. Deleted name appears in inserted node's GPU-path children
           -> substitute inserted UID with its GPU-path children.
        2. Inserted name appears in deleted node's GPU-path children
           -> substitute deleted UID with its GPU-path children.
        3. Both have GPU-path children with identical normalized name lists
           -> substitute both with their GPU-path children.

        Does NOT mutate the tree. Returns (changed, new_children1, new_children2).
        """
        delete_indices = [i for op, i, _ in ops if op == "delete"]
        insert_indices = [j for op, _, j in ops if op == "insert"]

        if not delete_indices or not insert_indices:
            return False, children1, children2

        sub1 = {}  # index in children1 -> replacement child nodes
        sub2 = {}  # index in children2 -> replacement child nodes
        for di in delete_indices:
            for ii in insert_indices:
                sub_d, sub_i = self._wrapper_pair(children1[di], children2[ii])
                if sub_d is not None:
                    sub1[di] = sub_d
                if sub_i is not None:
                    sub2[ii] = sub_i

        if not sub1 and not sub2:
            return False, children1, children2

        return (
            True,
            _sort_by_ts(_expand(children1, sub1, self.uid2node1)),
            _sort_by_ts(_expand(children2, sub2, self.uid2node2)),
        )

    def _wrapper_pair(self, uid_d, uid_i):
        """GPU-path children to substitute for this delete/insert pair, if any."""
        node_d = self.uid2node1.get(uid_d)
        node_i = self.uid2node2.get(uid_i)
        if _is_skipped_dispatch(node_d) or _is_skipped_dispatch(node_i):
            return None, None

        imm_d = _gpu_path_children(self.tree1, node_d)
        imm_i = _gpu_path_children(self.tree2, node_i)
        names_d = [_get_name_node(c) for c in imm_d]
        names_i = [_get_name_node(c) for c in imm_i]
        if imm_d and imm_i and names_d == names_i:
            return imm_d, imm_i

        name_d = _get_name_node(node_d)
        name_i = _get_name_node(node_i)
        return (
            imm_d if name_i and name_i in names_d else None,
            imm_i if name_d and name_d in names_i else None,
        )

    def merge(self, uid1, uid2):
        """Merge the subtrees rooted at this UID pair; return their merged id."""
        key = (uid1, uid2)
        if key in self.merged_id_by_uid_pair:
            return self.merged_id_by_uid_pair[key]

        self._record_boundary(uid1, uid2)
        merged_id, nn_module_stack = self._register(key)
        children1, children2 = self._children_with_missing(uid1, uid2)
        uid1, children1, uid2, children2 = self._collapse_lists(
            uid1, children1, uid2, children2
        )
        ops = self._align(children1, children2)
        ops, children1, children2 = self._reconcile(ops, children1, children2)
        ops = self._canonicalize_unmatched(ops, children1, children2)
        child_merged_ids = self._merge_children(ops, children1, children2)

        self.events.append(
            _make_merged_event(merged_id, uid1, uid2, child_merged_ids, nn_module_stack)
        )
        return merged_id

    def _record_boundary(self, uid1, uid2):
        """Mark a point of difference where the two sides disagree."""
        if uid2 is None:
            if uid1 is not None:
                self.diff.pod1.add(uid1)
            return
        if uid1 is None:
            self.diff.pod2.add(uid2)
            return
        if uid1 in self.diff.pod1 or uid2 in self.diff.pod2:
            return
        node1 = self.uid2node1.get(uid1)
        node2 = self.uid2node2.get(uid2)
        if _get_name_node(node1) != _get_name_node(node2):
            self.diff.db1.append(node1)
            self.diff.db2.append(node2)
            self.diff.pod1.add(uid1)
            self.diff.pod2.add(uid2)

    def _register(self, key):
        """Allocate this pair's merged id and cross-reference their UIDs."""
        uid1, uid2 = key
        node1 = self.uid2node1.get(uid1) if uid1 is not None else None
        node2 = self.uid2node2.get(uid2) if uid2 is not None else None

        merged_id = self.next_merged_id
        self.next_merged_id += 1
        self.merged_id_by_uid_pair[key] = merged_id

        if uid1 is not None and uid2 is not None:
            self.diff.merged_uid_map[(1, uid1)] = uid2
            self.diff.merged_uid_map[(2, uid2)] = uid1
            nn_module_stack = node1.get("nn_module_stack", "")
        elif uid1 is not None:
            self.diff.merged_uid_map[(1, uid1)] = -1
            nn_module_stack = node1.get("nn_module_stack", "")
        else:
            self.diff.merged_uid_map[(2, uid2)] = -1
            nn_module_stack = node2.get("nn_module_stack", "")
        return merged_id, nn_module_stack

    def _collapse_lists(self, uid1, children1, uid2, children2):
        """Descend a lone GPU child through wrappers to find the branch point."""
        if len(children1) == len(children2):
            return uid1, children1, uid2, children2
        if len(children1) == 1:
            collapsed, deeper = _collapse_single_gpu_child(
                children1[0], self.uid2node1, self.tree1
            )
            if deeper:
                return collapsed, deeper, uid2, children2
        elif len(children2) == 1:
            collapsed, deeper = _collapse_single_gpu_child(
                children2[0], self.uid2node2, self.tree2
            )
            if deeper:
                return uid1, children1, collapsed, deeper
        return uid1, children1, uid2, children2

    def _align(self, items1, items2):
        """Equal-length sibling lists free of kernel dispatches align positionally."""
        any_cr = any(
            _subtree_contains_cuda_runtime(self.uid2node1.get(c)) for c in items1
        ) or any(_subtree_contains_cuda_runtime(self.uid2node2.get(c)) for c in items2)
        if len(items1) == len(items2) and not any_cr:
            return [("match", i, i) for i in range(len(items1))]
        return self._aligned_wf(items1, items2)

    @staticmethod
    def _remap_ops(ops, index1, index2):
        """Translate op indices back into the caller's index space."""
        remapped = []
        for op, i, j in ops:
            if op == "match":
                remapped.append(("match", index1[i], index2[j]))
            elif op == "delete":
                remapped.append(("delete", index1[i], None))
            elif op == "insert":
                remapped.append(("insert", None, index2[j]))
        return remapped

    @staticmethod
    def _pin_matches(pinned, recon1, recon2):
        """Locate the pinned UID pairs in the expanded lists; report what is left."""
        uid_to_ri = {c: idx for idx, c in enumerate(recon1)}
        uid_to_rj = {c: idx for idx, c in enumerate(recon2)}
        new_ops = []
        pinned_ri = set()
        pinned_rj = set()
        for u1, u2 in pinned:
            ri = uid_to_ri.get(u1)
            rj = uid_to_rj.get(u2)
            if ri is not None and rj is not None:
                new_ops.append(("match", ri, rj))
                pinned_ri.add(ri)
                pinned_rj.add(rj)
        free1 = [i for i in range(len(recon1)) if i not in pinned_ri]
        free2 = [j for j in range(len(recon2)) if j not in pinned_rj]
        return new_ops, free1, free2

    def _align_free(self, recon1, recon2, free1, free2):
        """Align the sibling indices left unpinned by the first alignment."""
        if not free1 or not free2:
            return [("delete", i, None) for i in free1] + [
                ("insert", None, j) for j in free2
            ]
        free_ops = self._align([recon1[i] for i in free1], [recon2[j] for j in free2])
        return self._remap_ops(free_ops, free1, free2)

    def _reconcile(self, ops, children1, children2):
        """Re-align after substituting wrapper nodes for their GPU-path children."""
        if all(op == "match" for op, _, _ in ops):
            return ops, children1, children2
        changed, recon1, recon2 = self._substitute_wrappers(ops, children1, children2)
        if not changed:
            return ops, children1, children2

        # Matches are tracked by UID pair: expansion and re-sorting move indices.
        pinned = [
            (children1[i], children2[j]) for op_type, i, j in ops if op_type == "match"
        ]
        new_ops, free1, free2 = self._pin_matches(pinned, recon1, recon2)
        return new_ops + self._align_free(recon1, recon2, free1, free2), recon1, recon2

    def _align_collapsed(self, collapsed1, collapsed2):
        """Align collapsed nodes, retrying the leftovers with details stripped."""
        pass1 = self._aligned_wf(collapsed1, collapsed2, strip_details=False)
        still1 = [ci for cop, ci, _ in pass1 if cop == "delete"]
        still2 = [cj for cop, _, cj in pass1 if cop == "insert"]
        if not still1 or not still2:
            return pass1
        pass2 = self._aligned_wf(
            [collapsed1[ci] for ci in still1],
            [collapsed2[cj] for cj in still2],
            strip_details=True,
        )
        return [op for op in pass1 if op[0] == "match"] + self._remap_ops(
            pass2, still1, still2
        )

    def _canonicalize_unmatched(self, ops, children1, children2):
        """Collapse still-unmatched nodes through wrappers and re-align them."""
        unmatched_idx1 = [i for op, i, _ in ops if op == "delete"]
        unmatched_idx2 = [j for op, _, j in ops if op == "insert"]
        if not unmatched_idx1 or not unmatched_idx2:
            return ops

        collapsed1 = [
            _collapse_single_gpu_child(children1[i], self.uid2node1, self.tree1)[0]
            for i in unmatched_idx1
        ]
        collapsed2 = [
            _collapse_single_gpu_child(children2[j], self.uid2node2, self.tree2)[0]
            for j in unmatched_idx2
        ]
        collapsed_ops = self._align_collapsed(collapsed1, collapsed2)

        for cop, ci, cj in collapsed_ops:
            if cop in ("match", "delete"):
                children1[unmatched_idx1[ci]] = collapsed1[ci]
            if cop in ("match", "insert"):
                children2[unmatched_idx2[cj]] = collapsed2[cj]
        kept = [(op, i, j) for op, i, j in ops if op == "match"]
        return kept + self._remap_ops(collapsed_ops, unmatched_idx1, unmatched_idx2)

    def _merge_children(self, ops, children1, children2):
        """Recurse into each aligned child pair, recording one-sided children."""
        child_merged_ids = []
        for op, i, j in ops:
            if op == "match":
                child_merged_ids.append(self.merge(children1[i], children2[j]))
            elif op == "delete":
                c1 = children1[i]
                child_node = self.uid2node1.get(c1)
                if child_node:
                    self.diff.db1.append(child_node)
                child_merged_ids.append(self.merge(c1, None))
            elif op == "insert":
                c2 = children2[j]
                child_node = self.uid2node2.get(c2)
                if child_node:
                    self.diff.db2.append(child_node)
                child_merged_ids.append(self.merge(None, c2))
        return child_merged_ids
