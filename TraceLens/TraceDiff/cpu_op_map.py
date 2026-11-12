###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mapping CPU ops to the kernels they launch, across a pair of traces."""

import re


def _find_common_name(name1, name2, module_map):
    """A shared display name for two op names, or None when they are unrelated."""
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


def _lca_name_groups(df):
    """Per-LCA op names and module parents, keyed by LCA id then source."""
    return {
        str(lca_id): {
            source: {
                "name": list(group["cpu_op_name"].unique()),
                "nn_module_parent": list(group["nn_module_parent"].unique()),
            }
            for source, group in df[df["lowest_common_ancestor_id"] == lca_id].groupby(
                "source"
            )
        }
        for lca_id in df["lowest_common_ancestor_id"].unique()
    }


def _module_map(df):
    """Module parents seen for each cpu op name."""
    module_map = {}
    for cpu_op in df["cpu_op_name"].unique():
        for source, group in df[df["cpu_op_name"] == cpu_op].groupby("source"):
            module_map[cpu_op] = list(group["nn_module_parent"].unique())
    return module_map


def _rename_zipped(names1, names2, module_map, rename_map):
    """Rename position-matched op names that differ between the two traces."""
    for n1, n2 in zip(names1, names2):
        if n1 == n2:
            continue
        common_name = _find_common_name(n1, n2, module_map)
        if common_name is None:
            print(
                f"[TraceDiff] No common name found for {n1} and {n2} under the same LCA, keeping original names."
            )
            continue
        print(f"[TraceDiff] Renaming: {n1}, {n2} to {common_name}")
        rename_map[n2] = common_name
        rename_map[n1] = common_name


def _rename_greedy(lcaid, n1_list, n2_list, module_map, rename_map):
    """Greedily pair differing op names when the two sides have unequal counts."""
    n1_list_copy = n1_list.copy()
    for n1 in n1_list:
        for n2 in n2_list:
            if n1 == n2:
                n1_list_copy.remove(n1)
                n2_list.remove(n2)
                break
    n1_list = n1_list_copy.copy()
    for n1 in n1_list_copy:
        for n2 in n2_list:
            common_name = _find_common_name(n1, n2, module_map)
            if common_name is not None:
                print(f"[TraceDiff] Renaming: {n1}, {n2} to {common_name}")
                rename_map[n1] = common_name
                rename_map[n2] = common_name
                n2_list.remove(n2)
                n1_list.remove(n1)
                break
    if len(n1_list) > 0 or len(n2_list) > 0:
        print(f"[TraceDiff] Unmatched for LCA {lcaid}: {n1_list} vs {n2_list}")


def _build_rename_map(df):
    """Map op names that differ between traces under one LCA onto a common name."""
    groups = _lca_name_groups(df)
    module_map = _module_map(df)
    visited_cpu_op = []
    rename_map = {}
    for lcaid, mapping in groups.items():
        if "trace1" not in mapping or "trace2" not in mapping:
            continue
        names1 = mapping["trace1"]["name"]
        names2 = mapping["trace2"]["name"]
        if all(op in visited_cpu_op for op in names1) and all(
            op in visited_cpu_op for op in names2
        ):
            continue
        visited_cpu_op.extend(names1)
        visited_cpu_op.extend(names2)
        if len(names1) == len(names2):
            _rename_zipped(names1, names2, module_map, rename_map)
        else:
            _rename_greedy(lcaid, names1, names2, module_map, rename_map)
    return rename_map


def _cpu_op_kernel_map(df_agg):
    """Kernels and module parents per cpu op, split by source."""
    cpu_op_map = {}
    for cpu_op in df_agg["cpu_op_name"].unique():
        cpu_op_map[cpu_op] = {}
        for source, group in df_agg[df_agg["cpu_op_name"] == cpu_op].groupby("source"):
            cpu_op_map[cpu_op][source] = {
                "kernels": sorted(list(group["name"].unique())),
                "nn_module_parents": sorted(list(group["nn_module_parent"].unique())),
            }
    return cpu_op_map


def _print_one_to_many(df_agg):
    """Report kernels that map to more than one cpu op in either trace."""
    result = {
        kernel_name: {
            source: {"cpu_op_name": list(group["cpu_op_name"].unique())}
            for source, group in df_agg[df_agg["name"] == kernel_name].groupby("source")
        }
        for kernel_name in df_agg["name"].unique()
    }
    print("[TraceDiff] Kernel to CPU op mapping (showing entries with 1:n mapping):")
    for name, mapping in result.items():
        for source in ("trace1", "trace2"):
            cpu_op_names = mapping.get(source, {}).get("cpu_op_name", [])
            if len(cpu_op_names) > 1:
                print(" ", name[0:30], "\t", cpu_op_names)
