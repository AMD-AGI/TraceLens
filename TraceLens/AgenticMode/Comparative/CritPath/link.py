###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import pprint


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


# Helper to recursively convert lists to tuples
def list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple(x) for x in obj)
    return obj


def process_files(kineto_file, et_file, linked_file):
    et_data = load_json_file(et_file)
    kin_data = load_json_file(kineto_file)

    # Extract nodes
    et_nodes = et_data.get("nodes", [])
    kin_nodes = kin_data.get("traceEvents", [])

    # Build a lookup for et nodes by attrs[0]['value']
    et_lookup = {}
    et_data_map = {}
    for et_node in et_nodes:
        attrs = et_node.get("attrs", [])
        if attrs and isinstance(attrs, list) and "value" in attrs[0]:
            et_id = attrs[0]["value"]
            et_lookup[et_id] = et_node.get("id")
            et_data_map[et_id] = et_node

    # Augment kin nodes with et_inputs, et_outputs (temporary fields for computing data_deps)
    for kin_node in kin_nodes:
        args = kin_node.get("args", {})
        record_func_id = args.get("Record function id")
        if isinstance(record_func_id, int) and record_func_id in et_lookup:
            et_node = et_data_map[record_func_id]
            kin_node["et_inputs"] = et_node.get("inputs", {}).get("values", [])
            kin_node["et_outputs"] = et_node.get("outputs", {}).get("values", [])

    # Build a map from tensor pointer to kin node record function id (for outputs)
    tensor_to_record_func_id = {}
    for kin_node in kin_nodes:
        et_outputs = kin_node.get("et_outputs", [])
        args = kin_node.get("args", {})
        record_func_id = args.get("Record function id")
        for tensor in et_outputs:
            if isinstance(tensor, list):
                tensor_key = list_to_tuple(tensor)
                tensor_to_record_func_id[tensor_key] = record_func_id

    # For each kin node, find data dependencies by matching et_inputs to tensor_to_record_func_id
    for kin_node in kin_nodes:
        et_inputs = kin_node.get("et_inputs", [])
        data_deps = []
        for tensor in et_inputs:
            if isinstance(tensor, list):
                tensor_key = list_to_tuple(tensor)
                dep_id = tensor_to_record_func_id.get(tensor_key)
                if dep_id is not None and dep_id != kin_node.get("args", {}).get(
                    "Record function id"
                ):
                    data_deps.append(dep_id)
        kin_node["data_deps"] = data_deps

    # Clean up temporary fields (et_inputs, et_outputs) to save memory
    for kin_node in kin_nodes:
        kin_node.pop("et_inputs", None)
        kin_node.pop("et_outputs", None)

    # Update the trace name by appending "_linked"
    if "traceName" in kin_data:
        trace_name = kin_data["traceName"]
        if trace_name.endswith(".json"):
            kin_data["traceName"] = trace_name[:-5] + "_linked.json"
        else:
            kin_data["traceName"] += "_linked"

    # Save the updated kineto data to a new JSON file
    with open(linked_file, "w") as f:
        json.dump(kin_data, f)


def prune_spillover_kernels(events):
    """
    Remove GPU kernels from previous iteration that spilled over into this trace.
    Finds the earliest GPU kernel with a CPU parent and removes all GPU kernels before it.

    Args:
        events: List of trace events to prune.
    """

    # Find earliest GPU kernel with CPU parent
    gpu_kernels = []
    event_dict = {e.get("UID"): e for e in events if "UID" in e}

    for event in events:
        category = event.get("cat", "")
        if category in {"kernel", "gpu_memset", "gpu_memcpy"}:
            ts = event.get("ts", 0)
            uid = event.get("UID")
            parent_uid = event.get("parent")

            has_cpu_parent = False
            if parent_uid is not None and parent_uid != -1:
                parent_event = event_dict.get(parent_uid)
                if parent_event:
                    parent_cat = parent_event.get("cat", "")
                    if parent_cat in {"cpu_op", "cuda_runtime", "cuda_driver"}:
                        has_cpu_parent = True

            gpu_kernels.append(
                {
                    "uid": uid,
                    "ts": ts,
                    "has_cpu_parent": has_cpu_parent,
                }
            )

    if not gpu_kernels:
        return

    gpu_kernels.sort(key=lambda x: x["ts"])

    # Find earliest kernel with CPU parent
    earliest_valid_ts = None
    spillover_count = 0
    first_valid_kernel = None
    for kernel in gpu_kernels:
        if kernel["has_cpu_parent"]:
            earliest_valid_ts = kernel["ts"]
            first_valid_kernel = kernel
            break
        spillover_count += 1

    if earliest_valid_ts is None:
        return

    # Remove spillover kernels
    removed_uids = set()
    indices_to_remove = []

    for i, event in enumerate(events):
        category = event.get("cat", "")
        ts = event.get("ts", 0)
        uid = event.get("UID")

        if (
            category in {"kernel", "gpu_memset", "gpu_memcpy"}
            and ts < earliest_valid_ts
        ):
            removed_uids.add(uid)
            indices_to_remove.append(i)

    # Remove events in reverse order to maintain indices
    for i in reversed(indices_to_remove):
        events.pop(i)

    # Clean up references
    for event in events:
        if "gpu_events" in event:
            event["gpu_events"] = [
                uid for uid in event["gpu_events"] if uid not in removed_uids
            ]
        if "children" in event:
            event["children"] = [
                uid for uid in event["children"] if uid not in removed_uids
            ]

