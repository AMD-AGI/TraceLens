###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from dag import DAG, DAGNode, EdgeType
from TraceLens import TraceDiff, TreePerfAnalyzer


def load_trace_and_tree(trace_file1, trace_file2=None):
    perf_analyzer1 = TreePerfAnalyzer.from_file(trace_file1)
    tree1 = perf_analyzer1.tree

    tree2 = None
    if trace_file2 is not None:
        perf_analyzer2 = TreePerfAnalyzer.from_file(trace_file2)
        tree2 = perf_analyzer2.tree

    return tree1, tree2


def build_dependency_map(tree1):
    """
    Build a dependency map based on data dependencies from the execution trace.

    Key insight: When a tensor ID appears as both input and output to operations,
    we must respect temporal ordering to avoid spurious backward dependencies.

    For each tensor ID (external_id):
    - Track which operation last WROTE to it (by timestamp)
    - Operations that READ this tensor depend on the last writer
    - This prevents backward-in-time dependencies
    """
    external_id_to_uid = {}
    uid_to_external_id = {}
    uid_to_timestamp = {}

    # First pass: build mappings and collect timestamps
    for event in tree1.events:
        args = event.get("args", {})
        external_id = args.get("External id")
        uid = event.get("UID")
        ts = event.get("ts", 0)

        if external_id is not None and uid is not None:
            external_id_to_uid[external_id] = uid
            uid_to_external_id[uid] = external_id
            uid_to_timestamp[uid] = ts

    # Track the last writer for each tensor (external_id) by timestamp
    # external_id -> (uid, timestamp) of the operation that wrote to it
    tensor_last_writer = {}

    # Second pass: identify writers (operations that produce a tensor as output)
    for event in tree1.events:
        uid = event.get("UID")
        if uid is None:
            continue

        args = event.get("args", {})
        external_id = args.get("External id")
        ts = uid_to_timestamp.get(uid, 0)

        if external_id is not None:
            # This operation produces output with this external_id
            # Update if this is the first writer or a later writer
            if (
                external_id not in tensor_last_writer
                or ts > tensor_last_writer[external_id][1]
            ):
                tensor_last_writer[external_id] = (uid, ts)

    # Third pass: build dependency map based on temporal ordering
    dependency_map = {}
    for event in tree1.events:
        uid = event.get("UID")
        if uid is None:
            continue

        current_ts = uid_to_timestamp.get(uid, 0)
        current_external_id = uid_to_external_id.get(uid)
        data_deps = event.get("data_deps", [])
        dependency_map[uid] = []

        for ext_id in data_deps:
            # Skip self-referencing dependencies (node depending on its own external_id)
            if ext_id == current_external_id:
                continue

            # Find the writer of this tensor
            if ext_id in tensor_last_writer:
                writer_uid, writer_ts = tensor_last_writer[ext_id]

                # Only create dependency if writer happened BEFORE current operation
                if writer_ts < current_ts and writer_uid != uid:
                    dependency_map[uid].append(writer_uid)
            else:
                # Fallback: use the simple mapping if no writer found
                dep_uid = external_id_to_uid.get(ext_id)
                if dep_uid is not None and dep_uid != uid:
                    # Check timestamp ordering
                    dep_ts = uid_to_timestamp.get(dep_uid, 0)
                    if dep_ts < current_ts:
                        dependency_map[uid].append(dep_uid)

    return dependency_map


def remove_dangling_nodes(dag):
    """
    Remove all dangling nodes from the DAG. Dangling nodes are those with no input
    pointers and no output pointers.

    Args:
        dag (DAG): The DAG object to clean up.
    """
    dangling_nodes = [
        node
        for node in dag.nodes
        if not node.input_pointers and not node.output_pointers
    ]

    for node in dangling_nodes:
        dag.nodes.remove(node)
        del dag.uid_to_dagnode[node.dag_uid]

    print(f"Removed {len(dangling_nodes)} dangling nodes.")


def _add_source_and_sink_nodes(dag, tree1, start_uid=None, end_uid=None, debug=False):
    """
    Add source and sink nodes to the DAG.
    Source node connects to all nodes with no input pointers.
    Sink node connects from all nodes with no outgoing edges (leaf nodes).

    Args:
        dag (DAG): The DAG object to update.
        tree1: The tree containing cpu_root_nodes.
        start_uid (optional): UID of a specific node to use as trace start.
                             If None, connects to all nodes with no input pointers.
        debug (bool): Whether to print debug output.
        end_uid (optional): UID of a specific node to use as trace end.
                           If None, uses all nodes with out-degree 0.
    """
    if not tree1.cpu_root_nodes:
        print("Warning: No CPU root nodes found, skipping source/sink creation")
        return

    # Determine source connection - connect to all nodes with no input pointers
    source_nodes = []
    if start_uid is not None:
        if dag.get_node(start_uid):
            source_nodes = [start_uid]
            print(f"Using custom trace start: {start_uid}")
        else:
            print(
                f"Warning: Specified start_uid {start_uid} not found. Using all nodes with no inputs."
            )

    if not source_nodes:
        # Find all nodes with no input pointers
        for node in dag.nodes:
            if not node.input_pointers:
                source_nodes.append(node.dag_uid)
        if debug:
            print(
                f"Found {len(source_nodes)} nodes with no input pointers (connecting source to all)"
            )

    # Find all nodes with no outgoing edges (leaf nodes that should connect to sink)
    leaf_nodes = []
    if end_uid is not None:
        if dag.get_node(end_uid):
            leaf_nodes = [end_uid]
            print(f"Using custom trace end: {end_uid}")
        else:
            print(
                f"Warning: Specified end_uid {end_uid} not found. Using all leaf nodes."
            )

    if not leaf_nodes:
        # Find all nodes with no output pointers
        for node in dag.nodes:
            if not node.output_pointers:
                leaf_nodes.append(node.dag_uid)
        print(f"Found {len(leaf_nodes)} leaf nodes (nodes with no outputs)")

    # Create source node that connects to all nodes with no input pointers
    source_uid = "trace_source"
    source_node = DAGNode(
        treenode_uid=source_uid,
        name="TRACE_START",
        category="trace_source",
        measured_latency=0,  # No latency for source node
        input_pointers=[],  # Source has no inputs
        output_pointers=source_nodes.copy(),  # Points to all nodes with no inputs
        input_edge_types=[],
        output_edge_types=[EdgeType.TRACE_BOUNDARY] * len(source_nodes),
    )
    dag.add_node(source_node)

    # Update all nodes with no inputs to have trace_source as input
    for node_uid in source_nodes:
        node = dag.get_node(node_uid)
        if node:
            dag.update_node(
                node_uid,
                input_pointers=node.input_pointers + [source_uid],
                input_edge_types=node.input_edge_types + [EdgeType.TRACE_BOUNDARY],
            )

    # Create sink node that receives from all leaf nodes
    sink_uid = "trace_sink"
    sink_node = DAGNode(
        treenode_uid=sink_uid,
        name="TRACE_END",
        category="trace_sink",
        measured_latency=0,  # No latency for sink node
        input_pointers=leaf_nodes.copy(),  # Receives from all leaf nodes
        output_pointers=[],  # Sink has no outputs
        input_edge_types=[EdgeType.TRACE_BOUNDARY] * len(leaf_nodes),
        output_edge_types=[],
    )
    dag.add_node(sink_node)

    # Update all leaf nodes to have trace_sink as output
    for node_uid in leaf_nodes:
        node = dag.get_node(node_uid)
        if node:
            dag.update_node(
                node_uid,
                output_pointers=node.output_pointers + [sink_uid],
                output_edge_types=node.output_edge_types + [EdgeType.TRACE_BOUNDARY],
            )

    if debug:
        print(f"Added source node connected to {len(source_nodes)} source nodes")
        print(f"Added sink node connected to {len(leaf_nodes)} leaf nodes")


def _remove_duplicate_dependencies(dag):
    """
    Remove duplicate dependencies from all nodes in the DAG.
    If a node has the same UID appearing multiple times in input_pointers or output_pointers,
    keep only one instance and its corresponding edge type.

    Also removes self-loops (nodes pointing to themselves) which create cycles.

    Args:
        dag (DAG): The DAG object to clean up.
    """
    duplicate_count = 0
    self_loop_count = 0

    for node in dag.nodes:
        # Remove duplicate input pointers and self-loops
        if node.input_pointers:
            seen = set()
            new_input_pointers = []
            new_input_edge_types = []

            for i, uid in enumerate(node.input_pointers):
                # Skip self-loops
                if uid == node.dag_uid:
                    self_loop_count += 1
                    continue

                if uid not in seen:
                    seen.add(uid)
                    new_input_pointers.append(uid)
                    new_input_edge_types.append(node.input_edge_types[i])
                else:
                    duplicate_count += 1

            if len(new_input_pointers) < len(node.input_pointers):
                dag.update_node(
                    node.dag_uid,
                    input_pointers=new_input_pointers,
                    input_edge_types=new_input_edge_types,
                )

        # Remove duplicate output pointers and self-loops
        if node.output_pointers:
            seen = set()
            new_output_pointers = []
            new_output_edge_types = []

            for i, uid in enumerate(node.output_pointers):
                # Skip self-loops
                if uid == node.dag_uid:
                    self_loop_count += 1
                    continue

                if uid not in seen:
                    seen.add(uid)
                    new_output_pointers.append(uid)
                    new_output_edge_types.append(node.output_edge_types[i])
                else:
                    duplicate_count += 1

            if len(new_output_pointers) < len(node.output_pointers):
                dag.update_node(
                    node.dag_uid,
                    output_pointers=new_output_pointers,
                    output_edge_types=new_output_edge_types,
                )

    print(f"Removed {duplicate_count} duplicate dependencies from DAG")
    print(f"Removed {self_loop_count} self-loop edges from DAG")


def _get_directly_launched_gpu_kernels(event, uid_to_event):
    """
    Get GPU kernels that are directly launched from this CPU op.
    Only looks at immediate cuda_runtime/cuda_driver children, not nested CPU ops.

    Args:
        event: The CPU operation event
        uid_to_event: Dictionary mapping UIDs to events

    Returns:
        List of GPU kernel UIDs that are directly launched by this CPU op
    """
    gpu_kernels = []

    # Get all immediate children
    children_uids = event.get("children", [])

    for child_uid in children_uids:
        child = uid_to_event.get(child_uid)
        if not child:
            continue

        child_cat = child.get("cat", "")

        # Only look at cuda_runtime or cuda_driver events (not nested CPU ops)
        if child_cat in {"cuda_runtime", "cuda_driver"}:
            runtime_children = child.get("children", [])
            for runtime_child_uid in runtime_children:
                runtime_child = uid_to_event.get(runtime_child_uid)
                if runtime_child and runtime_child.get("cat") in {
                    "kernel",
                    "gpu_memset",
                    "gpu_memcpy",
                }:
                    gpu_kernels.append(runtime_child_uid)

    return gpu_kernels


def _preprocess_trace_indices(tree1):
    """
    Preprocess trace to build indices for dependency functions.

    Args:
        tree1: The trace tree containing events

    Returns:
        Dictionary with uid_to_event, cpu_ops_by_thread, gpu_events_by_stream, root_ops_by_thread
    """
    from collections import defaultdict

    uid_to_event = {}
    cpu_ops_by_thread = defaultdict(list)  # (pid, tid) -> [(ts, uid), ...]
    gpu_events_by_stream = defaultdict(list)  # stream -> [event, ...]
    root_ops_by_thread = defaultdict(list)  # (pid, tid) -> [(ts, uid), ...]

    # SINGLE PASS: Build all indices at once
    for event in tree1.events:
        uid = event.get("UID")
        if uid is None:
            continue

        uid_to_event[uid] = event

        cat = event.get("cat", "")
        ts = event.get("ts", 0)
        pid = event.get("pid")
        tid = event.get("tid")

        # Index CPU ops by thread
        if cat == "cpu_op":
            cpu_ops_by_thread[(pid, tid)].append((ts, uid))

        # Index GPU events by stream
        if cat in {"kernel", "gpu_memset", "gpu_memcpy"}:
            stream = event.get("args", {}).get("stream")
            if stream is not None:
                gpu_events_by_stream[stream].append(event)

        # Index root ops by thread
        if event.get("parent") == -1:
            root_ops_by_thread[(pid, tid)].append((ts, uid))

    # Sort each thread's ops by timestamp (needed for sequential dependencies)
    for thread_ops in cpu_ops_by_thread.values():
        thread_ops.sort(key=lambda x: x[0])
    for thread_ops in root_ops_by_thread.values():
        thread_ops.sort(key=lambda x: x[0])

    return {
        "uid_to_event": uid_to_event,
        "cpu_ops_by_thread": cpu_ops_by_thread,
        "gpu_events_by_stream": gpu_events_by_stream,
        "root_ops_by_thread": root_ops_by_thread,
    }


def _add_data_dependencies(dag, dependency_map):
    """
    Add data dependencies between nodes based on the dependency map.

    Args:
        dag (DAG): The DAG object to update.
        dependency_map: Dictionary mapping UIDs to their data dependencies.
    """
    from collections import defaultdict

    # Build reverse dependency map for output pointers
    reverse_dependency_map = defaultdict(list)
    for uid, deps in dependency_map.items():
        for dep_uid in deps:
            reverse_dependency_map[dep_uid].append(uid)

    # Process forward dependencies (input pointers)
    for uid in dependency_map:
        node = dag.get_node(uid)
        if not node:
            continue

        input_pointers = node.input_pointers.copy()
        input_edge_types = node.input_edge_types.copy()

        # Add forward dependencies (input pointers)
        for dep_uid in dependency_map.get(uid, []):
            if dag.get_node(dep_uid):
                input_pointers.append(dep_uid)
                input_edge_types.append(EdgeType.DATA_DEPENDENCY)

        dag.update_node(
            uid,
            input_pointers=input_pointers,
            input_edge_types=input_edge_types,
        )

    # Process reverse dependencies (output pointers)
    # Uses pre-built index for efficiency
    for uid, dependents in reverse_dependency_map.items():
        node = dag.get_node(uid)
        if not node:
            continue

        output_pointers = node.output_pointers.copy()
        output_edge_types = node.output_edge_types.copy()

        # Add reverse dependencies (output pointers)
        for dependent_uid in dependents:
            if dag.get_node(dependent_uid):
                output_pointers.append(dependent_uid)
                output_edge_types.append(EdgeType.DATA_DEPENDENCY)

        dag.update_node(
            uid,
            output_pointers=output_pointers,
            output_edge_types=output_edge_types,
        )


def _add_sequential_cpu_thread_dependencies(dag, root_ops_by_thread, debug=False):
    """
    Add sequential CPU thread dependencies.

    Args:
        dag (DAG): The DAG object to update.
        root_ops_by_thread: Pre-built dict (pid, tid) -> [(ts, uid), ...]
        debug (bool): Whether to print debug output.
    """
    # root_ops_by_thread already has sorted ops, just add dependencies
    for thread_key, ops in root_ops_by_thread.items():
        # Connect consecutive operations
        for i in range(len(ops) - 1):
            current_uid = ops[i][1]
            next_uid = ops[i + 1][1]

            current_node = dag.get_node(current_uid)
            next_node = dag.get_node(next_uid)

            if current_node and next_node:
                # Add dependency: current -> next
                dag.update_node(
                    next_uid,
                    input_pointers=next_node.input_pointers + [current_uid],
                    input_edge_types=next_node.input_edge_types
                    + [EdgeType.INTRATHREAD_CPU_DEPENDENCY],
                )

                dag.update_node(
                    current_uid,
                    output_pointers=current_node.output_pointers + [next_uid],
                    output_edge_types=current_node.output_edge_types
                    + [EdgeType.INTRATHREAD_CPU_DEPENDENCY],
                )

    if debug:
        print(
            f"Added sequential thread dependencies for {len(root_ops_by_thread)} threads"
        )


def _add_intrathread_gpu_dependencies(dag, gpu_events_by_stream):
    """
    Add intrathread GPU dependencies.

    Args:
        dag (DAG): The DAG object to update.
        gpu_events_by_stream: Pre-built dict stream -> [events, ...]
    """
    for stream, events in gpu_events_by_stream.items():
        # Events already grouped, just sort by timestamp
        gpu_events_sorted = sorted(events, key=lambda e: e.get("ts", 0))

        # Add GPU-GPU intrathread dependencies within the stream
        for i in range(len(gpu_events_sorted) - 1):
            current_event = gpu_events_sorted[i]
            next_event = gpu_events_sorted[i + 1]

            current_uid = current_event.get("UID")
            next_uid = next_event.get("UID")

            current_dag_node = dag.get_node(current_uid)
            next_dag_node = dag.get_node(next_uid)

            if current_dag_node and next_dag_node:
                # Add the dependency: current -> next
                dag.update_node(
                    next_uid,
                    input_pointers=next_dag_node.input_pointers
                    + [current_dag_node.dag_uid],
                    input_edge_types=next_dag_node.input_edge_types
                    + [EdgeType.INTRATHREAD_GPU_DEPENDENCY],
                )

                dag.update_node(
                    current_uid,
                    output_pointers=current_dag_node.output_pointers
                    + [next_dag_node.dag_uid],
                    output_edge_types=current_dag_node.output_edge_types
                    + [EdgeType.INTRATHREAD_GPU_DEPENDENCY],
                )


def _add_cpu_gpu_dependencies(dag, uid_to_event):
    """
    Add CPU→GPU dependencies.

    Args:
        dag (DAG): The DAG object to update.
        uid_to_event: Pre-built dict UID -> event
    """
    kernel_launch_delay_placeholder = 10

    # Only iterate CPU ops (filtered from uid_to_event)
    for uid, event in uid_to_event.items():
        if event.get("cat") != "cpu_op":
            continue

        # Get GPU kernels directly launched by this CPU op
        gpu_children = _get_directly_launched_gpu_kernels(event, uid_to_event)

        cpu_dag_node = dag.get_node(uid)

        if cpu_dag_node:
            for gpu_uid in gpu_children:
                gpu_dag_node = dag.get_node(gpu_uid)

                if gpu_dag_node:
                    # Create a kernel launch node for the kernel launch delay
                    kl_uid = f"kl_{uid}_{gpu_uid}"
                    kl_node = DAGNode(
                        treenode_uid=kl_uid,
                        name=f"Kernel Launch ({uid} -> {gpu_uid})",
                        category="kernel_launch",
                        measured_latency=kernel_launch_delay_placeholder,
                        input_pointers=[cpu_dag_node.dag_uid],
                        output_pointers=[gpu_dag_node.dag_uid],
                        input_edge_types=[EdgeType.CPU_GPU_DEPENDENCY],
                        output_edge_types=[EdgeType.CPU_GPU_DEPENDENCY],
                    )

                    # Add the kernel launch node to the DAG
                    dag.add_node(kl_node)

                    # Update the CPU and GPU nodes to connect through the kernel launch node
                    dag.update_node(
                        uid,
                        output_pointers=cpu_dag_node.output_pointers + [kl_uid],
                        output_edge_types=cpu_dag_node.output_edge_types
                        + [EdgeType.CPU_GPU_DEPENDENCY],
                    )

                    dag.update_node(
                        gpu_uid,
                        input_pointers=gpu_dag_node.input_pointers + [kl_uid],
                        input_edge_types=gpu_dag_node.input_edge_types
                        + [EdgeType.CPU_GPU_DEPENDENCY],
                    )


def _add_device_sync_dependencies(dag, uid_to_event, cpu_ops_by_thread, debug=False):
    """
    Add device sync dependencies.

    Args:
        dag (DAG): The DAG object to update.
        uid_to_event: Pre-built dict UID -> event
        cpu_ops_by_thread: Pre-built dict (pid, tid) -> [(ts, uid), ...]
        debug (bool): Whether to print debug output.
    """
    # Only iterate through events once (via uid_to_event)
    for uid, event in uid_to_event.items():
        event_name = event.get("name", "")

        # Check if this is a device sync operation
        if "DeviceSynchronize" not in event_name:
            continue

        sync_node = dag.get_node(uid)
        if not sync_node:
            continue

        sync_ts = event.get("ts", 0)
        sync_tid = event.get("tid")
        sync_pid = event.get("pid")
        sync_dur = event.get("dur", 0)
        sync_end = sync_ts + sync_dur

        # O(1) lookup instead of O(n) scan!
        thread_cpu_ops = cpu_ops_by_thread.get((sync_pid, sync_tid), [])

        # Filter ops before this sync (thread_cpu_ops is pre-sorted)
        cpu_ops_before = [
            (ts, cop_uid) for ts, cop_uid in thread_cpu_ops if ts < sync_ts
        ]

        # Find the first CPU op that has GPU children (launches kernels)
        target_kernel = None
        for _, cpu_op_uid in reversed(cpu_ops_before):  # Iterate from most recent
            cpu_op_event = uid_to_event.get(cpu_op_uid)
            if not cpu_op_event:
                continue

            # Check if this CPU op has GPU children
            gpu_children = _get_directly_launched_gpu_kernels(
                cpu_op_event, uid_to_event
            )

            if gpu_children:
                # Find the kernel with the latest end time among this CPU op's kernels
                latest_kernel_end = 0
                for kernel_uid in gpu_children:
                    kernel_event = uid_to_event.get(kernel_uid)
                    if kernel_event:
                        kernel_ts = kernel_event.get("ts", 0)
                        kernel_dur = kernel_event.get("dur", 0)
                        kernel_end = kernel_ts + kernel_dur

                        if kernel_end > latest_kernel_end:
                            latest_kernel_end = kernel_end
                            target_kernel = kernel_event

                if target_kernel:
                    break

        if target_kernel:
            kernel_uid = target_kernel.get("UID")
            kernel_node = dag.get_node(kernel_uid)

            if kernel_node:
                kernel_ts = target_kernel.get("ts", 0)
                kernel_dur = target_kernel.get("dur", 0)
                kernel_end = kernel_ts + kernel_dur

                # Calculate the effective sync latency: time from kernel end to sync end
                effective_sync_latency = max(0, sync_end - kernel_end)

                # Update the sync node's latency to only reflect the wait time
                dag.update_node(uid, measured_latency=effective_sync_latency)

                # Add GPU→CPU dependency from kernel to sync
                dag.update_node(
                    kernel_uid,
                    output_pointers=kernel_node.output_pointers + [uid],
                    output_edge_types=kernel_node.output_edge_types
                    + [EdgeType.GPU_CPU_DEPENDENCY],
                )

                # Update sync's inputs
                dag.update_node(
                    uid,
                    input_pointers=sync_node.input_pointers + [kernel_uid],
                    input_edge_types=sync_node.input_edge_types
                    + [EdgeType.GPU_CPU_DEPENDENCY],
                )

                if debug:
                    print(f"Added GPU→CPU dependency: kernel {kernel_uid} → sync {uid}")
                    print(f"  Kernel end: {kernel_end:.3f}, Sync end: {sync_end:.3f}")
                    print(
                        f"  Adjusted sync latency: {effective_sync_latency:.3f} (was {sync_dur:.3f})"
                    )


def create_dag(
    tree1, dependency_map, trace_start_uid=None, trace_end_uid=None, debug=False
):
    """
    Create a DAG from the tree and dependency map with various types of dependencies.

    Args:
        tree1: The trace tree containing events and cpu_root_nodes.
        dependency_map: Dictionary mapping UIDs to their data dependencies.
        trace_start_uid (optional): UID of the CPU root node to use as trace start.
                                   If None, uses the first CPU root node by timestamp.
        trace_end_uid (optional): UID of the CPU root node to use as trace end.
                                 If None, uses the last CPU root node by timestamp.
        debug (bool): Whether to print debug output.

    Returns:
        DAG: The constructed DAG with all dependencies and source/sink nodes.
    """
    dag = DAG()

    print("Phase 1: Preprocessing trace for indices...")
    trace_indices = _preprocess_trace_indices(tree1)
    uid_to_event = trace_indices["uid_to_event"]
    cpu_ops_by_thread = trace_indices["cpu_ops_by_thread"]
    gpu_events_by_stream = trace_indices["gpu_events_by_stream"]
    root_ops_by_thread = trace_indices["root_ops_by_thread"]

    print(f"  Identified {len(uid_to_event)} total events")
    print(f"  Grouped into {len(cpu_ops_by_thread)} CPU threads")
    print(f"  Grouped into {len(gpu_events_by_stream)} GPU streams")
    print(f"  Identified {len(root_ops_by_thread)} thread root operations")

    print("Phase 2: Creating DAG nodes with control dependencies...")
    # Second pass: create nodes with control dependencies from tree structure
    # Filter out "PyTorch Profiler" nodes and CPU call stack nodes
    for event in tree1.events:
        uid = event.get("UID")
        if uid is None:
            continue
        name = event.get("name", "")
        category = event.get("cat", "")

        # Skip nodes containing "PyTorch Profiler" in the name
        if "PyTorch Profiler" in name:
            continue

        # Skip CPU call stack nodes (python_function category)
        if category in {
            "python_function",
            "ac2g",
            "fwdbwd",
            "overhead",
            "user_annotation",
            "gpu_user_annotation",
            "",
        }:
            continue

        measured_latency = event.get("dur", 0)
        parent_dependencies = event.get("parent", -1)
        # Add target_dur to target_latencies if present
        target_latencies = []
        if "target_dur" in event:
            target_latencies.append(event["target_dur"])

        # Initialize input and output pointers from tree node structure
        input_pointers = []
        input_edge_types = []
        output_pointers = []
        output_edge_types = []

        # Copy parent as input pointer (control dependency)
        parent_uid = event.get("parent")
        if parent_uid is not None and parent_uid != -1:
            parent_event = uid_to_event.get(parent_uid)
            # Only add parent if it's not a "PyTorch Profiler" node
            if parent_event and "PyTorch Profiler" not in parent_event.get("name", ""):
                input_pointers.append(parent_uid)
                input_edge_types.append(EdgeType.CONTROL_DEPENDENCY)

        # Copy children as output pointers (control dependency)
        children = event.get("children", [])
        # Filter out "PyTorch Profiler" children
        if children:
            for child_uid in children:
                child_event = uid_to_event.get(child_uid)
                if child_event and "PyTorch Profiler" not in child_event.get(
                    "name", ""
                ):
                    output_pointers.append(child_uid)
                    output_edge_types.append(EdgeType.CONTROL_DEPENDENCY)

        dag_node = DAGNode(
            treenode_uid=uid,
            name=name,
            category=category,
            measured_latency=measured_latency,
            parent_dependencies=parent_dependencies,
            target_latencies=target_latencies,
            input_pointers=input_pointers,
            output_pointers=output_pointers,
            input_edge_types=input_edge_types,
            output_edge_types=output_edge_types,
        )
        dag.add_node(dag_node)

    print(f"  Created {len(dag.nodes)} DAG nodes")

    # Phase 3: Add various types of dependencies using pre-built indices
    print("Phase 3: Adding data dependencies...")
    _add_data_dependencies(dag, dependency_map)

    print("Phase 4: Adding sequential CPU thread dependencies...")
    _add_sequential_cpu_thread_dependencies(dag, root_ops_by_thread, debug)

    print("Phase 5: Adding intrathread GPU dependencies...")
    _add_intrathread_gpu_dependencies(dag, gpu_events_by_stream)

    print("Phase 6: Adding CPU→GPU dependencies...")
    _add_cpu_gpu_dependencies(dag, uid_to_event)

    print("Phase 7: Adding device sync dependencies...")
    _add_device_sync_dependencies(dag, uid_to_event, cpu_ops_by_thread, debug)

    # Add source and sink nodes
    print("Phase 8: Adding source and sink nodes...")
    _add_source_and_sink_nodes(dag, tree1, trace_start_uid, trace_end_uid, debug)

    # Remove duplicate dependencies
    print("Phase 9: Removing duplicate dependencies...")
    _remove_duplicate_dependencies(dag)

    # Remove dangling nodes after DAG construction
    print("Phase 10: Removing dangling nodes...")
    remove_dangling_nodes(dag)

    return dag
