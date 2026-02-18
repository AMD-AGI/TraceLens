import argparse
import json
import os
import time

from construct_dag import build_dependency_map, create_dag, load_trace_and_tree
from link import process_files as link_main, prune_spillover_kernels


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Critical Path Analysis for PyTorch Distributed Training Traces"
    )

    # Input file options: either linked file OR kineto + et files
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--linked-trace",
        help="Path to linked trace JSON file (already linked kineto + ET data)",
    )
    input_group.add_argument(
        "--kineto-trace",
        help="Path to kineto trace JSON file (requires --et-trace)",
    )

    parser.add_argument(
        "--et-trace",
        help="Path to execution trace JSON file (required with --kineto-trace)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output files will be written",
    )
    args = parser.parse_args()

    # Validate that et-trace is provided when kineto-trace is used
    if args.kineto_trace and not args.et_trace:
        parser.error("--et-trace is required when using --kineto-trace")
    if args.et_trace and not args.kineto_trace:
        parser.error("--kineto-trace is required when using --et-trace")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Determine the linked file path
    if args.linked_trace:
        # Use provided linked file directly
        linked_file = args.linked_trace
        print(f"Using provided linked trace file: {linked_file}")
    else:
        # Link kineto and ET files
        kineto_file = args.kineto_trace
        et_file = args.et_trace
        linked_file = os.path.join(args.output_dir, "linked_trace.json")

        # Step 1: Run the link script to generate the linked file
        step_start = time.time()
        link_main(kineto_file, et_file, linked_file)
        print(
            f"Step 1 (Link script) completed in {time.time() - step_start:.2f} seconds"
        )

    # Step 2: Load the trace and tree
    step_start = time.time()
    tree1, tree2 = load_trace_and_tree(linked_file)
    print(
        f"Step 2 (Load trace and tree) completed in {time.time() - step_start:.2f} seconds"
    )

    # Step 2.5: Prune spillover kernels from previous iteration
    step_start = time.time()
    prune_spillover_kernels(tree1.events)
    print(
        f"Step 2.5 (Prune spillover kernels) completed in {time.time() - step_start:.2f} seconds"
    )

    # Step 3: Build the dependency map
    step_start = time.time()
    dependency_map = build_dependency_map(tree1)
    print(
        f"Step 3 (Build dependency map) completed in {time.time() - step_start:.2f} seconds"
    )

    # Step 4: Create the DAG
    step_start = time.time()
    dag = create_dag(tree1, dependency_map)
    print(f"Step 4 (Create DAG) completed in {time.time() - step_start:.2f} seconds")

    # Step 5: Cycle Detection and Critical Path Analysis
    step_start = time.time()
    critical_path_dag_nodes = dag.find_critical_path()

    print(f"Critical path: {len(critical_path_dag_nodes)} nodes")

    # Build set of critical path UIDs for quick lookup
    critical_uids = set(critical_path_dag_nodes)

    # Build UID to event mapping
    uid_to_event = {event.get("UID"): event for event in tree1.events}

    # Collect all CPU subtrees for critical path CPU root nodes
    def get_cpu_subtree(event_uid):
        """Recursively get all CPU descendants of a CPU operation."""
        subtree = []
        event = uid_to_event.get(event_uid)
        if not event:
            return subtree

        children = event.get("children", [])
        for child_uid in children:
            child_event = uid_to_event.get(child_uid)
            if child_event and child_event.get("cat") == "cpu_op":
                subtree.append(child_uid)
                subtree.extend(get_cpu_subtree(child_uid))

        return subtree

    # Expand critical path to include CPU subtrees
    expanded_critical_uids = set(critical_uids)
    for uid in critical_uids:
        event = uid_to_event.get(uid)
        if event and event.get("cat") == "cpu_op" and event.get("parent") == -1:
            # This is a CPU root node on critical path
            cpu_subtree = get_cpu_subtree(uid)
            expanded_critical_uids.update(cpu_subtree)

    print(
        f"Expanded critical path with CPU subtrees: {len(expanded_critical_uids)} nodes"
    )

    critical_trace_nodes = []
    for dag_uid in critical_path_dag_nodes:
        dag_node = dag.get_node(dag_uid)
        if dag_node is None:
            continue
        # Use UID->event mapping for fast lookup
        trace_node = uid_to_event.get(dag_node.treenode_uid)
        if trace_node:
            trace_node["critical_path_node"] = 1  # Mark as critical path node
            critical_trace_nodes.append(trace_node)

    # Add CPU subtree nodes that aren't already in critical path
    for uid in expanded_critical_uids:
        if uid not in critical_uids:
            event = uid_to_event.get(uid)
            if event:
                event["critical_path_node"] = 1
                critical_trace_nodes.append(event)

    print(
        f"Step 5 (Analyze critical path) completed in {time.time() - step_start:.2f} seconds"
    )

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    # Write the critical trace nodes to a JSON file in Chrome Trace Format for Perfetto
    output_file = os.path.join(args.output_dir, "critical_trace_nodes.json")

    # Chrome Trace Format requires events in a "traceEvents" array
    # Also add metadata for better visualization
    chrome_trace = {
        "traceEvents": critical_trace_nodes,
        "displayTimeUnit": "ns",
        "otherData": {
            "critical_path_length": len(critical_trace_nodes),
            "total_nodes": len(tree1.events),
            "critical_path_percentage": f"{100 * len(critical_trace_nodes) / len(tree1.events):.2f}%",
        },
    }

    with open(output_file, "w") as f:
        json.dump(chrome_trace, f, indent=2)

    print(
        f"Critical trace nodes written to {output_file}. Total critical nodes: {len(critical_trace_nodes)}"
    )


if __name__ == "__main__":
    main()
