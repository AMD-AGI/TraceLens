import json
import uuid
from collections import defaultdict, deque
from enum import Enum


class EdgeType(Enum):
    DATA_DEPENDENCY = "data_dependency"
    CONTROL_DEPENDENCY = "control_dependency"
    CPU_GPU_DEPENDENCY = "cpu_gpu_dependency"  # CPU launching GPU kernel
    GPU_CPU_DEPENDENCY = "gpu_cpu_dependency"  # GPU kernel completing before CPU sync
    INTRATHREAD_CPU_DEPENDENCY = (
        "intrathread_cpu_dependency"  # New type for intrathread CPU dependencies
    )
    INTRATHREAD_GPU_DEPENDENCY = (
        "intrathread_gpu_dependency"  # New type for intrathread GPU dependencies
    )
    TRACE_BOUNDARY = (
        "trace_boundary"  # Edges connecting trace source/sink to actual trace nodes
    )
    OTHER = "other"  # Placeholder for additional edge types


try:
    from graphviz import Digraph

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


class DAGNode:
    def __init__(
        self,
        treenode_uid,
        name,
        category,
        measured_latency,
        parent_dependencies=None,
        target_latencies=None,
        input_pointers=None,
        output_pointers=None,
        input_edge_types=None,  # Updated to use EdgeType
        output_edge_types=None,  # Updated to use EdgeType
    ):
        self.dag_uid = treenode_uid  # node unique id is treenode_uid
        self.treenode_uid = treenode_uid
        self.name = name
        self.category = category
        self.measured_latency = measured_latency
        self.parent_dependencies = parent_dependencies or []  # list of node uids
        self.target_latencies = target_latencies or []
        self.input_pointers = input_pointers or []
        self.output_pointers = output_pointers or []
        self.input_edge_types = input_edge_types or []  # Initialize as empty list
        self.output_edge_types = output_edge_types or []  # Initialize as empty list

    def to_dict(self):
        return {
            "dag_uid": self.dag_uid,
            "treenode_uid": self.treenode_uid,
            "name": self.name,
            "category": self.category,
            "measured_latency": self.measured_latency,
            "parent_dependencies": self.parent_dependencies,
            "target_latencies": self.target_latencies,
            "input_pointers": self.input_pointers,
            "output_pointers": self.output_pointers,
            "input_edge_types": [
                edge_type.value for edge_type in self.input_edge_types
            ],
            "output_edge_types": [
                edge_type.value for edge_type in self.output_edge_types
            ],
        }


class DAG:
    def __init__(self):
        self.nodes = []  # Changed from dictionary to a list of DAGNode objects
        self.uid_to_dagnode = {}  # Mapping for lookups

    def add_node(self, node: DAGNode):
        self.nodes.append(node)
        self.uid_to_dagnode[node.dag_uid] = node

    def get_node(self, uid):
        return self.uid_to_dagnode.get(uid)

    def update_node(self, uid, **kwargs):
        """
        Update the attributes of a node and ensure changes are reflected in both
        the list of nodes and the uid_to_dagnode mapping.
        """
        node = self.get_node(uid)
        if not node:
            return

        for key, value in kwargs.items():
            if hasattr(node, key):
                setattr(node, key, value)

    def to_dict(self):
        return [
            node.to_dict() for node in self.nodes
        ]  # Updated to iterate over the list

    def visualize_dag_graphviz(
        self, filename="dag_graph", max_nodes=None, filter_categories=None
    ):
        """
        Visualize the DAG using Graphviz.

        Args:
            filename: Output filename (without extension)
            max_nodes: Limit visualization to first N nodes (None = all nodes)
            filter_categories: List of categories to include (None = all categories)
        """
        if not GRAPHVIZ_AVAILABLE:
            print("Warning: graphviz not available. Skipping visualization.")
            return

        dot = Digraph(comment="DAG Graph")
        dot.attr(rankdir="TB")  # Top to Bottom layout
        dot.attr(concentrate="true")  # Merge parallel edges
        dot.attr(fontsize="10")
        dot.attr(ranksep="0.5")

        # Filter nodes if needed
        nodes_to_visualize = self.nodes
        if filter_categories:
            nodes_to_visualize = [
                n for n in nodes_to_visualize if n.category in filter_categories
            ]
        if max_nodes:
            nodes_to_visualize = nodes_to_visualize[:max_nodes]

        node_uid_set = {n.dag_uid for n in nodes_to_visualize}

        print(
            f"Visualizing {len(nodes_to_visualize)} nodes out of {len(self.nodes)} total nodes"
        )
        print("Nodes:")

        for node in nodes_to_visualize:
            # Shorten label for readability
            name_short = node.name[:50] + "..." if len(node.name) > 50 else node.name
            label = f"{name_short}\\n({node.dag_uid})\\n{node.measured_latency:.3f}ms"

            # Color by category
            color_map = {
                "cpu_op": "lightblue",
                "kernel": "lightgreen",
                "user_annotation": "lightyellow",
                "profiler": "lightgray",
            }
            color = color_map.get(node.category, "white")
            dot.node(
                str(node.dag_uid), label, style="filled", fillcolor=color, fontsize="8"
            )

            if len(nodes_to_visualize) <= 100:  # Only print for small graphs
                print(f"  Node: {node.dag_uid}, label: {name_short}")

        print("Edges:")
        edge_count = 0
        for node in nodes_to_visualize:
            for dep_uid in node.input_pointers:
                # Only draw edges where both nodes are in visualization
                if dep_uid in node_uid_set:
                    dot.edge(str(dep_uid), str(node.dag_uid))
                    edge_count += 1

                    if len(nodes_to_visualize) <= 100:  # Only print for small graphs
                        print(f"  Edge: {dep_uid} -> {node.dag_uid}")

        print(f"Total edges: {edge_count}")

        # Export DOT source
        with open(filename + ".dot", "w") as f:
            f.write(dot.source)

    def find_critical_path(self):
        """Find the critical path in the DAG."""

        def get_node_name(uid):
            """Helper to get readable node name."""
            node = self.get_node(uid)
            if node:
                return f"{node.name[:60]}{'...' if len(node.name) > 60 else ''} (UID:{uid})"
            return f"UNKNOWN (UID:{uid})"

        print("=== Critical Path Calculation ===")
        print(f"Total nodes in DAG: {len(self.nodes)}")

        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        longest_path = {}
        topological_order = []

        # Compute in-degrees and out-degrees for each node
        print("Phase 1: Computing in-degrees and out-degrees...")

        for node in self.nodes:
            out_degree[node.dag_uid] = len(node.output_pointers)
            for neighbor in node.output_pointers:
                in_degree[neighbor] += 1

        # Find nodes with in-degree 0 and out-degree 0
        source_nodes = [
            node.dag_uid for node in self.nodes if in_degree[node.dag_uid] == 0
        ]
        sink_nodes = [
            node.dag_uid for node in self.nodes if out_degree[node.dag_uid] == 0
        ]

        # Perform topological sort using Kahn's algorithm
        print("Phase 2: Topological sort using Kahn's algorithm...")

        queue = deque()
        for node in self.nodes:
            if in_degree[node.dag_uid] == 0:
                queue.append(node.dag_uid)

        while queue:
            current = queue.popleft()
            topological_order.append(current)

            current_node = self.get_node(current)
            if current_node is None:
                continue

            for neighbor in current_node.output_pointers:
                neighbor_node = self.get_node(neighbor)
                if neighbor_node is None:
                    continue

                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        print(f"  Topological order computed: {len(topological_order)} nodes")

        # Initialize the longest path for each node to 0
        # The measured_latency will be added as edge weights during the forward pass
        print("Phase 3: Initializing longest path values...")
        predecessor = {}  # Maps node_uid -> predecessor_uid
        for node in self.nodes:
            longest_path[node.dag_uid] = 0
            predecessor[node.dag_uid] = None

        # Process nodes in topological order
        print("Phase 4: Computing longest paths...")
        for i, node_uid in enumerate(topological_order):
            current_node = self.get_node(node_uid)
            if current_node is None:
                continue

            for idx, neighbor_uid in enumerate(current_node.output_pointers):
                neighbor_node = self.get_node(neighbor_uid)
                if neighbor_node is None:
                    continue

                # Determine edge weight based on edge type
                # Get the edge type for this specific edge
                edge_type = (
                    current_node.output_edge_types[idx]
                    if idx < len(current_node.output_edge_types)
                    else EdgeType.OTHER
                )

                # Determine edge weight based on edge type and node category
                if edge_type == EdgeType.CONTROL_DEPENDENCY:
                    edge_weight = 0  # Don't add latency for control dependencies
                elif (
                    edge_type == EdgeType.CPU_GPU_DEPENDENCY
                    and neighbor_node.category == "kernel_launch"
                ):
                    # For CPU -> kernel_launch edge, don't add CPU latency
                    edge_weight = 0
                else:
                    # For all other edges (DATA_DEPENDENCY, TRACE_BOUNDARY, etc.), add latency
                    edge_weight = current_node.measured_latency

                old_path = longest_path[neighbor_uid]
                new_path = longest_path[node_uid] + edge_weight

                # Update the longest path for the neighbor and store predecessor
                if new_path > longest_path[neighbor_uid]:
                    longest_path[neighbor_uid] = new_path
                    predecessor[neighbor_uid] = node_uid

        print(f"  Longest paths computed for {len(longest_path)} nodes")

        # Find the trace sink node as the critical path endpoint
        print("Phase 5: Finding critical path endpoint...")

        # Look for the trace sink node specifically
        trace_sink_uid = "trace_sink"
        sink_node = self.get_node(trace_sink_uid)

        if sink_node and trace_sink_uid in longest_path:
            critical_path_end = trace_sink_uid

        else:
            # Fallback to maximum path length node if sink not found
            critical_path_end = max(
                (node.dag_uid for node in self.nodes if node.dag_uid in longest_path),
                key=lambda uid: longest_path[uid],
                default=None,
            )

        if critical_path_end is None:
            return []

        critical_path = []
        current = critical_path_end

        # Backtrack to find the critical path
        print("Phase 6: Backtracking to reconstruct critical path...")
        step = 0
        while current is not None:
            step += 1
            critical_path.append(current)
            current_node = self.get_node(current)
            if current_node is None:
                break

            # Check if we've reached the trace source
            if current == "trace_source":
                break

            # Follow the stored predecessor
            next_node = predecessor.get(current)

            if next_node:
                current = next_node
            else:
                current = None

        final_path = list(reversed(critical_path))
        print(f"  Critical path reconstructed: {len(final_path)} nodes")

        # Calculate total path length for verification
        total_length = sum(
            node.measured_latency
            for uid in final_path
            if (node := self.get_node(uid)) is not None
        )
        print(f"Total critical path latency: {total_length:.3f} ms")

        return final_path

    def analyze_isolated_nodes(self, tree1=None):
        """
        Analyze nodes with zero input pointers to understand why they exist.
        """
        isolated_nodes = [node for node in self.nodes if len(node.input_pointers) == 0]

        print(f"\n=== ISOLATED NODES ANALYSIS ===")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Nodes with 0 input pointers: {len(isolated_nodes)}")

        if not isolated_nodes:
            print("✓ No isolated nodes found!")
            return

        # Categorize isolated nodes
        categories = defaultdict(list)
        threads = defaultdict(list)

        for node in isolated_nodes:
            categories[node.category].append(node)

            # Extract thread info if available
            if hasattr(node, "name") and node.name:
                # Look for thread info in name
                if "tid:" in node.name:
                    tid = node.name.split("tid:")[1].split()[0]
                    threads[f"tid_{tid}"].append(node)

        print(f"\nIsolated nodes by category:")
        for category, nodes in categories.items():
            print(f"  {category}: {len(nodes)} nodes")

        print(f"\nIsolated nodes by thread:")
        for thread, nodes in threads.items():
            print(f"  {thread}: {len(nodes)} nodes")

        # Build a UID->event map once for efficient lookup
        tree_event_map = {}
        if tree1 is not None:
            tree_event_map = {e.get("UID"): e for e in tree1.events}

        # Check if CPU ops are in cpu_root_nodes
        if tree1 is not None:
            cpu_isolated = [n for n in isolated_nodes if n.category == "cpu_op"]
            cpu_root_uids = set(tree1.cpu_root_nodes)

            cpu_isolated_in_roots = [
                n for n in cpu_isolated if n.treenode_uid in cpu_root_uids
            ]
            cpu_isolated_not_in_roots = [
                n for n in cpu_isolated if n.treenode_uid not in cpu_root_uids
            ]

            print(f"\nCPU operation analysis:")
            print(f"  Total isolated cpu_op nodes: {len(cpu_isolated)}")
            print(
                f"  Isolated cpu_op nodes in cpu_root_nodes: {len(cpu_isolated_in_roots)}"
            )
            print(
                f"  Isolated cpu_op nodes NOT in cpu_root_nodes: {len(cpu_isolated_not_in_roots)}"
            )

            # Check if these CPU ops have parents in the tree
            cpu_ops_with_no_parent = []
            cpu_ops_with_parent = []

            for node in cpu_isolated:
                tree_event = tree_event_map.get(node.treenode_uid)
                if tree_event:
                    parent_uid = tree_event.get("parent")
                    if parent_uid is None or parent_uid == -1:
                        cpu_ops_with_no_parent.append(node)
                    else:
                        cpu_ops_with_parent.append(node)

            print(
                f"  Isolated cpu_op nodes with no parent in tree: {len(cpu_ops_with_no_parent)}"
            )
            print(
                f"  Isolated cpu_op nodes with parent in tree: {len(cpu_ops_with_parent)}"
            )

            # Sample a few that should have parents but don't have input pointers
            if cpu_ops_with_parent:
                print(
                    f"\nSample cpu_op nodes that have tree parents but no input pointers:"
                )
                for i, node in enumerate(cpu_ops_with_parent[:5]):
                    tree_event = tree_event_map.get(node.treenode_uid)
                    parent_uid = tree_event.get("parent") if tree_event else None
                    parent_exists = (
                        self.get_node(parent_uid) is not None if parent_uid else False
                    )

                    # Check what the parent UID value actually is
                    print(f"    {i+1}. UID: {node.dag_uid}")
                    print(f"        Tree parent raw value: {repr(parent_uid)}")
                    print(f"        Tree parent type: {type(parent_uid)}")
                    print(f"        Parent exists in DAG: {parent_exists}")
                    print(f"        Node input_pointers: {node.input_pointers}")
                    print(
                        f"        Node input_edge_types: {[et.name if hasattr(et, 'name') else et for et in node.input_edge_types]}"
                    )

                    # Check if parent is -1 or some other sentinel value
                    if parent_uid == -1:
                        print(f"        ⚠ Parent is -1 (sentinel value)")
                    elif parent_uid is None:
                        print(f"        ⚠ Parent is None")
                    else:
                        print(f"        Parent should be valid")
                    print()

        # Sample some isolated nodes for detailed inspection
        print(f"\nFirst 10 isolated nodes (detailed):")
        for i, node in enumerate(isolated_nodes[:10]):
            print(f"  {i+1}. UID: {node.dag_uid}")
            print(f"     Name: {node.name}")
            print(f"     Category: {node.category}")
            print(f"     Output pointers: {len(node.output_pointers)}")
            if hasattr(node, "treenode_uid"):
                print(f"     TreeNode UID: {node.treenode_uid}")
            print()

    def export_dag_to_json(self, output_file):
        """
        Utility function to export DAG nodes to a JSON file.

        Args:
            dag (DAG): The DAG object containing nodes to export.
            output_file (str): Path to the output JSON file.
        """
        dag_data = {
            "nodes": [
                {
                    "dag_uid": node.dag_uid,
                    "name": node.name,
                    "category": node.category,
                    "measured_latency": node.measured_latency,
                    "parent_dependencies": node.parent_dependencies,
                    "target_latencies": node.target_latencies,
                    "input_pointers": node.input_pointers,
                    "output_pointers": node.output_pointers,
                    "input_edge_types": [
                        edge_type.name for edge_type in node.input_edge_types
                    ],
                    "output_edge_types": [
                        edge_type.name for edge_type in node.output_edge_types
                    ],
                }
                for node in self.nodes
            ]
        }

        with open(output_file, "w") as f:
            json.dump(dag_data, f, indent=4)

        print(f"DAG nodes exported to {output_file}")
