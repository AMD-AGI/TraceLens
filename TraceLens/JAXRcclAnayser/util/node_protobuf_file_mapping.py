import os
import glob
from pathlib import Path
import re
from collections import defaultdict

def get_node_protobuf_mapping(traces_folder, pattern="*.xplane.pb"):
    """
    Map nodes to their corresponding protobuf files.
    
    Args:
        traces_folder: Path to the traces folder containing log files or subdirectories
        pattern: File pattern to search for protobuf files (default: "*.xplane.pb")
    
    Returns:
        dict: Mapping of node names to protobuf file paths
    """
    node_protobuf_map = {}
    
    # Convert to Path object for easier manipulation
    traces_path = Path(traces_folder)
    
    if not traces_path.exists():
        print(f"Error: Traces folder {traces_folder} does not exist")
        return node_protobuf_map
    
    print(f"Searching for protobuf files in: {traces_path}")
    
    # Method 1: Look for protobuf files in subdirectories (like your notebook example)
    # Pattern: /logs/*/node_name/*/tensorboard/plugins/profile/*/*.xplane.pb
    pb_files = list(traces_path.rglob(pattern))
    
    for pb_file in pb_files:
        # Extract node name from path
        node_name = extract_node_name(pb_file)
        if node_name:
            node_protobuf_map[node_name] = str(pb_file)
            print(f"Found: {node_name} -> {pb_file}")
    
    # Method 2: If no protobuf files found, look for log files and infer structure
    if not node_protobuf_map:
        print("No protobuf files found directly. Checking log files for node names...")
        log_files = list(traces_path.glob("*.log"))
        
        for log_file in log_files:
            # Extract node name from log file name
            # Pattern: step_X_node_NODENAME_rank_Y.log
            match = re.search(r'step_\d+_node_([^_]+(?:-[^_]+)*(?:-\d+)*(?:-\d+)*)_rank_\d+\.log', log_file.name)
            if match:
                node_name = match.group(1)
                # Look for corresponding protobuf file
                potential_pb_paths = [
                    traces_path.parent / "logs" / "*" / node_name / "*" / "tensorboard" / "plugins" / "profile" / "*" / f"{node_name}.xplane.pb",
                    traces_path / node_name / "tensorboard" / "plugins" / "profile" / "*" / f"{node_name}.xplane.pb"
                ]
                
                for pb_pattern in potential_pb_paths:
                    pb_files = glob.glob(str(pb_pattern))
                    if pb_files:
                        node_protobuf_map[node_name] = pb_files[0]
                        print(f"Inferred: {node_name} -> {pb_files[0]}")
                        break
    
    return node_protobuf_map

def extract_node_name(pb_file_path):
    """
    Extract node name from protobuf file path.
    
    Args:
        pb_file_path: Path object of the protobuf file
    
    Returns:
        str: Node name or None if not found
    """
    path_parts = pb_file_path.parts
    
    # Method 1: Node name is the parent of xla_dumps folder
    for i, part in enumerate(path_parts):
        if part == "xla_dumps" and i > 0:
            return path_parts[i-1]
    
    # Method 2: Node name is in the file name (e.g., tw026.xplane.pb)
    if pb_file_path.name.endswith('.xplane.pb'):
        node_name = pb_file_path.stem  # Remove .xplane.pb
        if node_name and not node_name.isdigit():  # Avoid pure numeric names
            return node_name
    
    # Method 3: Look for node pattern in path (e.g., /logs/*/tw026/*)
    for part in reversed(path_parts):
        # Check if part looks like a node name (starts with letters, may contain numbers/hyphens)
        if re.match(r'^[a-zA-Z][a-zA-Z0-9\-]*$', part) and len(part) > 2:
            # Skip common directory names
            if part not in ['logs', 'tensorboard', 'plugins', 'profile', 'xla_dumps']:
                return part
    
    return None

def find_all_protobuf_files(base_folder):
    """
    Find all protobuf files in the base folder and its subdirectories.
    
    Args:
        base_folder: Base folder to search in
    
    Returns:
        list: List of all protobuf file paths
    """
    base_path = Path(base_folder)
    protobuf_patterns = ["*.xplane.pb", "*.hlo_proto.pb", "*.pb"]
    
    all_pb_files = []
    for pattern in protobuf_patterns:
        pb_files = list(base_path.rglob(pattern))
        all_pb_files.extend(pb_files)
    
    return all_pb_files

def main():
    """Main function to demonstrate the mapping."""
    
    # Use the attached traces folder
    traces_folder = "/workspace/code/traces/cohere-32N-22B-927-20250820-105416"
    
    print("=" * 60)
    print("NODE TO PROTOBUF FILE MAPPING")
    print("=" * 60)
    
    # Get the mapping
    mapping = get_node_protobuf_mapping(traces_folder)
    
    if mapping:
        print(f"\nFound {len(mapping)} node-protobuf mappings:")
        print("-" * 40)
        for node, pb_file in sorted(mapping.items()):
            print(f"{node:20} -> {pb_file}")

            
    else:
        print("\nNo protobuf files found. Searching for all .pb files...")
        
        # Search in parent directories
        parent_dirs = [
            "/workspace/code/traces",
            "/workspace/code/traces/logs"
        ]
        
        for parent_dir in parent_dirs:
            if os.path.exists(parent_dir):
                print(f"\nSearching in {parent_dir}:")
                all_pb_files = find_all_protobuf_files(parent_dir)
                
                if all_pb_files:
                    print(f"Found {len(all_pb_files)} protobuf files:")
                    for pb_file in sorted(all_pb_files):
                        node_name = extract_node_name(pb_file)
                        print(f"  {node_name or 'unknown'} -> {pb_file}")
                else:
                    print("  No protobuf files found")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()