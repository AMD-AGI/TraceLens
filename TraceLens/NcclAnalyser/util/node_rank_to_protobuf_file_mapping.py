from pathlib import Path
import re
from collections import defaultdict

def get_node_rank_protobuf_mapping(traces_folder, pattern="*.xplane.pb"):
    """
    Map NODE_RANK to their corresponding protobuf files by parsing log files.
    
    Args:
        traces_folder: Path to the traces folder containing log files
        pattern: File pattern to search for protobuf files (default: "*.xplane.pb")
    
    Returns:
        dict: Mapping of NODE_RANK to protobuf file paths
        int: World size (total number of ranks found)
    """
    node_protobuf_map = {}
    node_to_rank_map = {}
    nnodes = None
    
    traces_path = Path(traces_folder)
    
    if not traces_path.exists():
        print(f"Error: Traces folder {traces_folder} does not exist")
        return node_protobuf_map, 0
    
    print(f"Parsing log files in: {traces_path}")
    
    # Step 1: Parse log files to get NODE_RANK -> node mapping and nnodes
    log_files = list(traces_path.glob("*.log"))
    
    if not log_files:        
        log_files = list(traces_path.rglob("*.log"))
    
    for log_file in log_files:
        node_rank, node_name, file_nnodes = parse_log_file_for_node_rank(log_file)
        if node_rank is not None and node_name:
            node_to_rank_map[node_name] = str(node_rank)
        
        # Get nnodes value from any log file that has it
        if file_nnodes is not None and nnodes is None:
            nnodes = file_nnodes
    
    # Step 2: Validate NODE_RANK count against nnodes
    if nnodes is not None:
        expected_ranks = nnodes
        found_ranks = len(set(node_to_rank_map.values()))
        
        if found_ranks != expected_ranks:
            print(f"Warning: NODE_RANK count ({found_ranks}) does not match nnodes ({expected_ranks})")
    
    # Step 3: Find protobuf files and match them to NODE_RANKs via nodes
    pb_files = list(traces_path.rglob(pattern))

    for pb_file in pb_files:
        node_name = extract_node_name_from_path(pb_file)
        if node_name and node_name in node_to_rank_map:
            node_rank = node_to_rank_map[node_name]
            node_protobuf_map[node_rank] = str(pb_file)
        elif node_name:
            print(f"Warning: Found protobuf for node {node_name} but no corresponding NODE_RANK in log files")
    
    world_size = nnodes if nnodes is not None else len(node_protobuf_map)
    
    return node_protobuf_map, world_size

def parse_log_file_for_node_rank(log_file):
    """
    Parse a log file to extract NODE_RANK, hostname, and nnodes.
    Looks for NODE_RANK variable value and hostname variable value in log content.
    
    Args:
        log_file: Path to the log file
    
    Returns:
        tuple: (node_rank, node_name, nnodes) or (None, None, None) if not found
    """
   
  
    # Method 1: Parse file content for NODE_RANK, hostname, and nnodes
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(5000)  # Read more content to find all variables
            
            # Look for NODE_RANK variable assignment patterns
            node_rank_patterns = [
                r'NODE_RANK\s*[=:]\s*(\d+)',        # NODE_RANK=0 or NODE_RANK: 0
                r'NODE_RANK\s*=\s*["\']?(\d+)["\']?',  # NODE_RANK="0"
                r'export\s+NODE_RANK\s*=\s*(\d+)',   # export NODE_RANK=0
                r'set\s+NODE_RANK\s*=\s*(\d+)',      # set NODE_RANK=0
                r'NODE_RANK[:\s=]+(\d+)',            # Keep original pattern
                r'node_rank[:\s=]+(\d+)'             # Keep original pattern
            ]
            
            # Look for hostname variable assignment patterns
            hostname_patterns = [
                r'hostname\s*[=:]\s*([a-zA-Z0-9\-_.]+)',      # hostname=nodeXXX
                r'HOSTNAME\s*[=:]\s*([a-zA-Z0-9\-_.]+)',      # HOSTNAME=nodeXXX
                r'export\s+HOSTNAME\s*=\s*["\']?([a-zA-Z0-9\-_.]+)["\']?',  # export HOSTNAME="nodeXXX"
                r'set\s+HOSTNAME\s*=\s*([a-zA-Z0-9\-_.]+)',   # set HOSTNAME=nodeXXX
                r'hostname:\s*([a-zA-Z0-9\-_.]+)',            # hostname: nodeXXX
                r'Host:\s*([a-zA-Z0-9\-_.]+)',                # Host: nodeXXX
                r'Node:\s*([a-zA-Z0-9\-_.]+)',                # Node: nodeXXX
                r'Running on\s+([a-zA-Z0-9\-_.]+)',           # Running on nodeXXX
                r'nodename\s*[=:]\s*([a-zA-Z0-9\-_.]+)',      # nodename=nodeXXX
                r'NODENAME\s*[=:]\s*([a-zA-Z0-9\-_.]+)'       # NODENAME=nodeXXX
            ]
            
            # Look for nnodes patterns
            nnodes_patterns = [
                r'nnodes\s*[=:]\s*(\d+)',           # nnodes=4
                r'NNODES\s*[=:]\s*(\d+)',           # NNODES=4
                r'num_nodes\s*[=:]\s*(\d+)',        # num_nodes=4
                r'--nnodes\s*[=:]\s*(\d+)',         # --nnodes=4
                r'export\s+NNODES\s*=\s*(\d+)',     # export NNODES=4
                r'nnodes[:\s=]+(\d+)',              # Keep original pattern
                r'NNODES[:\s=]+(\d+)',              # Keep original pattern
                r'num_nodes[:\s=]+(\d+)',           # Keep original pattern
                r'--nnodes[:\s=]+(\d+)'             # Keep original pattern
            ]
            
            node_rank = None
            hostname = None
            nnodes = None
            
            # Extract NODE_RANK
            for pattern in node_rank_patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    node_rank = int(match.group(1))
                    break
            
            # Extract hostname
            for pattern in hostname_patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    hostname = match.group(1).strip().strip('"\'')
                    break
            
            # Extract nnodes
            for pattern in nnodes_patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
                if match:
                    nnodes = int(match.group(1))
                    break
            
            # Return results if we found NODE_RANK
            if node_rank is not None:
                # Use hostname if found, otherwise try to extract from path
                node_name = hostname
                if not node_name:
                    node_name = extract_node_name_from_path(log_file)              
                if node_name:
                    return node_rank, node_name, nnodes
                
    except Exception as e:
        print(f"Warning: Could not read log file {log_file}: {e}")

    # Method 2: Extract NODE_RANK from filename patterns
    filename_patterns = [
        r'.*node_rank_(\d+)\.log',           # node_rank_0.log
        r'.*NODE_RANK_(\d+)\.log',           # NODE_RANK_0.log  
        r'.*_node_rank_(\d+)\.log',          # something_node_rank_0.log
        r'.*_NODE_RANK_(\d+)\.log'           # something_NODE_RANK_0.log
    ]
    
    for pattern in filename_patterns:
        filename_match = re.search(pattern, log_file.name)
        if filename_match:
            node_rank = int(filename_match.group(1))
            node_name = extract_node_name_from_path(log_file)
            if node_name:
                return node_rank, node_name, None
    
    return None, None, None

def extract_node_name_from_path(file_path):
    """
    Extract node name from file path.
    
    Args:
        file_path: Path object or string
    
    Returns:
        str: Node name or None if not found
    """
    path = Path(file_path)
    path_parts = path.parts
    
    # Method 1: Node name from filename (e.g., nodeXXX.xplane.pb -> nodeXXX)
    if path.is_file() and path.suffix in ['.pb', '.log']:
        stem = path.stem
        if stem.endswith('.xplane'):
            stem = stem[:-7]  # Remove .xplane
        
        # Check if it looks like a node name
        if re.match(r'^[a-zA-Z][a-zA-Z0-9\-]*$', stem):
            return stem
            
    return None