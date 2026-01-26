#!/usr/bin/env python3
"""
Compare two vLLM GPU traces and align kernels between them.

This script takes two GPU trace files (e.g., from capture mode vs graph mode execution)
and compares the GPU kernels executed. It identifies "anchor" kernels (kernels that appear
in both traces with similar execution counts) and uses them to align the kernel sequences,
producing a CSV report showing kernel-by-kernel comparison.

Usage:
    python vllm_align_kernels.py <trace1.json.gz> <trace2.json.gz>

Example:
    python vllm_align_kernels.py capture_trace.json.gz graph_trace.json.gz

Output:
    Creates a CSV file (e.g., alignment_results.csv) with kernel comparisons
"""
# Standard library imports
from pprint import pprint
import json
import sys
from collections import Counter

# Third-party imports
import pandas as pd

# TraceLens imports for performance analysis
from TraceLens import TreePerfAnalyzer, GPUEventAnalyser, PerfModel
import TraceLens



# Load the two trace files using TreePerfAnalyzer
# Argument 1: First trace (e.g., capture mode trace)
path = sys.argv[1]
perf_analyzer = TreePerfAnalyzer.from_file(path, add_python_func=True)

# Argument 2: Second trace (e.g., graph mode trace or reference trace)
ref_path = sys.argv[2]
ref_perf_analyzer = TreePerfAnalyzer.from_file(ref_path, add_python_func=True)

# Display GPU timeline summaries for both traces
print("GPU timeline summary for trace 1:")
print(perf_analyzer.get_df_gpu_timeline())
print("\n \n GPU timeline summary for trace 2:")
print(ref_perf_analyzer.get_df_gpu_timeline())

# Extract the event tree from each analyzer
tree = perf_analyzer.tree
ref_tree = ref_perf_analyzer.tree



# Filter GPU events from both traces
# We focus on kernels, memory sets, and memory copies (core GPU operations)
cat_list = ["kernel", "gpu_memset", "gpu_memcpy"]
gpu_events = [e for e in tree.events if e.get("cat", "") in cat_list]
ref_gpu_events = [e for e in ref_tree.events if e.get("cat", "") in cat_list]

def get_kernel_dict(gpu_events):
    """
    Analyze GPU events and return a count of each kernel type.
    
    Args:
        gpu_events: List of GPU event dictionaries
    
    Returns:
        Counter object with kernel names as keys and execution counts as values
    
    Side effects:
        Prints a formatted table of kernel names, counts, and total durations
    """
    kernel_names = [e["name"] for e in gpu_events]
    kernel_counts = Counter(kernel_names)
    
    print("\n================================")
    for name in sorted(kernel_counts.keys()):
        count = kernel_counts[name]
        # Sum total duration across all executions of this kernel
        duration = sum([e["dur"] for e in gpu_events if e["name"] == name])
        print(f"{name[0:50]:<50} | {count} | {duration}")
    
    return kernel_counts
    
kernel_counts = get_kernel_dict(gpu_events)
ref_kernel_counts = get_kernel_dict(ref_gpu_events)

# Identify "anchor" kernels - kernels that appear in both traces with similar execution counts
# These serve as reference points for aligning the two traces
# We use a tolerance of 3 to account for minor execution count differences
anchor_kernels = [i for i in kernel_counts.keys() 
                  if i in ref_kernel_counts.keys() 
                  and abs(kernel_counts[i] - ref_kernel_counts[i]) < 3]

# Display the anchor kernels found
print("Found anchor kernels:")
for i in anchor_kernels:
    print(f"  {i[0:50]:<50} | Count: {kernel_counts[i]}")

def align_lists_by_anchors(list1, list2, anchor_set, key_func=lambda x: x.get("name")):
    """
    Align two lists using anchor points as reference markers.
    
    This function finds anchor points (known matching items) in both lists and uses them
    to create alignment segments. Events between anchors are grouped together, allowing
    comparison of kernels in corresponding execution phases.
    
    Args:
        list1: First list to align
        list2: Second list to align
        anchor_set: Set of anchor values to use as alignment reference points
        key_func: Function to extract comparison key from list items (default: get 'name' field)
    
    Returns:
        List of alignment blocks, each containing:
            - 'anchor': The anchor kernel name
            - 'list1_idx': Index of anchor in list1
            - 'list2_idx': Index of anchor in list2
            - 'list1_segment': (start, end) indices for events after this anchor in list1
            - 'list2_segment': (start, end) indices for events after this anchor in list2
    """
    # Find all anchor positions in both lists
    anchors_list1 = [(i, key_func(item)) for i, item in enumerate(list1) 
                     if key_func(item) in anchor_set]
    anchors_list2 = [(i, key_func(item)) for i, item in enumerate(list2) 
                     if key_func(item) in anchor_set]
    
    # Match anchors between lists in order
    alignment = []
    j = 0
    
    for i, (idx1, anchor_name) in enumerate(anchors_list1):
        # Find matching anchor in list2
        while j < len(anchors_list2) and anchors_list2[j][1] != anchor_name:
            j += 1
        
        if j < len(anchors_list2):
            # Create alignment block from current anchor to next anchor
            alignment.append({
                'anchor': anchor_name,
                'list1_idx': idx1,
                'list2_idx': anchors_list2[j][0],
                'list1_segment': (idx1, anchors_list1[i+1][0] if i+1 < len(anchors_list1) else len(list1)),
                'list2_segment': (anchors_list2[j][0], anchors_list2[j+1][0] if j+1 < len(anchors_list2) else len(list2))
            })
            j += 1
    
    return alignment

# Align the GPU events using the anchor kernels as reference points
alignment = align_lists_by_anchors(gpu_events, ref_gpu_events, set(anchor_kernels))
print(f"Found {len(alignment)} anchor-based alignments:")

# Build a dataframe comparing kernels from both traces
# For each alignment segment, compare the GPU events side-by-side
df_alignment = []
for align in alignment:
    # Get segment boundaries for this alignment block
    l1_start, l1_end = align['list1_segment']
    l2_start, l2_end = align['list2_segment']
    l1_len = l1_end - l1_start
    l2_len = l2_end - l2_start
    
    # Compare events in the segment
    for i in range(max(l1_len, l2_len)):
        # Get kernel names for comparison
        l1_name = gpu_events[l1_start + i].get("name") if i < l1_len else None
        l2_name = ref_gpu_events[l2_start + i].get("name") if i < l2_len else None
        
        if l1_name == l2_name:
            # Kernels match - single row entry
            row = {
                'anchor': align['anchor'],
                'Capture_idx': l1_start + i if i < l1_len else None,
                'Capture': l1_name,
                'Dur_capture': gpu_events[l1_start + i].get("dur") if i < l1_len else None,
                'Graph_idx': l2_start + i if i < l2_len else None,
                'Graph': l2_name,
                'Dur_graph': ref_gpu_events[l2_start + i].get("dur") if i < l2_len else None,
            }
            df_alignment.append(row)
        else:
            # Kernels don't match - create separate rows for each trace
            if i < l1_len and l1_name is not None:
                row1 = {
                        'anchor': align['anchor'],
                    'Capture_idx': l1_start + i,
                    'Capture': l1_name,
                    'Dur_capture': gpu_events[l1_start + i].get("dur"),
                    'Graph_idx': None,
                    'Graph': None,
                    'Dur_graph': None,
                }
                df_alignment.append(row1)
            
            if i < l2_len and l2_name is not None:
                row2 = {
                    'anchor': align['anchor'],
                    'Capture_idx': None,
                    'Capture': None,
                    'Dur_capture': None,
                    'Graph_idx': l2_start + i,
                    'Graph': l2_name,
                    'Dur_graph': ref_gpu_events[l2_start + i].get("dur"),
                }
                df_alignment.append(row2)

df = pd.DataFrame(df_alignment)
print("\n" + "="*100)
print("Alignment DataFrame:")

# Save dataframe to CSV for analysis and sharing
output_csv = 'alignment_results.csv'
df.to_csv(output_csv, index=False)
print(f"\nDataframe saved to: {output_csv}")
print(f"Total rows: {len(df)}")
print(f"\nColumns:")
print(f"  - anchor: The anchor kernel used for alignment")
print(f"  - Capture_idx: Index of the kernel in the first trace (capture mode)")
print(f"  - Capture: Kernel name from first trace (capture mode)")
print(f"  - Dur_capture: Duration (microseconds) in capture mode")
print(f"  - Graph_idx: Index of the kernel in the second trace (graph mode)")
print(f"  - Graph: Kernel name from second trace (graph mode)")  
print(f"  - Dur_graph: Duration (microseconds) in graph mode")




