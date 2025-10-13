import re

def parse_tensor_info(value_lines):
    """Parse tensor information from XLA dump value lines"""
    tensor_info = []
    
    for line in value_lines:
        # Extract size, offset, and shape information
        if 'size=' in line and 'offset=' in line:
            size_match = re.search(r'size=(\d+)', line)
            offset_match = re.search(r'offset=(\d+)', line)
            shape_match = re.search(r'bf16\[([^\]]+)\]', line)
            
            if size_match and offset_match and shape_match:
                size = int(size_match.group(1))
                offset = int(offset_match.group(1))
                shape = shape_match.group(1)
                tensor_info.append({
                    'size_bytes': size,
                    'offset': offset,
                    'shape': shape,
                    'dtype': 'bf16'
                })
    
    return tensor_info

def analyze_replica_groups(replica_groups_str):
    """Analyze replica group configuration"""
    # Parse replica groups from string format
    groups = re.findall(r'\{([0-9,]+)\}', replica_groups_str)
    
    analysis = {
        'num_groups': len(groups),
        'group_size': len(groups[0].split(',')) if groups else 0,
        'total_devices': sum(len(group.split(',')) for group in groups),
        'groups': [list(map(int, group.split(','))) for group in groups]
    }
    
    return analysis

# Parse the all-gather-start.28 information
value_lines = [
    "value: <13398 all-gather-start.28{} @0> (size=16,offset=125184): ((bf16[1,768,48,128]{3,2,0,1}, bf16[1,768,8,128]{3,2,0,1}, bf16[1,768,8,128]{3,2,0,1}), (bf16[1,6144,48,128]{3,2,0,1}, bf16[1,6144,8,128]{3,2,0,1}, bf16[1,6144,8,128]{3,2,0,1}))",
    "value: <13401 all-gather-start.28{1,0} @0> (size=75497472,offset=103423377280): bf16[1,6144,48,128]{3,2,0,1}",
    "value: <13402 all-gather-start.28{1,1} @0> (size=12582912,offset=103612120960): bf16[1,6144,8,128]{3,2,0,1}",
    "value: <13403 all-gather-start.28{1,2} @0> (size=12582912,offset=103599538048): bf16[1,6144,8,128]{3,2,0,1}"
]

replica_groups_str = "{{0,1,2,3,4,5,6,7},{8,9,10,11,12,13,14,15},{16,17,18,19,20,21,22,23},{24,25,26,27,28,29,30,31}}"

# Analyze the data
tensor_info = parse_tensor_info(value_lines)
replica_analysis = analyze_replica_groups(replica_groups_str)

print("Tensor Information:")
for i, tensor in enumerate(tensor_info):
    size_mb = tensor['size_bytes'] / (1024 * 1024)
    print(f"  Tensor {i}: {tensor['shape']} ({tensor['dtype']}) - {size_mb:.2f} MB")

print(f"\nReplica Group Analysis:")
print(f"  Number of groups: {replica_analysis['num_groups']}")
print(f"  Devices per group: {replica_analysis['group_size']}")
print(f"  Total devices: {replica_analysis['total_devices']}")
print(f"  Groups: {replica_analysis['groups']}")

# Calculate total data size
total_size_bytes = sum(tensor['size_bytes'] for tensor in tensor_info)
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"\nTotal data size: {total_size_mb:.2f} MB ({total_size_bytes:,} bytes)")