#!/usr/bin/env python3
"""
Find runtime thread IDs in JAX trace
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.util import DataLoader, TraceEventUtils
from TraceLens.TreePerf.jax_analyses import JaxAnalyses

def find_runtime_threads():
    """Find runtime thread IDs"""
    print("🔍 Finding Runtime Thread IDs")
    print("=" * 40)
    
    # Load the JAX trace data
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    data = DataLoader.load_data(xplane_path)
    trace_events = data['traceEvents']
    
    print(f"📊 Total events: {len(trace_events):,}")
    
    # Get metadata
    metadata = TraceEventUtils.get_metadata(trace_events)
    categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
    
    # Find runtime events
    runtime_events = []
    for event in trace_events[:10000]:  # Sample first 10K events
        if categorizer(event) in ['cuda_runtime', 'cuda_driver']:
            runtime_events.append(event)
    
    print(f"📊 Runtime events found: {len(runtime_events)}")
    
    if len(runtime_events) > 0:
        print("📋 First few runtime events:")
        for i, event in enumerate(runtime_events[:5]):
            print(f"   {i}: {event.get('name', 'UNKNOWN')} - PID:{event.get('pid', 'N/A')} TID:{event.get('tid', 'N/A')}")
            if event.get('pid') in metadata and event.get('tid') in metadata[event.get('pid')]:
                thread_name = metadata[event.get('pid')][event.get('tid')].get(TraceEventUtils.MetadataFields.ThreadName, 'UNKNOWN')
                print(f"       Thread name: {thread_name}")
    
    # Check what thread names exist
    print("\n📋 Available thread names:")
    thread_names = set()
    for pid in metadata:
        for tid in metadata[pid]:
            thread_name = metadata[pid][tid].get(TraceEventUtils.MetadataFields.ThreadName, 'UNKNOWN')
            thread_names.add(thread_name)
    
    for name in sorted(thread_names):
        print(f"   {name}")
    
    # Find a suitable thread for runtime events
    suitable_threads = []
    for pid in metadata:
        for tid in metadata[pid]:
            thread_name = metadata[pid][tid].get(TraceEventUtils.MetadataFields.ThreadName, 'UNKNOWN')
            if 'cuda' in thread_name.lower() or 'runtime' in thread_name.lower() or thread_name == 'GPU':
                suitable_threads.append((pid, tid, thread_name))
    
    print(f"\n📋 Suitable runtime threads: {len(suitable_threads)}")
    for pid, tid, name in suitable_threads[:10]:
        print(f"   PID:{pid} TID:{tid} - {name}")
        
    return suitable_threads

if __name__ == "__main__":
    find_runtime_threads()