#!/usr/bin/env python3
"""
Quick search for correlation_id == 5533 without full tree construction
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.util import DataLoader, TraceEventUtils
from TraceLens.TreePerf.jax_analyses import JaxAnalyses

def quick_search_correlation_5533():
    """Quick search without full tree construction"""
    print("🔍 Quick search for correlation_id == 5533")
    print("=" * 50)
    
    # Load raw data
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    data = DataLoader.load_data(xplane_path)
    trace_events = data['traceEvents']
    
    print(f"📊 Total events: {len(trace_events):,}")
    
    # Get categorizer
    categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
    non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
    
    print(f"📊 Non-metadata events: {len(non_metadata_events):,}")
    
    # Search for correlation_id == 5533
    target_correlation_id = 5533
    matching_events = []
    
    print(f"\n🔍 Searching for correlation_id == {target_correlation_id}...")
    
    for event in non_metadata_events:
        args = event.get('args', {})
        if args.get('correlation_id') == target_correlation_id:
            matching_events.append(event)
    
    print(f"📊 Found {len(matching_events)} events")
    
    if len(matching_events) == 0:
        # Show sample correlation_ids
        print("\n🔍 Sample correlation_ids (first 50 found):")
        correlation_ids = set()
        for event in non_metadata_events:
            args = event.get('args', {})
            if 'correlation_id' in args:
                correlation_ids.add(args['correlation_id'])
                if len(correlation_ids) >= 50:
                    break
        
        sorted_ids = sorted(list(correlation_ids))
        print(f"   {sorted_ids}")
        return
    
    # Show details of found events
    for i, event in enumerate(matching_events):
        print(f"\n📋 Event #{i+1}:")
        print(f"   Name: {event.get('name', 'UNKNOWN')}")
        print(f"   PID: {event.get('pid')}, TID: {event.get('tid')}")
        print(f"   Phase: {event.get('ph')}")
        print(f"   Timestamp: {event.get('ts')}")
        print(f"   Duration: {event.get('dur')}")
        print(f"   🏷️  Classification: '{categorizer(event)}'")
        
        # Show all args
        args = event.get('args', {})
        print(f"   📋 Args ({len(args)} items):")
        for key, value in args.items():
            if key == 'correlation_id':
                print(f"      ✅ {key}: {value}")
            else:
                if isinstance(value, str) and len(value) > 100:
                    print(f"         {key}: {value[:97]}...")
                else:
                    print(f"         {key}: {value}")

if __name__ == "__main__":
    quick_search_correlation_5533()