#!/usr/bin/env python3
"""
Find events with correlation_id == 5533 and trace their complete ancestry to the root
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.util import DataLoader, TraceEventUtils
from TraceLens.TreePerf.jax_analyses import JaxAnalyses

def trace_ancestry_to_root(event_uid, events_by_uid, event_to_category):
    """Trace an event's complete ancestry path to the root"""
    ancestry_path = []
    current_uid = event_uid
    
    while current_uid is not None:
        if current_uid not in events_by_uid:
            break
            
        current_event = events_by_uid[current_uid]
        classification = event_to_category(current_event)
        
        ancestry_path.append({
            'uid': current_uid,
            'event': current_event,
            'classification': classification,
            'name': current_event.get('name', 'UNKNOWN'),
            'level': len(ancestry_path)
        })
        
        # Find parent (event that has this UID in its children)
        parent_uid = None
        for other_uid, other_event in events_by_uid.items():
            if current_uid in other_event.get('children', []):
                parent_uid = other_uid
                break
        
        current_uid = parent_uid
    
    return ancestry_path

def find_correlation_5533_with_ancestry():
    """Find events with correlation_id == 5533 and show complete ancestry"""
    print("🔍 Finding events with correlation_id == 5533 and tracing ancestry")
    print("=" * 70)
    
    # Load raw data first for fast search
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading raw data: {xplane_path}")
    
    data = DataLoader.load_data(xplane_path)
    trace_events = data['traceEvents']
    
    print(f"📊 Total raw events: {len(trace_events):,}")
    
    # Get categorizer
    categorizer = JaxAnalyses.prepare_event_categorizer(trace_events)
    non_metadata_events = TraceEventUtils.non_metadata_events(trace_events)
    
    print(f"📊 Non-metadata events: {len(non_metadata_events):,}")
    
    # Quick search for correlation_id == 5533 in raw data
    target_correlation_id = 5533
    raw_matching_events = []
    
    print(f"\n🔍 Quick search for correlation_id == {target_correlation_id} in raw data...")
    
    for event in non_metadata_events:
        args = event.get('args', {})
        if args.get('correlation_id') == target_correlation_id:
            raw_matching_events.append(event)
    
    print(f"📊 Found {len(raw_matching_events)} raw events with correlation_id == {target_correlation_id}")
    
    if len(raw_matching_events) == 0:
        # Show some example correlation_ids
        print("\n🔍 Sample correlation_ids found in trace:")
        correlation_ids = set()
        for i, event in enumerate(non_metadata_events[:10000]):
            args = event.get('args', {})
            if 'correlation_id' in args:
                correlation_ids.add(args['correlation_id'])
                if len(correlation_ids) >= 20:
                    break
        
        sorted_ids = sorted(list(correlation_ids))
        print(f"   First 20 correlation_ids: {sorted_ids}")
        return
    
    # Show raw event details
    for i, event in enumerate(raw_matching_events):
        print(f"\n📋 Raw Event #{i+1}:")
        print(f"   Name: {event.get('name', 'UNKNOWN')}")
        print(f"   PID: {event.get('pid')}, TID: {event.get('tid')}")
        print(f"   Phase: {event.get('ph')}")
        print(f"   Timestamp: {event.get('ts')}")
        print(f"   Duration: {event.get('dur')}")
        print(f"   🏷️  Raw classifier: '{categorizer(event)}'")
        
        # Show key args
        args = event.get('args', {})
        print(f"   📋 Key Args:")
        for key in ['correlation_id', 'hlo_op', 'hlo_module', 'tf_op']:
            if key in args:
                value = args[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"      {key}: {value[:97]}...")
                else:
                    print(f"      {key}: {value}")
    
    # Now build the tree and find these events
    print(f"\n🌳 Building tree and tracing ancestry...")
    from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Tree constructed with {len(perf_analyzer.tree.events):,} events")
    
    # Find matching events in the tree
    for i, raw_event in enumerate(raw_matching_events):
        print(f"\n🌳 ANCESTRY TRACE for Raw Event #{i+1}: {raw_event.get('name', 'UNKNOWN')}")
        print("=" * 80)
        
        # Find this event in the tree
        tree_event = None
        for tree_ev in perf_analyzer.tree.events:
            # Match by key properties
            if (tree_ev.get('name') == raw_event.get('name') and
                tree_ev.get('ts') == raw_event.get('ts') and
                tree_ev.get('pid') == raw_event.get('pid') and
                tree_ev.get('tid') == raw_event.get('tid')):
                
                # Verify correlation_id if present in tree event
                tree_args = tree_ev.get('args', {})
                if tree_args.get('correlation_id') == target_correlation_id:
                    tree_event = tree_ev
                    break
        
        if not tree_event:
            print("❌ Event not found in constructed tree (may have been filtered)")
            continue
        
        print(f"✅ Found in tree with UID: {tree_event.get('UID')}")
        
        # Trace complete ancestry
        ancestry = trace_ancestry_to_root(
            tree_event.get('UID'), 
            perf_analyzer.tree.events_by_uid,
            perf_analyzer.event_to_category
        )
        
        print(f"\n📈 Complete ancestry path (from target event to root):")
        print(f"   Path length: {len(ancestry)} levels")
        
        for j, ancestor in enumerate(ancestry):
            indent = "   " + "  " * j
            level_info = f"Level {ancestor['level']}"
            
            print(f"{indent}├─ {level_info}: {ancestor['name']}")
            print(f"{indent}   UID: {ancestor['uid']}")
            print(f"{indent}   🏷️  Classification: '{ancestor['classification']}'")
            print(f"{indent}   PID: {ancestor['event'].get('pid')}, TID: {ancestor['event'].get('tid')}")
            print(f"{indent}   Timestamp: {ancestor['event'].get('ts')}")
            
            # Show children count (except for the target event itself)
            if j > 0:  # Not the target event
                children = ancestor['event'].get('children', [])
                print(f"{indent}   Children: {len(children)}")
            
            # Show key args for interesting events
            ancestor_args = ancestor['event'].get('args', {})
            interesting_args = {}
            for key in ['correlation_id', 'hlo_op', 'hlo_module', 'tf_op', 'Input Dims']:
                if key in ancestor_args:
                    value = ancestor_args[key]
                    if isinstance(value, str) and len(value) > 50:
                        interesting_args[key] = value[:47] + "..."
                    else:
                        interesting_args[key] = value
            
            if interesting_args:
                print(f"{indent}   Key Args: {interesting_args}")
            
            print()  # Empty line between levels
        
        # Show the root event details
        if ancestry:
            root = ancestry[-1]
            print(f"🌲 ROOT EVENT DETAILS:")
            print(f"   This ancestry chain reaches root: {root['name']}")
            print(f"   Root classification: '{root['classification']}'")
            print(f"   Root UID: {root['uid']}")
            
            # Check if root has any parent (should be None for true root)
            root_has_parent = False
            for other_event in perf_analyzer.tree.events:
                if root['uid'] in other_event.get('children', []):
                    root_has_parent = True
                    break
            
            if root_has_parent:
                print(f"   ⚠️  Warning: Root event has a parent (unexpected)")
            else:
                print(f"   ✅ Confirmed: This is a true root event")

if __name__ == "__main__":
    find_correlation_5533_with_ancestry()