#!/usr/bin/env python3
"""
Analyze events with correlation_id == 5533 in the JAX trace
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer

def analyze_correlation_5533():
    """Find and analyze events with correlation_id == 5533"""
    print("🔍 Analyzing events with correlation_id == 5533")
    print("=" * 60)
    
    # Create the analyzer
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    print(f"📁 Loading: {xplane_path}")
    
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"📊 Total events in tree: {len(perf_analyzer.tree.events):,}")
    
    # Search for events with correlation_id == 5533
    target_correlation_id = 5533
    matching_events = []
    
    print(f"\n🔍 Searching for events with correlation_id == {target_correlation_id}...")
    
    for event in perf_analyzer.tree.events:
        # Check args.correlation_id
        args = event.get('args', {})
        if args.get('correlation_id') == target_correlation_id:
            matching_events.append(event)
    
    print(f"📊 Found {len(matching_events)} events with correlation_id == {target_correlation_id}")
    
    if len(matching_events) == 0:
        print("❌ No events found with the specified correlation_id")
        return
    
    # Analyze each matching event
    for i, event in enumerate(matching_events):
        print(f"\n🔍 Event #{i+1}:")
        print(f"   Name: {event.get('name', 'UNKNOWN')}")
        print(f"   UID: {event.get('UID', 'NO_UID')}")
        print(f"   PID: {event.get('pid', 'UNKNOWN')}")
        print(f"   TID: {event.get('tid', 'UNKNOWN')}")
        print(f"   Phase: {event.get('ph', 'UNKNOWN')}")
        print(f"   Category (cat): {event.get('cat', 'MISSING')}")
        print(f"   Timestamp: {event.get('ts', 'MISSING')}")
        print(f"   Duration: {event.get('dur', 'MISSING')}")
        
        # Show classifier result
        classification = perf_analyzer.event_to_category(event)
        print(f"   🏷️  Classifier result: '{classification}'")
        
        # Show args
        args = event.get('args', {})
        print(f"   📋 Args:")
        for key, value in args.items():
            if key == 'correlation_id':
                print(f"      ✅ {key}: {value}")
            else:
                print(f"         {key}: {value}")
        
        # Find parent-child relationships
        print(f"   🌳 Tree placement:")
        
        # Check if this event has children
        children = event.get('children', [])
        if children:
            print(f"      Children ({len(children)}): {children[:5]}{'...' if len(children) > 5 else ''}")
            
            # Show first few children details
            for child_uid in children[:3]:
                if child_uid in perf_analyzer.tree.events_by_uid:
                    child = perf_analyzer.tree.events_by_uid[child_uid]
                    child_classification = perf_analyzer.event_to_category(child)
                    print(f"         Child {child_uid}: {child.get('name', 'UNKNOWN')} - {child_classification}")
        else:
            print(f"      No children")
        
        # Find if this event is a child of another event
        print(f"      Parents:")
        found_parent = False
        for other_event in perf_analyzer.tree.events:
            if event.get('UID') in other_event.get('children', []):
                parent_classification = perf_analyzer.event_to_category(other_event)
                print(f"         Parent {other_event.get('UID', 'NO_UID')}: {other_event.get('name', 'UNKNOWN')} - {parent_classification}")
                found_parent = True
                break
        
        if not found_parent:
            print(f"         No parent found (root-level event)")
        
        # Show correlation with other events
        print(f"   🔗 Correlation analysis:")
        correlated_events = []
        for other_event in perf_analyzer.tree.events:
            other_args = other_event.get('args', {})
            if (other_args.get('correlation_id') == target_correlation_id and 
                other_event.get('UID') != event.get('UID')):
                correlated_events.append(other_event)
        
        if correlated_events:
            print(f"      Found {len(correlated_events)} other events with same correlation_id:")
            for j, corr_event in enumerate(correlated_events[:3]):
                corr_classification = perf_analyzer.event_to_category(corr_event)
                print(f"         {j+1}. {corr_event.get('name', 'UNKNOWN')} - {corr_classification} (UID: {corr_event.get('UID', 'NO_UID')})")
        else:
            print(f"      No other events found with same correlation_id")

if __name__ == "__main__":
    analyze_correlation_5533()