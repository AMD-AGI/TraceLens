#!/usr/bin/env python3
"""
Trace ancestry for the specific correlation_id 5533 events found
"""

import sys
sys.path.insert(0, '/home/juhaj/projects/TraceLens')

def trace_specific_events():
    """Trace ancestry for the specific events we found"""
    print("🌳 Tracing ancestry for correlation_id == 5533 events")
    print("=" * 60)
    
    # The two events we found:
    target_events = [
        {
            'name': 'Cijk_Ailk_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_K1_LBSPPA2048_LBSPPB512_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA4_VWB4_WSGRA0_WSGRB0_WG32_8_1',
            'pid': 1, 'tid': 1, 'ts': 173140.952, 'dur': 58.046,
            'classification': 'kernel',
            'hlo_op': 'custom-call.1110.0',
            'tf_op': 'jit(train_step)/jit(main)/dot_general'
        },
        {
            'name': 'Cijk_Ailk_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs_MT128x128x64_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_K1_LBSPPA2048_LBSPPB512_LBSPPM0_LPA0_LPB16_LPM0_LRVW8_LWPMn1_MIAV0_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD4_NEPBS16_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VWA4_VWB4_WSGRA0_WSGRB0_WG32_8_1',
            'pid': 701, 'tid': 3031311314, 'ts': 171796.456, 'dur': 7.231,
            'classification': 'Unknown',
            'device_id': 0
        }
    ]
    
    print("📋 Target Events Summary:")
    for i, event in enumerate(target_events):
        print(f"   Event #{i+1}: {event['name'][:50]}...")
        print(f"      PID: {event['pid']}, TID: {event['tid']}")
        print(f"      Classification: '{event['classification']}'")
        print(f"      Timestamp: {event['ts']}")
        if 'hlo_op' in event:
            print(f"      HLO op: {event['hlo_op']}")
        if 'tf_op' in event:
            print(f"      TF op: {event['tf_op']}")
        print()
    
    # Build the tree to find these events and trace ancestry
    print("🌳 Building JAX tree to find events and trace ancestry...")
    from TraceLens.Trace2Tree.jax_trace_to_tree import JaxTreePerfAnalyzer
    
    xplane_path = "./2025_06_25_12_43_37/chi-mi300x-007.ord.vultr.cpe.ice.amd.com.xplane.pb"
    perf_analyzer = JaxTreePerfAnalyzer.from_xplane_pb(xplane_path)
    
    print(f"✅ Tree built with {len(perf_analyzer.tree.events):,} events")
    
    def find_ancestors(event_uid, events_by_uid, event_to_category):
        """Find all ancestors of an event up to the root"""
        ancestors = []
        current_uid = event_uid
        
        while True:
            # Find parent (event that has current_uid in its children)
            parent_uid = None
            for uid, event in events_by_uid.items():
                if current_uid in event.get('children', []):
                    parent_uid = uid
                    break
            
            if parent_uid is None:
                break  # Reached root
            
            parent_event = events_by_uid[parent_uid]
            classification = event_to_category(parent_event)
            
            ancestors.append({
                'uid': parent_uid,
                'event': parent_event,
                'classification': classification,
                'name': parent_event.get('name', 'UNKNOWN'),
                'level': len(ancestors) + 1
            })
            
            current_uid = parent_uid
        
        return ancestors
    
    # Find and analyze each target event in the tree
    for i, target_event in enumerate(target_events):
        print(f"\n{'='*80}")
        print(f"🎯 ANALYZING TARGET EVENT #{i+1}")
        print(f"{'='*80}")
        
        # Find this event in the tree
        tree_event = None
        for tree_ev in perf_analyzer.tree.events:
            if (tree_ev.get('name') == target_event['name'] and
                tree_ev.get('ts') == target_event['ts'] and
                tree_ev.get('pid') == target_event['pid'] and
                tree_ev.get('tid') == target_event['tid']):
                tree_event = tree_ev
                break
        
        if not tree_event:
            print(f"❌ Event not found in constructed tree")
            continue
        
        tree_uid = tree_event.get('UID')
        tree_classification = perf_analyzer.event_to_category(tree_event)
        
        print(f"✅ Found in tree:")
        print(f"   UID: {tree_uid}")
        print(f"   🏷️  Tree Classification: '{tree_classification}'")
        print(f"   Original Classification: '{target_event['classification']}'")
        
        # Show event details
        print(f"\n📋 Event Details:")
        print(f"   Name: {tree_event.get('name')}")
        print(f"   PID: {tree_event.get('pid')}, TID: {tree_event.get('tid')}")
        print(f"   Timestamp: {tree_event.get('ts')}")
        print(f"   Duration: {tree_event.get('dur')}")
        
        # Show args with correlation_id
        args = tree_event.get('args', {})
        print(f"   📋 Key Args:")
        for key in ['correlation_id', 'hlo_op', 'hlo_module', 'tf_op', 'name', 'device_id']:
            if key in args:
                print(f"      {key}: {args[key]}")
        
        # Check children
        children = tree_event.get('children', [])
        print(f"   👶 Children: {len(children)}")
        if children:
            print(f"      Child UIDs: {children[:5]}{'...' if len(children) > 5 else ''}")
        
        # Find and display complete ancestry
        ancestors = find_ancestors(tree_uid, perf_analyzer.tree.events_by_uid, perf_analyzer.event_to_category)
        
        print(f"\n🌲 COMPLETE ANCESTRY PATH (from event to root):")
        print(f"   Ancestry levels: {len(ancestors)}")
        
        if len(ancestors) == 0:
            print(f"   🌲 This event is a ROOT EVENT (no parents)")
        else:
            print(f"\n   📈 Path from target event to root:")
            
            # Show the target event first
            print(f"   ├─ Level 0 (TARGET): {tree_event.get('name')[:60]}...")
            print(f"      UID: {tree_uid}")
            print(f"      🏷️  '{tree_classification}'")
            print(f"      PID: {tree_event.get('pid')}, TID: {tree_event.get('tid')}")
            
            # Show each ancestor
            for j, ancestor in enumerate(ancestors):
                level_num = j + 1
                is_last = (j == len(ancestors) - 1)
                prefix = "   └─" if is_last else "   ├─"
                
                print(f"{prefix} Level {level_num}: {ancestor['name'][:60]}...")
                print(f"      UID: {ancestor['uid']}")
                print(f"      🏷️  '{ancestor['classification']}'")
                print(f"      PID: {ancestor['event'].get('pid')}, TID: {ancestor['event'].get('tid')}")
                print(f"      Timestamp: {ancestor['event'].get('ts')}")
                
                # Show children count and key args
                children_count = len(ancestor['event'].get('children', []))
                print(f"      Children: {children_count}")
                
                ancestor_args = ancestor['event'].get('args', {})
                key_args = {}
                for key in ['correlation_id', 'hlo_op', 'hlo_module', 'tf_op', 'Input Dims']:
                    if key in ancestor_args:
                        value = ancestor_args[key]
                        if isinstance(value, str) and len(value) > 60:
                            key_args[key] = value[:57] + "..."
                        else:
                            key_args[key] = value
                
                if key_args:
                    print(f"      Key Args: {key_args}")
                
                if not is_last:
                    print()
            
            # Show root analysis
            root = ancestors[-1]
            print(f"\n🌲 ROOT EVENT ANALYSIS:")
            print(f"   Root: {root['name']}")
            print(f"   Root UID: {root['uid']}")
            print(f"   Root Classification: '{root['classification']}'")
            
            # Verify it's truly a root
            root_has_parent = any(root['uid'] in event.get('children', []) 
                                for event in perf_analyzer.tree.events)
            
            if root_has_parent:
                print(f"   ⚠️  Warning: Root has a parent (unexpected)")
            else:
                print(f"   ✅ Confirmed: True root event (no parent)")

if __name__ == "__main__":
    trace_specific_events()