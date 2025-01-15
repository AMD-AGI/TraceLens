
class TraceToTree:
    def __init__(self, events_data):
        self.events = [{**data, 'UID': i} for i, data in enumerate(events_data)]
        self.events_by_uid = {event['UID']: event for event in self.events}
        self._compute_event_end_times()
        self._set_linking_key()
        self._preprocess_and_index_events()
        self.cpu_root_nodes = []

    def _compute_event_end_times(self) -> None:
        for event in self.events:
            if 'ts' in event and 'dur' in event:
                event['t_end'] = event['ts'] + event['dur']

    def _set_linking_key(self):
        launch_event = next(
            ( event for event in self.events if event.get('cat') in ['cuda_runtime', 'cuda_driver'] and 'launch' in event.get('name', '').lower() )
            , None)
        self.linking_key = 'correlation' if 'correlation' in launch_event['args'] else 'External id'

    def _preprocess_and_index_events(self) -> None:
        # 1. Create a dictionary to map the linking id to the start and end ac2g events
        # 2. Create a dictionary to map the pid, tid, and linking id to the actual event
        # 3. Create a dictionary to map the sequence number to the list of event uids
        # 4. Create a dictionary to map the python id to the event uid
        # This is done to quickly link events based on various keys

        self.ac2g_event_map = {'start': {}, 'end': {}}
        self.pid_tid_event_map = {}
        self.seq_num2event_uids_map = {} #from seq id to list uids
        self.dict_pythonID2UID = {}

        for event in self.events:
            # Process ac2g events
            if event.get('cat') == 'ac2g':
                if event['ph'] == 's':
                    self.ac2g_event_map['start'][event['id']] = event
                elif event['ph'] == 'f':
                    self.ac2g_event_map['end'][event['id']] = event
                continue

            # Process PID-TID-linking key events
            pid = event.get('pid')
            tid = event.get('tid')
            link_id = event.get('args', {}).get(self.linking_key)
            if not all([pid, tid, link_id]):
                self.pid_tid_event_map[(pid, tid, link_id)] = event
            
            # Process sequence number events
            seq_num = event.get('args', {}).get('Sequence number')
            if seq_num is not None:
                self.seq_num2event_uids_map.setdefault(seq_num, []).append(event['UID'])

            # Process python_function events
            if event.get('cat') == 'python_function':
                self.dict_pythonID2UID[event['args']['Python id']] = event['UID']

    def build_host_call_stack_tree(self, add_python_func=False) -> None:
    # 1. Filter and sort events based on their start timestamps.
    #    - Include only CPU, CUDA runtime, and optionally Python function events.
    # 2. Iterate through the sorted events and maintain a stack to track the current call hierarchy.
    #    - Pop events from the stack if they end before the current event starts to find the parent.
    #    - Set the parent of the current event as the top of the stack if the stack is not empty.
    #    - Push the current event onto the stack.
    #    - For CPU operations:
    #      - Mark as a root node if it is the first CPU operation in the stack.
    #      - Increment the count of CPU operations in the stack.

        print(f"Building CPU op tree with add_python_func={add_python_func}")
        list_events = []
        for event in self.events:
            is_cpu_or_cuda_event = event.get('cat') in {'cpu_op', 'cuda_runtime', 'cuda_driver'}
            is_cpu_or_cuda_event = event.get('cat') in {'cpu_op', 'cuda_runtime', 'cuda_driver'}
            is_python_event = event.get('cat') == 'python_function'
            if is_cpu_or_cuda_event or (add_python_func and is_python_event):
                list_events.append(event)

        events_sorted = sorted(list_events, key=lambda e: e['ts'])
        stack = []
        num_cpu_ops_in_stack = 0

        for event in events_sorted:
            event['tree'] = True

            while stack and event['ts'] >= stack[-1]['t_end']:
                popped_event = stack.pop()
                if popped_event.get('cat') == 'cpu_op':
                    num_cpu_ops_in_stack -= 1

            if stack:
                parent = stack[-1]
                parent.setdefault('children', []).append(event['UID'])
                event['parent'] = parent['UID']

            stack.append(event)
            if event.get('cat') == 'cpu_op':
                if num_cpu_ops_in_stack == 0:
                    event['cpu_op_root'] = True
                    self.cpu_root_nodes.append(event['UID'])
                num_cpu_ops_in_stack += 1

    def add_gpu_ops_to_tree(self):
        for event in self.events:
            if event.get('cat') not in {'cuda_runtime', 'cuda_driver'}:
                continue
            corresponding_gpu_event = self._find_corresponding_output_event(event)
            if not corresponding_gpu_event:
                continue
            event.setdefault('children', []).append(corresponding_gpu_event['UID'])
            corresponding_gpu_event['parent'] = event['UID']
            corresponding_gpu_event['tree'] = True

    def build_tree(self, add_python_func=False) -> None:
        print(f"Building tree with add_python_func={add_python_func}")
        self.build_host_call_stack_tree(add_python_func)
        self.add_gpu_ops_to_tree()

    def get_parent_event(self, event):
        if event.get('parent') is None:
            return None
        return self.events_by_uid[event['parent']]

    def get_children_events(self, event):
        if 'children' not in event:
            return []
        return [self.events_by_uid[child] for child in event['children']]

    def get_node_by_ext_id_pid_tid(self, ext_id, pid, tid):
        for event in self.events:
            if event.get('args', {}).get('External id') == ext_id and event.get('pid') == pid and event.get('tid') == tid:
                return event
        return None

    def traverse_subtree_and_print(self, node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        name = node.get('name', 'Unknown')
        max_len = 64
        if len(name) > max_len:
            name = name[:max_len] + '...'
        print(f"{prefix}{connector}UID: {node['UID']}, Category: {node.get('cat')}, Name: {name}, Duration: {node.get('dur')}")

        children = self.get_children_events(node)
        child_count = len(children)

        new_prefix = prefix + ("    " if is_last else "│   ")

        for i, child in enumerate(children):
            self.traverse_subtree_and_print(child, new_prefix, is_last=(i == child_count - 1))

    def traverse_parents_and_print(self, node):
        depth = 0
        while True:
            if depth == 0:
                print("Node:")
            else:
                print(f"{depth}-up:")

            # Print category and name
            print(f"  cat: {node['cat']}")
            name = node.get('name', 'Unknown')
            max_len = 64
            if len(name) > max_len:
                name = name[:max_len] + '...'
            print(f"  name: {name}")

            # Move to the parent node
            node = self.get_parent_event(node)
            if node is None:
                break
            depth += 1

    def get_seq_nums_for_node_subtree(self, node_UID):
        seq_nums = set()
        event = self.events_by_uid[node_UID]
        if event.get('args', {}).get('Sequence number') is not None:
            seq_nums.add(event['args']['Sequence number'])
        if 'children' in event:
            for child_UID in event['children']:
                seq_nums.update(self.get_seq_nums_for_node_subtree(child_UID))
        return seq_nums

    def link_bwd_events(self, event_UID):
        fwd_event = self.events_by_uid[event_UID]
        seq_nums = self.get_seq_nums_for_node_subtree(event_UID)
        bwd_event_UIDs = []
        for seq_num in seq_nums:
            for seq_num_match_UID in self.seq_num2event_uids_map.get(seq_num, []):
                if not self.events_by_uid[seq_num_match_UID].get('name').startswith('autograd::engine::evaluate_function:'):
                    continue
                bwd_event_UIDs.append(seq_num_match_UID)
                bwd_event = self.events_by_uid[seq_num_match_UID]
                bwd_event['fwd_event'] = event_UID
                break
        fwd_event['bwd_events'] = bwd_event_UIDs
    
    def _find_corresponding_output_event(self, input_event):
        # 1. Get the linking id from the input event
        # 2. Find the corresponding start and end ac2g events for the linking id
        # 3. Find the output event using the pid, tid, and linking id of the end ac2g event
        link_id = input_event.get('args', {}).get(self.linking_key)
        ac2g_start_event = self.ac2g_event_map['start'].get(link_id)
        ac2g_end_event = self.ac2g_event_map['end'].get(link_id)

        if not ac2g_start_event:
            return None

        if not ac2g_end_event:
            # print(f"Warning: start ac2g event found for {self.linking_key}={link_id} but no corresponding end ac2g event found.")
            # print(f"Input event name: {input_event['name']}")
            # print(('-'*64))
            return None

        pid = ac2g_end_event.get('pid')
        tid = ac2g_end_event.get('tid')
        link_id = ac2g_end_event.get('id')

        output_event = self.pid_tid_event_map.get((pid, tid, link_id))
        return output_event