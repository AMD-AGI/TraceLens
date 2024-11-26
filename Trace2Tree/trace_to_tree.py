
class TraceToTree:
    def __init__(self, events_data):
        self.events = [{**data, 'UID': i} for i, data in enumerate(events_data)]
        self.events_by_uid = {event['UID']: event for event in self.events}
        self._add_t_end()
        self._preprocess_and_index_events()
        self.cpu_root_nodes = []

    def _add_t_end(self) -> None:
        for event in self.events:
            if 'ts' in event and 'dur' in event:
                event['t_end'] = event['ts'] + event['dur']

    def build_cpu_op_tree(self) -> None:
        # 1. Sort the events by timestamps and initialize an empty stack
        # 2. Iterate through the sorted events
        # 2.1. Pop the stack until the current event starts after the top of the stack ends
        # This is to find the parent of the current event
        # 2.2 If the stack is not empty, the parent of the current event is the top of the stack
        # 2.3 If the stack is empty, the current event is a root event
        # 2.4 Push the current event to the stack

        events_sorted = sorted([event for event in self.events if event.get('cat') in {'cpu_op', 'cuda_runtime'}], key=lambda e: e['ts'])
        stack = []

        for event in events_sorted:
            event['tree'] = True

            while stack and event['ts'] >= stack[-1]['t_end']:
                stack.pop()

            if stack:
                parent = stack[-1]
                if parent['UID'] == 148871:
                    print(event)
                parent.setdefault('children', []).append(event['UID'])
                event['parent'] = parent['UID']
            else:
                event['parent'] = None
                event['cpu_op_root'] = True
                self.cpu_root_nodes.append(event['UID'])

            stack.append(event)

    def _preprocess_and_index_events(self) -> None:
        # 1. Create a dictionary to map the correlation id to the start and end ac2g events
        # 2. Create a dictionary to map the pid, tid, and correlation id to the actual event
        # This is done to quickly find the corresponding output flow event for a given input flow event
        self.ac2g_event_map = {'start': {}, 'end': {}}
        self.pid_tid_event_map = {}

        for event in self.events:
            if event.get('cat') == 'ac2g':
                if event['ph'] == 's':
                    self.ac2g_event_map['start'][event['id']] = event
                elif event['ph'] == 'f':
                    self.ac2g_event_map['end'][event['id']] = event
            else:
                pid = event.get('pid')
                tid = event.get('tid')
                corr_id = event.get('args', {}).get('correlation')
                if pid is not None and tid is not None and corr_id is not None:
                    pid_tid_key = (pid, tid, corr_id)
                    self.pid_tid_event_map[pid_tid_key] = event

    def _find_corresponding_output_event(self, input_event):
        # 1. Get the correlation id from the input event
        # 2. Find the corresponding start and end ac2g events for the correlation id
        # 3. Find the output event using the pid, tid, and correlation id of the end ac2g event
        corr_id = input_event.get('args', {}).get('correlation')
        ac2g_start_event = self.ac2g_event_map['start'].get(corr_id)
        ac2g_end_event = self.ac2g_event_map['end'].get(corr_id)

        if not ac2g_start_event:
            return None

        if not ac2g_end_event:
            print(f"Warning: start event found for correlation id {corr_id} but no corresponding end event found.")
            return None

        pid = ac2g_end_event.get('pid')
        tid = ac2g_end_event.get('tid')
        corr_id = ac2g_end_event.get('id')

        output_event = self.pid_tid_event_map.get((pid, tid, corr_id))
        return output_event

    def add_gpu_ops_to_tree(self):
        for event in self.events:
            if event.get('cat') == 'cuda_runtime':
                corresponding_gpu_event = self._find_corresponding_output_event(event)
                if corresponding_gpu_event:
                    event.setdefault('children', []).append(corresponding_gpu_event['UID'])
                    corresponding_gpu_event['parent'] = event['UID']
                    corresponding_gpu_event['tree'] = True

    def build_tree(self) -> None:
        self.build_cpu_op_tree()
        self.add_gpu_ops_to_tree()

    def get_parent_event(self, event):
        if event['parent'] is None:
            return None
        return self.events_by_uid[event['parent']]

    def get_children_events(self, event):
        if 'children' not in event:
            return []
        return [self.events_by_uid[child] for child in event['children']]

    def get_node_by_ext_id_pid_tid(self, ext_id, pid, tid):
        for event in self.events:
            # if event['args'].get('External id') == ext_id and event['pid'] == pid and event['tid'] == tid:
            if event.get('args', {}).get('External id') == ext_id and event.get('pid') == pid and event.get('tid') == tid:
                return event
        return None
    
    def traverse_subtree_and_print(self, node, prefix="", is_last=True):
        # Determine the current node's tree prefix
        connector = "└── " if is_last else "├── "
        name = node.get('name', 'Unknown')
        max_len = 64
        if len(name) > max_len:
            name = name[:max_len] + '...'
        print(f"{prefix}{connector}Category: {node.get('cat')}, Name: {name}")

        # Get children of the current node
        children = self.get_children_events(node)
        child_count = len(children)

        # Update prefix for child nodes
        new_prefix = prefix + ("    " if is_last else "│   ")

        # Recursively traverse the child nodes
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

    def get_gpu_time_for_op(self, UID):
        total_dur = 0
        event = self.events_by_uid[UID]
        if event.get('cat') == 'kernel':
            total_dur += event['dur']
            return total_dur
        for child_UID in event.get('children', []):
            total_dur += self.get_gpu_time_for_op(child_UID)
        return total_dur