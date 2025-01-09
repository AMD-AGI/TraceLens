import json
import os
import gzip
import shutil
from collections import defaultdict
import math

class TraceFuse:
    def __init__(self, profiles_dir, world_size):
        """
        Initialize the TraceFuse class.
        
        :param profiles_dir: Root directory containing the profiles.
        :param world_size: Total number of ranks in the distributed setup.
        :param ranks_to_merge: Optional list of ranks to merge. Defaults to all ranks in the world_size.
        """
        self.profiles_dir = profiles_dir
        self.world_size = world_size
        # self.fields_to_adjust_offset = ['External id', 'id']
        self.fields_to_adjust_offset = ['id', 'correlation', 'pid']
        self.filter_event_fn = self.default_filter_event_fn
        self.fn_rank2file = self.default_fn_rank2file

    def default_filter_event_fn(self, event):
        """Default filter function to keep all events."""
        return True

    def default_fn_rank2file(self, rank):
        """Default function to map rank to file path."""
        return os.path.join(self.profiles_dir, str(rank), 'pytorch_profile.json')

    def set_filter(self, filter_fn):
        """Set a custom filter function."""
        self.filter_event_fn = filter_fn

    def set_rank2file_fn(self, fn_rank2file):
        """Set a custom rank-to-file mapping function."""
        self.fn_rank2file = fn_rank2file

    # def adjust_pid(self, event, rank):
    def adjust_pid(self, event, rank, offset_multiplier):
        # pid = str(event['pid'])
        # event['pid'] = f"rank{rank}_pid{pid}"
        # event['args']['pid_raw'] = pid
        try:
            event['args']['pid_raw'] = event['pid']
            event['pid'] += rank * offset_multiplier
        except TypeError:
            pass

    def adjust_id(self, event, rank, offset_multiplier):
        if event.get('cat', None) != 'ac2g':
            return
        try:
            event['args']['id_raw'] = event['id']
            event['id'] += rank * offset_multiplier
        except KeyError:
            pass
    
    def adjust_external_id(self, event, rank, offset_multiplier):
        try:
            event['args']['External id_raw'] = event['args']['External id']
            event['args']['External id'] += rank * offset_multiplier
        except KeyError:
            pass
    
    def adjust_correlation(self, event, rank, offset_multiplier):
        try:
            event['args']['correlation_raw'] = event['args']['correlation']
            event['args']['correlation'] += rank * offset_multiplier
        except KeyError:
            pass


    def get_offset_multiplier(self, data):
        """Calculate offset multipliers for each field."""
        max_values = defaultdict(int)
        for event in data['traceEvents']:
            if not self.filter_event_fn(event):
                continue
            if event.get('cat', None) == 'Trace':
                continue
            for field in self.fields_to_adjust_offset:
                # is_arg = field == 'External id'
                is_arg = field == 'correlation' or field == 'External id'
                try:
                    value = int(event['args'][field] if is_arg else event[field])
                    max_values[field] = max(max_values[field], value)
                except (KeyError, ValueError):
                    pass

        return {field: 10 ** (math.ceil(math.log10(max_value)) + 1)
                for field, max_value in max_values.items()}

    def process_single_file(self, filepath, rank):
        """Process a single trace file."""
        print(f"Processing file: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        dict_offset = self.get_offset_multiplier(data)
        print(f"Offset multipliers: {dict_offset}")
        processed_events = []

        for event in data['traceEvents']:
            if not self.filter_event_fn(event):
                continue
            if 'args' not in event:
                event['args'] = {}
            event['args']['rank'] = rank
            self.adjust_pid(event, rank, dict_offset.get('pid'))
            # self.adjust_id(event, rank, dict_offset.get('id'))
            self.adjust_id(event, rank, 100*dict_offset.get('pid')) # for now use same offset multiplier as pid
            # self.adjust_external_id(event, rank, dict_offset.get('External id'))
            self.adjust_correlation(event, rank, dict_offset.get('correlation'))
            processed_events.append(event)
        
        return processed_events

    def merge(self, ranks_to_merge=None):
        """Merge trace files."""
        self.merged_data = []

        for rank in ranks_to_merge:
            filepath = self.fn_rank2file(rank)
            if not os.path.exists(filepath):
                print(f"Warning: file {filepath} does not exist")
                continue
            self.merged_data.extend(self.process_single_file(filepath, rank))
        
    def merge_and_save(self, ranks_to_merge=None, output_file='merged_trace.json'):
        """Merge trace files and save the output."""
        self.merge(ranks_to_merge)

        json_data_out = {'traceEvents': self.merged_data}
        with open(output_file, 'w') as jsonfileout:
            print(f"Writing data to {output_file}...")
            json.dump(json_data_out, jsonfileout, indent=4)
        print(f"Data successfully written to {output_file}.")

        with open(output_file, 'rb') as f_in:
            with gzip.open(output_file + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Data successfully written to {output_file}.gz.")


