# MIT License

# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import gzip
from collections import defaultdict
import math

from ..util import DataLoader

class TraceFuse:
    def __init__(self, profile_filepaths_list_or_dict):
        """
        Initialize the TraceFuse class.

        :param profile_filepaths_or_dict:
            - If a list, assume it is already sorted by rank
              and each entry is a filepath for ranks [0..N-1]
            - If a dict, keys are rank and values are filepaths.
        """
        # we will map the list of filepaths to a dict
        if isinstance(profile_filepaths_list_or_dict, list):
            self.rank2filepath = {i: filepath for i, filepath in enumerate(profile_filepaths_list_or_dict)}
        elif isinstance(profile_filepaths_list_or_dict, dict):
            self.rank2filepath = profile_filepaths_list_or_dict

        # get the first file to set the linking key and offset multiplier
        filename = next(iter(self.rank2filepath.values()))
        data = DataLoader.load_data(filename)
        events = data['traceEvents']
        self._set_linking_key(events)

        self.fields_to_adjust_offset = ['id', 'pid', self.linking_key]
        self._set_offset_multiplier(events)

    def _set_linking_key(self, events):
        # load the first file to get the linking key
        launch_event = next(
            ( event for event in events if event.get('cat') in ['cuda_runtime', 'cuda_driver'] and 'launch' in event.get('name', '').lower() )
            , None)
        self.linking_key = 'correlation' if 'correlation' in launch_event['args'] else 'External id'

    def _set_offset_multiplier(self, events):
        """Calculate offset multipliers for each field."""
        max_values = defaultdict(int)
        for event in events:
            for field in self.fields_to_adjust_offset:
                if field == self.linking_key:
                    value = event.get('args', {}).get(field)
                else:
                    value = event.get(field)
                if isinstance(value, int):
                    max_values[field] = max(max_values[field], value)
        self.offset_multiplier = {field: 10 ** (math.ceil(math.log10(max_value)) + 1)
                                    for field, max_value in max_values.items()}

    @staticmethod
    def default_filter_fn(event):
        return event.get('cat', None) != 'Trace'

    def adjust_field(self, event, field, rank, offset_multiplier):
        is_arg = field == self.linking_key
        if is_arg and field in event['args']:
            value = event['args'][field]
            event['args'][f'{field}_raw'] = value
            event['args'][field] += rank * offset_multiplier
        elif not is_arg and field in event:
            value = event[field]
            event['args'][f'{field}_raw'] = value
            if type(value) == int:
                event[field] += rank * offset_multiplier

    def merge(self, filter_fn=None, include_pyfunc=False):
        """Merge trace files."""
        merged_data = []

        if filter_fn is None:
            filter_fn = lambda event: (
                event.get('cat', None) != 'Trace'
                and (include_pyfunc or event.get('cat') != 'python_function')
            )

        for rank, filepath in self.rank2filepath.items():
            print(f"Processing file: {filepath}")
            data = DataLoader.load_data(filepath)

            processed_events = []
            for event in data['traceEvents']:
                if not filter_fn(event):
                    continue
                if 'args' not in event:
                    event['args'] = {}
                event['args']['rank'] = rank

                for field, offset_multiplier in self.offset_multiplier.items():
                    self.adjust_field(event, field, rank, offset_multiplier)
                processed_events.append(event)
            merged_data.extend(processed_events)

        return merged_data

    def merge_and_save(self, output_file='merged_trace.json',
                        filter_fn=None, include_pyfunc=False):
        """Merge trace files and save the output."""
        merged_data = self.merge(filter_fn, include_pyfunc)

        json_data_out = {'traceEvents': merged_data}
        gz_output_file = output_file + '.gz'
        with gzip.open(gz_output_file, 'wt', encoding='utf-8') as f:
            print(f"Writing to file: {gz_output_file}")
            json.dump(json_data_out, f, indent=4)
        print(f"Data successfully written to {gz_output_file}")
        return gz_output_file