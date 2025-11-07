###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
import gzip
import os
from collections import defaultdict
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..util import DataLoader


def _default_filter_fn(event, include_pyfunc=False):
    """Default filter function - must be picklable for multiprocessing."""
    return (
        event.get("cat", None) != "Trace"
        and (include_pyfunc or event.get("cat") != "python_function")
    )


def _process_single_rank(rank, filepath, filter_fn, include_pyfunc, offset_multiplier, linking_key):
    """
    Standalone function to load and process trace data for a single rank.
    Must be at module level to be picklable for ProcessPoolExecutor.
    """
    data = DataLoader.load_data(filepath)
    
    processed_events = []
    for event in data["traceEvents"]:
        # If filter_fn is None, use default filter
        if filter_fn is None:
            if not _default_filter_fn(event, include_pyfunc):
                continue
        else:
            if not filter_fn(event):
                continue
        if "args" not in event:
            event["args"] = {}
        event["args"]["rank"] = rank
        
        # Adjust fields with offsets
        for field, offset_mult in offset_multiplier.items():
            is_arg = field == linking_key
            if is_arg and field in event["args"]:
                value = event["args"][field]
                event["args"][f"{field}_raw"] = value
                event["args"][field] += rank * offset_mult
            elif not is_arg and field in event:
                value = event[field]
                event["args"][f"{field}_raw"] = value
                if type(value) == int:
                    event[field] += rank * offset_mult
        
        processed_events.append(event)
    
    return rank, processed_events


class TraceFuse:
    def __init__(self, profile_filepaths_list_or_dict, use_multiprocessing=False, max_workers=None):
        """
        Initialize the TraceFuse class.

        :param profile_filepaths_or_dict:
            - If a list, assume it is already sorted by rank
              and each entry is a filepath for ranks [0..N-1]
            - If a dict, keys are rank and values are filepaths.
        :param use_multiprocessing: Whether to use multiprocessing for parallel trace loading (default: False).
        :param max_workers: Maximum number of worker processes (default: os.cpu_count()).
        """
        # we will map the list of filepaths to a dict
        if isinstance(profile_filepaths_list_or_dict, list):
            self.rank2filepath = {
                i: filepath for i, filepath in enumerate(profile_filepaths_list_or_dict)
            }
        elif isinstance(profile_filepaths_list_or_dict, dict):
            self.rank2filepath = profile_filepaths_list_or_dict

        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers if max_workers is not None else (os.cpu_count() or 8)

        # get the first file to set the linking key and offset multiplier
        filename = next(iter(self.rank2filepath.values()))
        data = DataLoader.load_data(filename)
        events = data["traceEvents"]
        self._set_linking_key(events)

        self.fields_to_adjust_offset = ["id", "pid", self.linking_key]
        self._set_offset_multiplier(events)

    def _set_linking_key(self, events):
        # load the first file to get the linking key
        launch_event = next(
            (
                event
                for event in events
                if event.get("cat") in ["cuda_runtime", "cuda_driver"]
                and "launch" in event.get("name", "").lower()
            ),
            None,
        )
        self.linking_key = (
            "correlation" if "correlation" in launch_event["args"] else "External id"
        )

    def _set_offset_multiplier(self, events):
        """Calculate offset multipliers for each field."""
        max_values = defaultdict(int)
        for event in events:
            for field in self.fields_to_adjust_offset:
                if field == self.linking_key:
                    value = event.get("args", {}).get(field)
                else:
                    value = event.get(field)
                if isinstance(value, int):
                    max_values[field] = max(max_values[field], value)
        self.offset_multiplier = {
            field: 10 ** (math.ceil(math.log10(max_value)) + 1)
            for field, max_value in max_values.items()
        }

    @staticmethod
    def default_filter_fn(event):
        return event.get("cat", None) != "Trace"

    def merge(self, filter_fn=None, include_pyfunc=False):
        """Merge trace files."""
        if self.use_multiprocessing:
            # Parallel processing using multiprocessing
            rank_results = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_rank = {
                    executor.submit(
                        _process_single_rank,
                        rank,
                        filepath,
                        filter_fn,
                        include_pyfunc,
                        self.offset_multiplier,
                        self.linking_key
                    ): rank
                    for rank, filepath in self.rank2filepath.items()
                }

                # Collect results as they complete
                for future in as_completed(future_to_rank):
                    rank, processed_events = future.result()
                    rank_results[rank] = processed_events

            # Merge results in rank order
            merged_data = []
            for rank in sorted(rank_results.keys()):
                merged_data.extend(rank_results[rank])
        else:
            # Sequential processing - reuse the same processing function
            merged_data = []
            for rank, filepath in self.rank2filepath.items():
                print(f"Processing file: {filepath}")
                _, processed_events = _process_single_rank(
                    rank, filepath, filter_fn, include_pyfunc, 
                    self.offset_multiplier, self.linking_key
                )
                merged_data.extend(processed_events)

        return merged_data

    def merge_and_save(
        self, output_file="merged_trace.json", filter_fn=None, include_pyfunc=False
    ):
        """Merge trace files and save the output."""
        merged_data = self.merge(filter_fn, include_pyfunc)

        json_data_out = {"traceEvents": merged_data}
        gz_output_file = output_file + ".gz"
        with gzip.open(gz_output_file, "wt", encoding="utf-8") as f:
            print(f"Writing to file: {gz_output_file}")
            json.dump(json_data_out, f, indent=4)
        print(f"Data successfully written to {gz_output_file}")
        return gz_output_file
