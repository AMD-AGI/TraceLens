import itertools
import json

try:
    from enum import StrEnum
except ImportError:
    try:
        from backports.strenum import StrEnum
    # fallback for Python 3.10
    except ImportError:
        from strenum import StrEnum
from typing import List, Dict, Callable, Tuple

# generic data loader class for json, json.gz, or tensorboard pb files
# tensorboard pb files are useful for Jax in particular because the json.gz traces produced by jax can have incorrect timestamps and missing information
class DataLoader:
    @staticmethod
    def load_data(filename_path:str, save_preprocessed: bool = False) -> dict:
        if filename_path.endswith('pb'):
            from tensorboard_plugin_profile.convert import raw_to_tool_data as convert
            data, _ = convert.xspace_to_tool_data([filename_path], "trace_viewer@^", {})
            data = data.decode("utf-8") # we get bytes back from the call above
        elif filename_path.endswith('json.gz'):
            import gzip
            with gzip.open(filename_path, 'r') as fin:
                data = fin.read().decode('utf-8')
        elif filename_path.endswith('json'):
            with open(filename_path, 'r') as fin:
                data = fin.read()
        else:
            raise ValueError("Unknown file type",filename_path)
        if (save_preprocessed):
            with open(filename_path.replace("pb", "processed.json"), 'w') as writefile:
                writefile.write(data)
        return json.loads(data)

# Trace event utilities to help with traces in the Google Trace Event format
# https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0
# This trace event format includes both Pytorch and Jax traces (and anything that can be viewed in Perfetto)
class TraceEventUtils:
    class TraceKeys(StrEnum):
        PID       = 'pid'
        TID       = 'tid'
        Phase     = 'ph'
        Args      = 'args'
        Name      = 'name'
        TimeStamp = 'ts'
        Duration  = 'dur'
        Category  = 'cat'
        TimeEnd   = 't_end'
        UID       = 'UID'

    class TracePhases(StrEnum):
        DurationBegin = 'B'
        DurationEnd   = 'E'
        Complete      = 'X'
        Counter       = 'C'
        Sample        = 'P'
        Metadata      = 'M'

    class MetadataFields(StrEnum):
        ProcessName   = 'process_name'
        ProcessLabels = 'process_labels'
        ProcessSort   = 'process_sort_index'
        ThreadName    = 'thread_name'
        ThreadSort    = 'thread_sort_index'

    class ArgNames(StrEnum):
        Name        = 'name'
        SortIndex   = 'sort_index'
        StreamIndex = 'stream_index'
        Labels      = 'labels'

    class GpuEventCategories(StrEnum):
        Kernel = 'kernel'
        MemSet = 'gpu_memset'
        MemCpy = 'gpu_memcpy'

    class CpuEventCategories(StrEnum):
        Kernel  = 'cpu_op'
        Runtime = 'cuda_runtime'
        Driver  = 'cuda_driver'


    @staticmethod
    def split_by_field(events: List[dict], field: str, defaultKey: str = None) -> Dict[str, List]:
        return dict(itertools.groupby(events, lambda event: event.get(field, defaultKey)))

    # Merges metadata events into a dictionary hierarchy per process
    # Process
    # None: {process_name, process_sort_index}
    # Thread_id: {thread_name, thread_sort_index} for each Thread_id
    @staticmethod
    def get_metadata(events: List[dict]) -> Dict[str, Dict[str, str]]:
        def get_metadata_val(x: dict) -> str:
            arg_labels = {
                TraceEventUtils.MetadataFields.ProcessName: TraceEventUtils.ArgNames.Name,
                TraceEventUtils.MetadataFields.ProcessLabels: TraceEventUtils.ArgNames.Labels,
                TraceEventUtils.MetadataFields.ProcessSort: TraceEventUtils.ArgNames.SortIndex,
                TraceEventUtils.MetadataFields.ThreadName: TraceEventUtils.ArgNames.Name,
                TraceEventUtils.MetadataFields.ThreadSort: TraceEventUtils.ArgNames.SortIndex,
            }
            key = x[TraceEventUtils.TraceKeys.Name]
            return (key, x[TraceEventUtils.TraceKeys.Args][arg_labels[key]])
        metadata_fields = itertools.takewhile(lambda x: x[TraceEventUtils.TraceKeys.Phase] == TraceEventUtils.TracePhases.Metadata, events)
        by_process = itertools.groupby(metadata_fields, lambda event: event[TraceEventUtils.TraceKeys.PID])
        # TID is not required for process-specific tags, so use null thread id for them
        fully_processed = map(lambda kv: (kv[0], itertools.groupby(kv[1], lambda event: event.get(TraceEventUtils.TraceKeys.TID))), by_process)
        return dict(map(lambda kv: (kv[0], dict(map(lambda kv1: (kv1[0], dict(map(lambda event: (get_metadata_val(event)), kv1[1]))), kv[1]))), fully_processed))

    @staticmethod
    def non_metadata_events(events:List[dict]) -> List[dict]:
        return list(itertools.dropwhile(lambda e: e[TraceEventUtils.TraceKeys.Phase] == TraceEventUtils.TracePhases.Metadata, events))

    @staticmethod
    def default_categorizer(event: dict) -> str:
        return event.get(TraceEventUtils.TraceKeys.Category)

    @staticmethod
    def split_events_by_pid_tid(events: List[dict]) -> Dict[str, Dict[str, List[Dict]]]:
        event_dict={}
        for event in TraceEventUtils.non_metadata_events(events):
            pid=event.get(TraceEventUtils.TraceKeys.PID)
            tid=event.get(TraceEventUtils.TraceKeys.TID)
            if pid in event_dict:
                pid_events = event_dict[pid]
            else:
                pid_events = event_dict[pid] = {}
            if tid in pid_events:
                pid_events[tid].append(event)
            else:
                pid_events[tid] = [event]
        return event_dict

    @staticmethod
    def sort_events_by_timestamp_duration(events: List[dict]) -> None:
        events.sort(key = lambda x: (x.get(TraceEventUtils.TraceKeys.TimeStamp), x.get(TraceEventUtils.TraceKeys.Duration)))

    @staticmethod
    def find_thread_by_item_in_metadata(metadata: dict[int, dict], select_item: Callable[[int], bool]) -> int:
        return next(filter(select_item, metadata.items()))[0]

    @staticmethod
    def compute_event_end_times(events: List[dict]) -> None:
        for event in events:
            TraceEventUtils.compute_single_event_end_time(event)

    @staticmethod
    def compute_single_event_end_time(event: dict) -> None:
        if TraceEventUtils.TraceKeys.TimeStamp in event and TraceEventUtils.TraceKeys.Duration in event and TraceEventUtils.TraceKeys.TimeEnd not in event:
            event[TraceEventUtils.TraceKeys.TimeEnd] = event[TraceEventUtils.TraceKeys.TimeStamp] + event[TraceEventUtils.TraceKeys.Duration]














