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

import pandas as pd
import itertools
import math
import string
import tqdm

class GPUEventAnalyser:
    def __init__(self, events):
        """
        Initialize with a list of event dictionaries.
        """
        self.events = events


    @staticmethod
    def merge_intervals(intervals):
        """
        Merge a list of intervals (each as a (start, end) tuple) into a union of non-overlapping intervals.
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged


    @staticmethod
    def subtract_intervalsA_from_B(intervals_to_subtract, intervals):
        """Subtract set of intervals from another set of intervals.

        Intervals in sets are expected to be non-overlapping and sorted.

        Returns a list of intervals as (start, end) tuples.
        """
        result = []
        a_idx = 0
        a_len = len(intervals_to_subtract)

        for b_start, b_end in intervals:
            current = b_start
            while a_idx < a_len and intervals_to_subtract[a_idx][1] <= b_start:
                a_idx += 1

            temp_idx = a_idx
            while temp_idx < a_len and intervals_to_subtract[temp_idx][0] < b_end:
                a_start, a_end = intervals_to_subtract[temp_idx]
                if a_start > current:
                    result.append((current, min(b_end, a_start)))
                current = max(current, a_end)
                if current >= b_end:
                    break
                temp_idx += 1
            if current < b_end:
                result.append((current, b_end))

        return result


    all_gpu_key = 'all_gpu'
    computation_key = 'computation'
    communication_key = 'communication'
    memcpy_key = 'memcpy'
    all_cpu_key = 'all_cpu'
    gpu_event_keys = [all_gpu_key, computation_key, communication_key, memcpy_key]
    cpu_event_keys = [all_cpu_key]
    @property
    @staticmethod
    def all_event_keys():
        return itertools.chain(GPUEventAnalyser.gpu_event_keys, GPUEventAnalyser.cpu_event_keys)

    def get_gpu_event_lists(self):
        """
        Return a dictionary of lists of events, categorized by event types
        Event types are all gpu events, computation, communication, and memcpy.
        Be sure that the returned events have 'ts' and 't_end' fields.
        The default implementation is for PyTorch json trace format.
        Inherit the class and reimplement this method for your profile format.
        """

        # note all events are not gpu events
        # the events list contains gpu events as well as host side events

        gpu_events = []
        comp_events = []
        comm_events = []
        memcpy_events = []

        for event in self.events:

            #TODO: ideally we want to get gpu events based on process id
            # That will be done shortly
            category = event.get('cat')
            if category in {'kernel', 'gpu_memcpy', 'gpu_memset'}:
                if 't_end' not in event:
                    event['t_end'] = event['ts'] + event['dur']
                gpu_events.append(event)

                if category == 'gpu_memcpy':
                    memcpy_events.append(event)
                elif category in {'kernel', 'gpu_memset'}:
                    if 'nccl' in event.get('name'):
                        comm_events.append(event)
                    else:
                        comp_events.append(event)
                else:
                    raise ValueError(f"Unknown event category: {category}")
        return {
            GPUEventAnalyser.all_gpu_key: gpu_events,
            GPUEventAnalyser.computation_key: comp_events,
            GPUEventAnalyser.communication_key: comm_events,
            GPUEventAnalyser.memcpy_key: memcpy_events,
        }


    @staticmethod
    def verify_dict_gpu_event_lists(dict_gpu_event_lists):
        # first check if the keys are correct
        # note the check before is a linear lookup, but there are only 4 elements in the list
        if not all (key in GPUEventAnalyser.gpu_event_keys for key in dict_gpu_event_lists):
            raise ValueError(f"Expected keys: {GPUEventAnalyser.gpu_event_keys}, " +
                             f"got: {dict_gpu_event_lists.keys()}")
        # next check if the events have 'ts' and 't_end' fields
        for _, events in dict_gpu_event_lists.items():
            for event in events:
                if 'ts' not in event or 't_end' not in event:
                    raise ValueError(f"Event {event} does not have 'ts' or 't_end' fields")
        if len(dict_gpu_event_lists['all_gpu']) == 0:
            raise ValueError("No GPU events found in the trace")

    @staticmethod
    def compute_metrics_dict(dict: dict):
        dict_intervals = {}
        for key, events in dict.items():
            dict_intervals[key] = [(event['ts'], event['t_end']) for event in events]

        # Merge intervals within each category.
        comp_union = GPUEventAnalyser.merge_intervals(dict_intervals['computation'])
        comm_union = GPUEventAnalyser.merge_intervals(dict_intervals['communication'])
        memcpy_union = GPUEventAnalyser.merge_intervals(dict_intervals['memcpy'])
        all_intervals = GPUEventAnalyser.merge_intervals(dict_intervals['all_gpu'])

        # end of the last event - start of the first event
        total_time = all_intervals[-1][1] - all_intervals[0][0]


        comp_time = sum(end - start for start, end in comp_union)

        total_comm_time = sum(end - start for start, end in comm_union)
        exposed_comm_intervals = GPUEventAnalyser.subtract_intervalsA_from_B(comp_union, comm_union)
        exposed_comm_time = sum(end - start for start, end in exposed_comm_intervals)

        total_memcpy_time = sum(end - start for start, end in memcpy_union)
        memcpy_minus_compute = GPUEventAnalyser.subtract_intervalsA_from_B(comp_union, memcpy_union)
        exposed_memcpy_intervals = GPUEventAnalyser.subtract_intervalsA_from_B(comm_union, memcpy_minus_compute)
        exposed_memcpy_time = sum(end - start for start, end in exposed_memcpy_intervals)

        busy_time = sum(end - start for start, end in all_intervals)
        idle_time = total_time - busy_time

        # assert that compute + exposed comm + exposed memcpy + idle = total time
        assert abs(comp_time + exposed_comm_time + exposed_memcpy_time + idle_time - total_time) < 1e-6

        return {
            "computation_time": comp_time,
            "exposed_comm_time": exposed_comm_time,
            "exposed_memcpy_time": exposed_memcpy_time,
            "busy_time": busy_time,
            "idle_time": idle_time,
            "total_time": total_time,
            "total_comm_time": total_comm_time,
            "total_memcpy_time": total_memcpy_time,
        }


    def compute_metrics(self):
        """
        Compute various metrics from the GPU event data.
        Computation is defined as the time spent in computation kernels.
        Communication is defined as the time spent in communication kernels.
        Memcpy is defined as the time spent in memcpy kernels.
        Exposed communication time is the time spent in communication kernels that is not overlapped by computation.
        Exposed memcpy time is the time spent in memcpy kernels that is not overlapped by computation or communication.
        """

        # Categorize events.
        dict_gpu_event_lists = self.get_gpu_event_lists()
        GPUEventAnalyser.verify_dict_gpu_event_lists(dict_gpu_event_lists)

        return GPUEventAnalyser.compute_metrics_dict(dict_gpu_event_lists)

    @staticmethod
    def get_breakdown_df_from_dict(dict_metrics: dict):
        df = pd.DataFrame(dict_metrics.items(), columns=['type', 'time'])
        # convert time to ms by div by 1e3
        df['time ms'] = df['time'] / 1e3
        df['percent'] = df['time'] / df.loc[df['type'] == 'total_time', 'time'].values[0] * 100
        df = df.drop(columns=['time'])

        return df

    def get_breakdown_df(self):
        dict_metrics = self.compute_metrics()
        return GPUEventAnalyser.get_breakdown_df_from_dict(dict_metrics)

# Pytorch GPU event analyser inherits everything from the base class
class PytorchGPUEventAnalyser(GPUEventAnalyser):
    pass

# Jax GPU event analyser supports multiple GPUs
class JaxGPUEventAnalyser(GPUEventAnalyser):
    # keywords for splitting jax events
    GemmKeys = ["Cijk", "gemm", "nvjet", "cublasLt"]
    FABwdKeys = ["FmhaBwd"]
    FAFwdKeys = ["FmhaFwd"]
    FAV3Keys = ["kernel_func"] # find a more precise way to do this
    ConvKeys = ["FillBuffer"]
    TEKeys = ["transformer_engine"]
    ClassCategories = {
        "GEMM": GemmKeys,
        "FA BWD": FABwdKeys,
        "FA FWD": FAFwdKeys,
        "FA V3": FAV3Keys,
        "ConvKeys": ConvKeys,
        "TEKeys": TEKeys,
    }
    UncategorizedEventKey = "Uncategorized Events"

    def get_gpu_event_lists(self, gpu_pid = None, event_filter = None):
        """
        Return a dictionory of GPU to dictionaries of lists of events,
        categorized by event types
        Event types are all gpu events, computation, communication, and memcpy.
        Be sure that the returned events have 'ts' and 't_end' fields.
        The default implementation is for PyTorch json trace format.
        Inherit the class and reimplement this method for your profile format.

        If pid is passed in, returns just a dictionary of that pid's events
        """

        # note all events are not gpu events
        # the events list contains gpu events as well as host side events
        return_dict = {}
        for event in self.events:
            if event_filter is not None and not event_filter(event):
                continue
            pid = event.get('pid')
            # jax uses pid > 100 for CPU evens
            # skip some dictionary setup events that do not have ts
            if 'ts' in event:
                if pid < 100:
                    cur_dict = return_dict.get(pid)
                    if cur_dict is None:
                        cur_dict = {key: [] for key in GPUEventAnalyser.gpu_event_keys}
                        return_dict[pid] = cur_dict
                    if 't_end' not in event:
                        event['t_end'] = event['ts'] + event['dur']
                    cur_dict[GPUEventAnalyser.all_gpu_key].append(event)
                    name = event.get('name')
                    if (any(name.lower().startswith(x) for x in ['copy', 'memcpy', 'memset'])):
                        cur_dict[GPUEventAnalyser.memcpy_key].append(event)
                    elif name.startswith('nccl'):
                        cur_dict[GPUEventAnalyser.communication_key].append(event)
                    else:
                        cur_dict[GPUEventAnalyser.computation_key].append(event)
                else:
                    cur_dict = return_dict.get(pid)
                    if cur_dict is None:
                        cur_dict = {key: [] for key in GPUEventAnalyser.cpu_event_keys}
                        return_dict[pid] = cur_dict
                    cur_dict[GPUEventAnalyser.all_cpu_key].append(event)
        if gpu_pid is None:
            return return_dict
        else:
            return return_dict.get(gpu_pid, {})

    def compute_metrics(self):
        """
        Compute various metrics from the GPU event data.
        Computation is defined as the time spent in computation kernels.
        Communication is defined as the time spent in communication kernels.
        Memcpy is defined as the time spent in memcpy kernels.
        Exposed communication time is the time spent in communication kernels that is not overlapped by computation.
        Exposed memcpy time is the time spent in memcpy kernels that is not overlapped by computation or communication.
        """

        # Categorize events.
        # get GPU 0 (PID 1) for Jax
        dict_gpu_event_lists = self.get_gpu_event_lists(1)
        GPUEventAnalyser.verify_dict_gpu_event_lists(dict_gpu_event_lists)

        return GPUEventAnalyser.compute_metrics_dict(dict_gpu_event_lists)

    def get_breakdown_df_multigpu(self, event_filter = None):
        events = self.get_gpu_event_lists(event_filter = event_filter)
        gpu_frames = {}
        print("Processing events by GPU")
        for gpu_id, cur_events in tqdm.tqdm(events.items()):
            self.verify_dict_gpu_event_lists(cur_events)
            cur_metrics = GPUEventAnalyser.compute_metrics_dict(cur_events)
            gpu_frames[gpu_id - 1] = GPUEventAnalyser.get_breakdown_df_from_dict(cur_metrics)
        return gpu_frames

    @staticmethod
    def breakdown_compute_events(event_list, group_by_gpu: bool = True, group_by_name = False):
        def add_event(cur_event_list, name, duration):
            current = cur_event_list.get(name, [0, 0])
            current[0] += 1
            current[1] += duration
            if current[0] == 1:
                cur_event_list[name] = current

        categorized_events = {}
        uncategorized_events = {}
        for compute_event in filter(lambda k: k.get('tid', 200) <= 100, event_list):
            if group_by_gpu:
                gpu = int(compute_event['pid'])
                if gpu in categorized_events:
                    cur_categorized_list = categorized_events[gpu]
                    cur_uncategorized_list = uncategorized_events[gpu]
                else:
                    cur_categorized_list = {}
                    categorized_events[gpu] = cur_categorized_list
                    cur_uncategorized_list = {}
                    uncategorized_events[gpu] = cur_uncategorized_list
            else:
                cur_categorized_list = categorized_events
                cur_uncategorized_list = uncategorized_events

            name=compute_event["name"]
            duration=compute_event["dur"]
            found = False
            for category, filters in JaxGPUEventAnalyser.ClassCategories.items():
                if any(f in name for f in filters):
                    add_event(cur_categorized_list, category, duration)
                    found = True
                    break
            if not found:
                if group_by_name:
                    name = name.rstrip(string.digits)
                add_event(cur_categorized_list, JaxGPUEventAnalyser.UncategorizedEventKey, duration)
                add_event(cur_uncategorized_list, name, duration)

        return categorized_events, uncategorized_events

    @staticmethod
    def create_breakdown_df(events: dict, total_time):
        df = pd.DataFrame.from_dict(events, orient='index', columns=['count', 'time'])
        # convert time to ms by div by 1e3
        df['time ms'] = df['time'] / 1e3
        df['percent'] = df['time'] / total_time * 100
        df = df.drop(columns=['time'])
        df = df.sort_values("percent", ascending=False)
        return df

    @staticmethod
    def default_gpu_event_filter(event: dict):
        return event.get("tid", 200) < 100 # ignore of supplemental events

    @staticmethod
    def get_just_gpu_events(events):
        return dict(filter(lambda v: len(v[1].get(GPUEventAnalyser.computation_key, {})) > 0, events.items()))


    def create_gpu_summary(self, group_kernels_by_name: bool = False):
        all_events = self.get_gpu_event_lists(event_filter = JaxGPUEventAnalyser.default_gpu_event_filter)

        # create an average across GPUs
        average_gpu_metrics = None
        num_gpus = len(self.get_just_gpu_events(all_events))
        for cur_events in all_events.items():
            self.verify_dict_gpu_event_lists(cur_events)
            current_metrics = self.compute_metrics_dict(cur_events)
            if average_gpu_metrics is None:
                average_gpu_metrics = current_metrics
            else:
                for k, v in current_metrics.items():
                    average_gpu_metrics[k] += v
        for k in average_gpu_metrics.keys():
            average_gpu_metrics[k] /= num_gpus

        # find compute times
        all_gpu_compute_events = [e for ge in all_events.values() for e in ge[GPUEventAnalyser.computation_key]]
        categorized_times, uncategorized_times = self.breakdown_compute_events(all_gpu_compute_events,
                                                                           group_by_gpu = False,
                                                                           group_by_name = group_kernels_by_name)

        categorized_df = self.create_breakdown_df(categorized_times, average_gpu_metrics["computation_time"])
        uncategorized_df = self.create_breakdown_df(uncategorized_times, categorized_times[self.UncategorizedEventKey][1])
        return self.get_breakdown_df_from_dict(average_gpu_metrics), categorized_df, uncategorized_df

    @staticmethod
    def summarize_gpu_events(filename):
        from ..util import DataLoader
        data = DataLoader.load_data(filename)
        events = data['traceEvents']
        my_gpu_event_analyser = JaxGPUEventAnalyser(events)
        return my_gpu_event_analyser.create_gpu_summary()

    communication_events_map={"all-gather-start":"all-gather", "all-reduce-start":"all-reduce", "reduce-scatter":"reduce-scatter", "collective-permute-start": "collective-permute"}

    @staticmethod
    def process_communication_events_from_event_dump(xla_file_name: str) -> dict:
        import re
        communication_events={key:[] for key in JaxGPUEventAnalyser.communication_events_map.keys()}

        event_key=str.join('|', JaxGPUEventAnalyser.communication_events_map.keys())
        pattern = re.compile(f"^.*value:.*({event_key})\.?([\d]+)?.*size=(\d+).*: ([a-zA-Z\d].*)\[.*$")
        for line in open(xla_file_name, "r"):
            m=pattern.search(line)
            if m:
                communication_events[m.group(1)].append([m.group(2), m.group(3), m.group(4)])

        return communication_events

    def process_communication_events_from_profile(self, messages: dict) -> dict:
        all_events = self.get_gpu_event_lists(event_filter = JaxGPUEventAnalyser.default_gpu_event_filter)
        just_gpu_events = self.get_just_gpu_events(all_events)
        all_comm_events = [e for ge in just_gpu_events.values() for e in ge[GPUEventAnalyser.communication_key]]
        num_gpus = len(just_gpu_events)

        rccl_stats={}

        for i in all_comm_events:
            pid=i["pid"]
            dur=i["dur"]
            op = i["args"]["hlo_op"]
            if op.startswith('reduce-scatter'):
                op = '.'.join(op.split('.')[:2]) # need to remove sub-communications from reduce-scatter only
            current = rccl_stats.get(op, [math.inf] * num_gpus)
            current[pid-1] = dur
            rccl_stats[op] = current


        #each dict is indexed by the hlo_op, and the value is a list [duration, total message size, number of tuple arguments,algbw]
        output = {}
        for msg_type, msg_values in messages.items():
            coll_dict={}
            output[JaxGPUEventAnalyser.communication_events_map[msg_type]] = coll_dict
            for msg in msg_values:
                collname=f"{msg_type}.{msg[0]}" if msg[0] is not None else msg_type
                collsize=int(msg[1])
                collval = rccl_stats.get(collname, None)
                if (collval is not None):
                    current = coll_dict.get(collname, [min(collval),0,0,0])
                    current[1] += collsize
                    current[2] += 1
                    coll_dict[collname] = current
                else:
                    print(collname," not found")
            scale = num_gpus if "reduce-scatter" in msg_type else 1
            for collname, current in coll_dict.items():
                current[3]=current[1]*scale*0.001/current[0]

        return output

    @staticmethod
    def summarize_communication_data(comm_event_data):
        summary_data = {}
        for collective, collective_stats in comm_event_data.items():
            current_data = [[collective, xfer_name, data[0], data[1] / 1024, data[3]]
                            for xfer_name, data in collective_stats.items()]
            df = pd.DataFrame(data=current_data,
                            columns = [
                                "base_collective",
                                "collective_name",
                                "latency_us",
                                "buffer_size_kb",
                                "effective_bw" ])

            bandwidth_stats = (
                df.groupby(["base_collective", "buffer_size_kb"])["effective_bw"]
                .agg(["mean", "std"])
                .reset_index()
            )

            # Group by collective type and buffer size for call counts
            call_counts = (
                df.groupby(["base_collective", "buffer_size_kb"])
                .size()
                .reset_index(name="count")
            )

            bw_data = bandwidth_stats.sort_values("buffer_size_kb")
            count_data = call_counts[
                call_counts["base_collective"] == collective
            ].sort_values("buffer_size_kb")

            time_by_size = (
                df.groupby("buffer_size_kb")["latency_us"].sum().reset_index()
            )
            total_time_us = time_by_size["latency_us"].sum()
            time_by_size["percentage"] = (time_by_size["latency_us"] / total_time_us) * 100 if total_time_us > 0 else 0


            # Calculate time spent in each bandwidth range
            bw_thresholds = [0, 50, 100, 200, 300, 350, 400]
            total_time = (df["latency_us"].sum()) / 1e6  # Convert to seconds
            time_in_ranges = []
            labels = []

            for i in range(len(bw_thresholds) - 1):
                mask = (df["effective_bw"] >= bw_thresholds[i]) & (
                    df["effective_bw"] < bw_thresholds[i + 1]
                )
                time_in_range = (df[mask]["latency_us"].sum()) / 1e6
                percentage = (time_in_range / total_time) * 100 if total_time > 0 else 0
                time_in_ranges.append(percentage)
                labels.append(f"{bw_thresholds[i]}-{bw_thresholds[i+1]} GB/s")

            # Add the highest range
            mask = df["effective_bw"] >= bw_thresholds[-1]
            time_in_range = (df[mask]["latency_us"].sum()) / 1e6
            percentage = (time_in_range / total_time) * 100 if total_time > 0 else 0
            time_in_ranges.append(percentage)
            range_data=pd.DataFrame(zip(labels, time_in_ranges), columns=("Bandwidth range", "Percentage of time"))

            summary_data[collective]=(df, bw_data, count_data, time_by_size, range_data)
        return summary_data


    @staticmethod
    def summarize_gpu_communication_events(profile_filename, xla_filename):
        # summarizes communication events from a single step
        from ..util import DataLoader
        data = DataLoader.load_data(profile_filename)
        events = data['traceEvents']
        my_gpu_event_analyser = JaxGPUEventAnalyser(events)
        comm_xla_events = my_gpu_event_analyser.process_communication_events_from_event_dump(xla_filename)
        processed = my_gpu_event_analyser.process_communication_events_from_profile(comm_xla_events)
        return my_gpu_event_analyser.summarize_communication_data(processed)









