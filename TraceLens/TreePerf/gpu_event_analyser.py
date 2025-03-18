import pandas as pd

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
    def subtract_intervals(interval, intervals_to_subtract):
        """
        Subtract a list of intervals (assumed non-overlapping and sorted) from a given interval.
        Returns a list of intervals (as (start, end) tuples) that represent the parts of 'interval'
        not covered by any of the intervals_to_subtract.
        """
        result = []
        current_start, current_end = interval
        for sub_start, sub_end in intervals_to_subtract:
            # Skip if there is no overlap.
            if sub_end <= current_start or sub_start >= current_end:
                continue
            # Add gap before the subtracting interval if any.
            if sub_start > current_start:
                result.append((current_start, sub_start))
            current_start = max(current_start, sub_end)
            if current_start >= current_end:
                break
        if current_start < current_end:
            result.append((current_start, current_end))
        return result

    @staticmethod
    def subtract_intervalsA_from_B(merged_intervalA, merged_intervalB):
        """
        Subtract a set of non-overlapping, sorted intervals from another set of non-overlapping, sorted intervals.
        Returns a list of intervals (as (start, end) tuples) that represent the parts of 'merged_intervalB'
        """
        result = []
        for b_interval in merged_intervalB:
            # Subtract the entire set of merged_A from the current interval b_interval.
            remaining_parts = GPUEventAnalyser.subtract_intervals(b_interval, merged_intervalA)
            result.extend(remaining_parts)

        return result


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
            'all_gpu': gpu_events,
            'computation': comp_events,
            'communication': comm_events,
            'memcpy': memcpy_events,
        }

    def get_gpu_event_lists_jax(self):
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
            args = event.get('args')
            if args is not None:
                device_id = args.get('device_id')
                if device_id is not None:
                    if 't_end' not in event:
                        event['t_end'] = event['ts'] + event['dur']
                    gpu_events.append(event)
                    name = event.get('name')
                    if name.startswith('hipMemcpy'):
                        memcpy_events.append(event)
                    elif name.startswith('nccl'):
                        comm_events.append(event)
                    else:
                        comp_events.append(event)
        return {
            'all_gpu': gpu_events,
            'computation': comp_events,
            'communication': comm_events,
            'memcpy': memcpy_events,
        }


    @staticmethod
    def verify_dict_gpu_event_lists(dict_gpu_event_lists):
        # first check if the keys are correct
        expected_keys = {'all_gpu', 'computation', 'communication', 'memcpy'}
        if set(dict_gpu_event_lists.keys()) != expected_keys:
            raise ValueError(f"Expected keys: {expected_keys}, got: {dict_gpu_event_lists.keys()}")
        # next check if the events have 'ts' and 't_end' fields
        for key, events in dict_gpu_event_lists.items():
            for event in events:
                if 'ts' not in event or 't_end' not in event:
                    raise ValueError(f"Event {event} does not have 'ts' or 't_end' fields")
        if len(dict_gpu_event_lists['all_gpu']) == 0:
            raise ValueError("No GPU events found in the trace")

    def compute_metrics(self, jax: bool = False):
        """
        Compute various metrics from the GPU event data.
        Computation is defined as the time spent in computation kernels.
        Communication is defined as the time spent in communication kernels.
        Memcpy is defined as the time spent in memcpy kernels.
        Exposed communication time is the time spent in communication kernels that is not overlapped by computation.
        Exposed memcpy time is the time spent in memcpy kernels that is not overlapped by computation or communication.
        """

        # Categorize events.
        dict_gpu_event_lists = self.get_gpu_event_lists() if not jax else self.get_gpu_event_lists_jax()
        GPUEventAnalyser.verify_dict_gpu_event_lists(dict_gpu_event_lists)

        dict_intervals = {}
        for key, events in dict_gpu_event_lists.items():
            dict_intervals[key] = [(event['ts'], event['t_end']) for event in events]

        # Merge intervals within each category.
        comp_union = self.merge_intervals(dict_intervals['computation'])
        comm_union = self.merge_intervals(dict_intervals['communication'])
        memcpy_union = self.merge_intervals(dict_intervals['memcpy'])
        all_intervals = self.merge_intervals(dict_intervals['all_gpu'])

        # end of the last event - start of the first event
        total_time = all_intervals[-1][1] - all_intervals[0][0]


        comp_time = sum(end - start for start, end in comp_union)

        total_comm_time = sum(end - start for start, end in comm_union)
        exposed_comm_intervals = self.subtract_intervalsA_from_B(comp_union, comm_union)
        exposed_comm_time = sum(end - start for start, end in exposed_comm_intervals)

        total_memcpy_time = sum(end - start for start, end in memcpy_union)
        memcpy_minus_compute = self.subtract_intervalsA_from_B(comp_union, memcpy_union)
        exposed_memcpy_intervals = self.subtract_intervalsA_from_B(comm_union, memcpy_minus_compute)
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

    def get_breakdown_df(self, jax: bool = False):
        dict_metrics = self.compute_metrics(jax = jax)
        df = pd.DataFrame(dict_metrics.items(), columns=['type', 'time'])
        # convert time to ms by div by 1e3
        df['time ms'] = df['time'] / 1e3
        df['percent'] = df['time'] / df.loc[df['type'] == 'total_time', 'time'].values[0] * 100
        df = df.drop(columns=['time'])

        return df
