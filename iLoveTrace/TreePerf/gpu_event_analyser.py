import pandas as pd

class GPUEventAnalyser:
    def __init__(self, events):
        """
        Initialize with a list of event dictionaries. Each event is expected to have
        'ts', 't_end', and 'name' keys.
        """
        self.events = events
        if events:
            self.overall_start = min(event['ts'] for event in events)
            self.overall_end = max(event['t_end'] for event in events)
            self.total_time = self.overall_end - self.overall_start
        else:
            self.overall_start = None
            self.overall_end = None
            self.total_time = 0

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

    def compute_metrics(self):
        """
        Compute the following percentages relative to the overall timeline:
          - computation_percent: time covered by computation events.
          - exposed_communication_percent: time for communication events (names containing 'nccl')
            after subtracting any overlapping computation time.
          - exposed_memcpy_percent: time for memcpy events (names containing 'memcpy')
            after subtracting any overlapping computation time.
          - idle_percent: time not covered by any event.
        """
        if not self.events or self.total_time <= 0:
            return None

        # Categorize events.
        comp_intervals = []
        comm_intervals = []
        memcpy_intervals = []
        for event in self.events:
            start = event['ts']
            end = event['t_end']
            name = event.get('name', '').lower()
            if 'nccl' in name:
                comm_intervals.append((start, end))
            elif 'memcpy' in name:
                memcpy_intervals.append((start, end))
            else:
                comp_intervals.append((start, end))

        # Merge intervals within each category.
        comp_union = self.merge_intervals(comp_intervals)
        comm_union = self.merge_intervals(comm_intervals)
        memcpy_union = self.merge_intervals(memcpy_intervals)
        all_intervals = self.merge_intervals(
            [(event['ts'], event['t_end']) for event in self.events]
        )

        # Compute total computation time.
        comp_time = sum(end - start for start, end in comp_union)

        # For communication, subtract overlapping computation time.
        exposed_comm_time = 0
        for comm_interval in comm_union:
            remaining_parts = self.subtract_intervals(comm_interval, comp_union)
            exposed_comm_time += sum(end - start for start, end in remaining_parts)

        # For memcpy, subtract overlapping computation time.
        exposed_memcpy_time = 0
        for memcpy_interval in memcpy_union:
            remaining_parts = self.subtract_intervals(memcpy_interval, comp_union)
            exposed_memcpy_time += sum(end - start for start, end in remaining_parts)

        # Idle time is the gap between overall timeline and busy intervals.
        busy_time = sum(end - start for start, end in all_intervals)
        idle_time = self.total_time - busy_time

        #TODO add communication time
        return {
            "busy_time": busy_time,
            "computation_time": comp_time,
            "exposed_communication_time": exposed_comm_time,
            "exposed_memcpy_time": exposed_memcpy_time,
            "idle_time": idle_time,
            "total_time": self.total_time,
        }
    
    def get_breakdown_df(self):
        dict_metrics = self.compute_metrics()
        # df = pd.DataFrame(dict_metrics, index=[0])
        # Compute percentages based on total time.
        # df['computation_percent'] = df['computation_time'] / df['total_time'] * 100
        # df['exposed_communication_percent'] = df['exposed_communication_time'] / df['total_time'] * 100
        # df['exposed_memcpy_percent'] = df['exposed_memcpy_time'] / df['total_time'] * 100
        # df['idle_percent'] = df['idle_time'] / df['total_time'] * 100
        # df['total_percent'] = df['computation_percent'] + df['exposed_communication_percent'] + df['exposed_memcpy_percent'] + df['idle_percent']
        # we need rows to be the computtation, communication, memcpy, idle, total
        # cols to be time, percent
        df = pd.DataFrame(dict_metrics.items(), columns=['type', 'time'])
        # convert time to ms by div by 1e3
        df['time ms'] = df['time'] / 1e3
        df['percent'] = df['time'] / df.loc[df['type'] == 'total_time', 'time'].values[0] * 100
        df = df.drop(columns=['time'])

        return df
