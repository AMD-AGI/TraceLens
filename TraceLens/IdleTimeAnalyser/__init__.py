###############################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from .classify import IdleThresholds

__all__ = ["IdleTimeAnalyser", "IdleThresholds"]


class IdleTimeAnalyser:
    """GPU idle time classifier for PyTorch profiler traces.

    Can be used standalone or as part of generate_perf_report_pytorch.

    Usage:
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser, IdleThresholds
        analyser = IdleTimeAnalyser(tree)        # or IdleTimeAnalyser(tree, IdleThresholds(...))
        dfs = analyser.get_dataframes()          # idle_overview, idle_groups, idle_intervals
        gaps = analyser.classify()               # list of per-gap dicts
    """

    def __init__(
        self, tree, thresholds=None, micro_thresh_us=None, include_call_stack=False
    ):
        """micro_thresh_us, if given, overrides thresholds.noise_us.

        include_call_stack adds sync_entry_point, sync_call_stack_full and
        covering_call_stack_full to idle_intervals and splits the overview drain rows by
        call site; build the tree with add_python_func=True for Python frames, as the
        perf report does.
        """
        from dataclasses import replace

        th = thresholds or IdleThresholds()
        if micro_thresh_us is not None:
            th = replace(th, noise_us=micro_thresh_us)
        self.thresholds = th
        self.include_call_stack = include_call_stack
        self._tree = tree
        self._gpu_events = None
        self._classified = None

    def _gpu_event_list(self):
        from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser

        if self._gpu_events is None:
            self._gpu_events = GPUEventAnalyser(
                self._tree.events
            ).get_gpu_event_lists()[GPUEventAnalyser.all_gpu_key]
        return self._gpu_events

    def classify(self):
        """Run classification. Returns list of interval dicts."""
        from .classify import assign_idle_ids, classify_idle_intervals

        if self._classified is None:
            self._classified = classify_idle_intervals(
                self._tree,
                self.thresholds,
                gpu_events=self._gpu_event_list(),
                include_call_stack=self.include_call_stack,
            )
            assign_idle_ids(self._classified)
        return self._classified

    def gpu_busy_time_us(self):
        """Union of GPU event intervals; matches the gpu_timeline sheet's busy_time."""
        from .classify import gpu_busy_time_us

        return gpu_busy_time_us(self._gpu_event_list())

    def get_dataframes(self, gpu_busy_time_us=None):
        """Return {"idle_overview": df, "idle_groups": df, "idle_intervals": df}."""
        from .report import build_idle_dataframes

        busy = (
            gpu_busy_time_us
            if gpu_busy_time_us is not None
            else self.gpu_busy_time_us()
        )
        return build_idle_dataframes(
            self.classify(),
            gpu_busy_time_us=busy,
            thresholds=self.thresholds,
            include_call_stack=self.include_call_stack,
        )

    def get_augmented_events(self, gpu_pid):
        """Return Chrome trace annotation events for Perfetto visualization."""
        from .classify import make_annotation_events

        return make_annotation_events(self.classify(), gpu_pid)
