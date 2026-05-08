###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

__all__ = ["IdleTimeAnalyser"]


class IdleTimeAnalyser:
    """GPU idle time classifier for PyTorch profiler traces.

    Can be used standalone or as part of generate_perf_report_pytorch.

    Usage:
        from TraceLens.IdleTimeAnalyser import IdleTimeAnalyser
        analyser = IdleTimeAnalyser(tree)
        dfs = analyser.get_dataframes()
        # or
        results = analyser.classify()
    """

    def __init__(self, tree, micro_thresh_us=5.0):
        self._tree = tree
        self._micro_thresh_us = micro_thresh_us
        self._classified = None

    def classify(self):
        """Run classification. Returns list of interval dicts."""
        from .classify import classify_idle_intervals, assign_idle_ids
        if self._classified is None:
            self._classified = classify_idle_intervals(
                self._tree, micro_thresh_us=self._micro_thresh_us
            )
            assign_idle_ids(self._classified)
        return self._classified

    def get_dataframes(self, gpu_busy_time_us=None):
        """Return {"idle_overview": df, "idle_summary": df, "idle_intervals": df}."""
        from .report import build_idle_dataframes
        classified = self.classify()
        return build_idle_dataframes(classified, gpu_busy_time_us=gpu_busy_time_us)

    def get_augmented_events(self, gpu_pid):
        """Return Chrome trace annotation events for Perfetto visualization."""
        from .classify import make_annotation_events
        classified = self.classify()
        return make_annotation_events(classified, gpu_pid)
