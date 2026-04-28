###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Bug verification tests — proves that specific bugs exist in the codebase.

Each test is structured as:
  - A "prove the bug" assertion that documents the WRONG current behaviour.
  - A "correct behaviour" assertion that shows what the code SHOULD do.

When a bug is fixed, the "prove the bug" assertion will start failing (which is
expected — delete or invert it) and the "correct behaviour" assertion will pass.

Run:
    pytest tests/test_bug_verification.py -v

Add new bugs by following the template at the bottom of this file.
"""

import pytest
from TraceLens.util import TraceEventUtils

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

# Canonical Chrome Trace Format phase strings (from TraceEventUtils enums)
PHASE_KEY = TraceEventUtils.TraceKeys.Phase  # "ph"
PHASE_METADATA = TraceEventUtils.TracePhases.Metadata  # "M"
PHASE_COMPLETE = TraceEventUtils.TracePhases.Complete  # "X"


def _make_metadata_event(pid=1, tid=1, thread_label="Stream #1") -> dict:
    """
    Minimal Chrome-trace metadata event (ph='M').

    The 'name' field must be a recognised MetadataFields key (e.g. 'thread_name')
    so that split_event_list() can parse it.  The 'args.name' value is the human-
    readable label that get_event_category() later looks up as ThreadName.
    """
    return {
        "ph": PHASE_METADATA,  # "M"
        "name": "thread_name",  # must match MetadataFields.ThreadName
        "pid": pid,
        "tid": tid,
        "args": {"name": thread_label},  # the actual thread label
    }


def _make_kernel_event(pid=1, tid=1, name="rocblas_kernel") -> dict:
    """Minimal Chrome-trace complete/kernel event (ph='X')."""
    return {
        "ph": PHASE_COMPLETE,  # "X"
        "name": name,
        "pid": pid,
        "tid": tid,
        "ts": 1000,
        "dur": 100,
    }


class TestBug1BooleanDictKey:
    """
    BUG #1 — util.py:514
    Boolean passed as dict key inside event.get(), so metadata events are never detected.

    Root cause:
        event.get(
            TraceEventUtils.TraceKeys.Phase == TraceEventUtils.TracePhases.Metadata
        )
        The inner expression ("ph" == "M") is evaluated first, producing the boolean
        False. event.get(False) then looks for the key False in the dict, which never
        exists, so the branch is never entered and metadata events are silently
        mis-categorised as "Unknown".

    Fix (util.py:514):
        Change:
            if event.get(
                TraceEventUtils.TraceKeys.Phase == TraceEventUtils.TracePhases.Metadata
            ):
        To:
            if event.get(TraceEventUtils.TraceKeys.Phase) == TraceEventUtils.TracePhases.Metadata:
    """

    def test_metadata_event_is_categorised_as_metadata(self):
        """
        A metadata event (ph='M') must be categorised as 'metadata'.

        FAILS while the bug exists.
        PASSES once the fix is applied.
        """
        pid, tid, label = 1, 1, "Stream #1"
        metadata_event = _make_metadata_event(pid=pid, tid=tid, thread_label=label)
        metadata = {pid: {tid: {TraceEventUtils.MetadataFields.ThreadName: label}}}

        result = TraceEventUtils.get_event_category(metadata, metadata_event)

        assert result == "metadata", (
            f"Expected 'metadata', got {result!r}. "
            "Fix: event.get(TraceKeys.Phase) == TracePhases.Metadata"
        )


class TestBug3PropertyStaticmethod:
    """
    BUG #3 — gpu_event_analyser.py:75
    @property + @staticmethod stacked returns a descriptor object, not an iterable.

    Fix: replace @property+@staticmethod with @classmethod+@property
    """

    def test_all_event_keys_is_iterable(self):
        """
        FAILS while the bug exists.
        PASSES once the fix is applied.
        """
        from TraceLens.TreePerf.gpu_event_analyser import GPUEventAnalyser

        result = GPUEventAnalyser.all_event_keys
        assert hasattr(result, "__iter__"), (
            f"Expected iterable, got {type(result).__name__}. "
            "Fix: replace @property+@staticmethod with @classmethod+@property"
        )


class TestBug6BytesBwdWrongKwargs:
    """
    BUG #6 — perf_model.py:133
    bytes_bwd() passes bytes_per_element= but bytes_func expects
    bpe_mat1, bpe_mat2, bpe_bias, bpe_output — raises TypeError on every call.
    """

    def test_bytes_bwd_does_not_raise_typeerror(self):
        """
        FAILS while the bug exists (TypeError).
        PASSES once the fix is applied.
        """
        from TraceLens.PerfModel.perf_model import GEMM

        class _MinimalGEMM(GEMM):
            @staticmethod
            def get_param_details(event):
                return {"M": 4, "N": 4, "K": 4, "bias": False, "B": 1}

        gemm = _MinimalGEMM({})
        result = gemm.bytes_bwd(bytes_per_element=2)
        assert result is not None


class TestBug8NullLaunchEvent:
    """
    BUG #8 — trace_fuse.py:121
    next(..., None) returns None when no matching launch event exists;
    the very next line subscripts launch_event["args"] → TypeError.
    """

    def test_set_linking_key_raises_on_missing_launch_event(self):
        """
        FAILS while the bug exists (TypeError).
        PASSES once the fix is applied.
        """
        from TraceLens.TraceFusion.trace_fuse import TraceFuse

        fuse = object.__new__(TraceFuse)
        # events list has no cuda_runtime/cuda_driver "launch" event
        events = [{"cat": "cpu_op", "name": "aten::mm", "args": {}}]
        try:
            fuse._set_linking_key(events)
            # If no exception, the bug is fixed — linking_key should be set
            assert hasattr(fuse, "linking_key")
        except TypeError:
            pytest.fail(
                "TypeError raised: launch_event is None but code subscripts it. "
                "Fix: guard with 'if launch_event is None' before accessing launch_event[\"args\"]"
            )


class TestBug9ShallowCopyMutatesIR:
    """
    BUG #9 — event_replay.py:417
    list.copy() is shallow — inner dicts are shared references.
    Mutating the copy's inner dict also mutates the original list.

    NOTE: Unlike other tests in this file, test_shallow_copy_mutates_original
    is a "prove the behavior" test, not a regression guard. It passes while the
    bug exists (asserting the wrong behavior is happening) and will pass forever
    because it tests plain Python list.copy() semantics, not the application code.
    A proper regression test would require calling EventReplayer.get_repro_info()
    with a real TensorCfg object (needs torch), so this minimal version is kept
    as documentation of the root cause only.
    """

    def test_shallow_copy_mutates_original(self):
        """
        PROVES the bug: shallow copy shares inner dict references.
        This test documents the WRONG behaviour (mutation of original).
        If this assertion starts FAILING, the bug has been fixed.
        """
        original = [{"value": "original_value"}]
        copy = original.copy()
        copy[0]["value"] = "mutated"
        # Bug: original is also mutated because copy shares the inner dict
        assert original[0]["value"] == "mutated", (
            "If this fails, the shallow-copy bug no longer applies to plain list.copy(). "
            "The fix is to use copy.deepcopy() in event_replay.py:417."
        )

    def test_deepcopy_does_not_mutate_original(self):
        """
        CORRECT behaviour: deepcopy must isolate inner dicts.
        """
        import copy

        original = [{"value": "original_value"}]
        deep = copy.deepcopy(original)
        deep[0]["value"] = "mutated"
        assert (
            original[0]["value"] == "original_value"
        ), "deepcopy must not mutate the original list."


class TestBug10DefaultCategorizerSig:
    """
    BUG #10 — trace_to_tree.py:321
    JaxTraceToTree.default_categorizer is @staticmethod expecting one arg (event),
    but the base class calls self.default_categorizer() with zero args → TypeError.
    """

    def test_jax_trace_to_tree_default_categorizer_raises(self):
        """
        FAILS while the bug exists (TypeError on instantiation).
        PASSES once the fix is applied.
        """
        from TraceLens.Trace2Tree.trace_to_tree import JaxTraceToTree

        try:
            # Instantiation triggers base __init__ → self.default_categorizer()
            # With @staticmethod expecting event arg, this raises TypeError
            JaxTraceToTree(events_data=[], event_to_category=None)
            # If we get here, the bug is fixed
        except TypeError as exc:
            pytest.fail(
                f"TypeError on JaxTraceToTree init: {exc}. "
                "Fix: change default_categorizer from @staticmethod to a proper instance method."
            )
        except Exception:
            # Other errors (e.g. missing data) are acceptable — we only care about TypeError
            pass


class TestBug11Log10Zero:
    """
    BUG #11 — trace_fuse.py:136
    math.log10(max_value) crashes with ValueError when max_value == 0
    (defaultdict(int) default for fields with no integer values).
    """

    def test_set_offset_multiplier_raises_on_zero_max_value(self):
        """
        FAILS while the bug exists (ValueError: math domain error).
        PASSES once the fix is applied.
        """
        import math
        from TraceLens.TraceFusion.trace_fuse import TraceFuse

        fuse = object.__new__(TraceFuse)
        fuse.linking_key = "correlation"
        fuse.fields_to_adjust_offset = ["id"]
        # Event with "id": 0 → max_values["id"] = 0 → math.log10(0) crashes
        events = [{"cat": "cpu_op", "name": "op", "id": 0, "args": {}}]
        try:
            fuse._set_offset_multiplier(events)
        except ValueError as exc:
            if "math domain error" in str(exc) or "log10" in str(exc).lower():
                pytest.fail(
                    f"ValueError from math.log10(0): {exc}. "
                    "Fix: use math.log10(max(max_value, 1)) to guard against zero."
                )
            raise


# ---------------------------------------------------------------------------
# Template for adding future bug tests
# ---------------------------------------------------------------------------
#
# class TestBugN<ShortName>:
#     """<file>:<line> — one-line description of the bug."""
#
#     def test_proves_bug_<specific_symptom>(self):
#         """Show the wrong current behaviour."""
#         ...
#         assert <wrong_thing_happens>, "If this fails the bug may be fixed."
#
#     def test_correct_behaviour_<what_should_happen>(self):
#         """Show what the correct behaviour looks like."""
#         ...
#         assert <right_thing_happens>
#
# Bugs to add next (in priority order):
#   TestBug2NoneComparisonCrash     — trace_to_tree.py:358,425  (None <= 100)
#   TestBug3PropertyStaticmethod    — gpu_event_analyser.py:74  (@property+@staticmethod)
#   TestBug4HasattrOnDict           — perf_model.py:46          (hasattr vs "in")
#   TestBug5KernelNamesUnbound      — perf_model.py:36-40       (UnboundLocalError)
#   TestBug6BytesBwdWrongKwargs     — perf_model.py:123-139     (TypeError on bytes_bwd)
#   TestBug7IndexErrorAfterWarning  — util.py:106-112           (hlo_filename[0])
#   TestBug8NullLaunchEvent         — trace_fuse.py:115         (None dereference)
#   TestBug9ShallowCopyMutatesIR    — event_replay.py:417       (deepcopy needed)
#   TestBug10DefaultCategorizerSig  — trace_to_tree.py:319      (signature mismatch)
#   TestBug11Log10Zero              — trace_fuse.py:129         (math.log10(0))
