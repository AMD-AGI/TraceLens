###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Stage 1: find iteration execution roots in an inference trace."""

from ..annotation_utils import (
    ITERATION_BACKUP_PATTERNS,
    ITERATION_PATTERNS,
    find_events_by_patterns,
)


def find_iteration_roots(events: list[dict]) -> list[dict] | None:
    """Return iteration-root events.

    Tries the primary annotation pattern first, then backup patterns, then
    falls back to generic call-tree traversal via Trace2Tree.
    """
    roots = find_events_by_patterns(
        events, ITERATION_PATTERNS, label="execution steps (iteration)", verbose=True
    )
    if len(roots) == 0:
        print("No primary annotations found; falling back to backup patterns...")
        roots = find_events_by_patterns(
            events,
            ITERATION_BACKUP_PATTERNS,
            label="execution steps (iteration, backup)",
            verbose=True,
        )
    if len(roots) == 0:
        print("No annotation patterns found; trying generic call-tree traversal...")
        from ...Trace2Tree.inference_iteration_roots import (
            find_iteration_roots_generic,
        )

        roots = find_iteration_roots_generic(events)
    return roots
