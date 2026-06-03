###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# event_to_category inline optimization

Changes `default_categorizer` to use a direct dictionary lookup (`event["cat"]`)
instead of `event.get(TraceEventUtils.TraceKeys.Category)`. The `"cat"` key is
guaranteed to exist on every event by `_preprocess_and_index_events`, which
stamps it (as `None` if absent) during the single O(N) setup pass. JAX traces
are unaffected and continue to use `prepare_event_categorizer`.

Baseline: upstream/main @ `26aba9bd` | Branch: `8fdbbffd`

## Total runtime

| Trace | Upstream (s) | Branch (s) | Δ (s) | Δ % |
|---|---|---|---|---|
| trace1 | 89.6 | 83.9 | −5.7 | −6% |
| trace2 | 0.48 | 0.42 | −0.06 | −12% |
| trace3 | 73.9 | 68.1 | −5.8 | −8% |
| trace4 | 74.6 | 72.3 | −2.3 | −3% |
| trace5 | 111.6 | 108.0 | −3.6 | −3% |
| trace6 | 1295.6 | 1245.5 | −50.1 | −4% |
| trace7 | 55.7 | 50.3 | −5.4 | −10% |
| trace8 | 263.7 | 232.2 | −31.5 | −12% |
| trace9 | 219.9 | 198.7 | −21.2 | −10% |

## Peak RSS

| Trace | Upstream (GB) | Branch (GB) | Δ (GB) |
|---|---|---|---|
| trace1 | 5.79 | 5.79 | 0.00 |
| trace2 | 0.17 | 0.17 | 0.00 |
| trace3 | 5.37 | 5.36 | −0.01 |
| trace4 | 4.85 | 4.85 | 0.00 |
| trace5 | 8.64 | 8.63 | −0.01 |
| trace6 | 66.64 | 66.63 | −0.01 |
| trace7 | 7.38 | 7.39 | +0.01 |
| trace8 | 51.02 | 51.02 | 0.00 |
| trace9 | 32.84 | 32.83 | −0.01 |

RSS is unchanged — the optimization eliminates function-call overhead, not
allocations.
