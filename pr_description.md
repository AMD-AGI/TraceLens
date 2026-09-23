## Summary

- When a capture folder is provided, `TreePerfAnalyzer.from_file()` was loading and parsing
  the replay trace, then discarding the result and letting `merge_capture_trace_into_graph()`
  _reload the same file from disk_
- For large traces like DeepSeek V4 Flash (e.g. 36.9M events) this caused 2x decompression, 2x JSON parsing,
  2x tree construction, and ~170GB+ peak memory, eventually failing with OOM.

## This PR

- Adds an optional `graph_tree` parameter to `merge_capture_trace_into_graph()` to accept a
  pre-built tree, skipping the redundant load and making large trace processing complete without OOM error.
