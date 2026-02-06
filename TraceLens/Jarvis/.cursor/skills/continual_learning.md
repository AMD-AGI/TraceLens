---
name: Post-Analysis Continual Learning
description: Update category analyzer patterns when a missed issue is reported. Use only when the user explicitly reports a missed analysis finding and confirms it should be added as a new pattern. Do NOT invoke automatically.
triggers:
  - missed issue
  - update patterns
  - self-improvement
  - add pattern
  - continual learning
tools:
  - file_read
  - file_write
---

# Post-Analysis Continual Learning

Update category analyzer agent files with new "Common Patterns" entries based on confirmed missed issues from standalone analysis runs.

**Invocation:** User-initiated only. The user must describe a missed issue and confirm it is valid before any changes are made.

---

## Workflow

### Step 1: Understand the Missed Issue

Ask the user to provide:
1. **What was missed** -- the issue the analysis failed to flag
2. **Which category** -- which analyzer(s) should have caught it (gemm, sdpa, elementwise, reduce, triton, moe, batchnorm, convolution, cpu-idle, other)
3. **Evidence** -- how they identified the issue (tree traversal, manual inspection, etc.)

If the user has already provided this context in the conversation, do not re-ask.

### Step 2: Read the Target Agent File(s)

Read the relevant agent file(s) from the agent file map below. Focus on the `## Common Patterns` section to understand existing entries and the format used in that specific file.

### Step 3: Check for Duplicate Patterns

Scan existing patterns in the target agent to confirm the missed issue is not already covered. If an existing pattern partially covers it, note which fields need to be added or extended.

### Step 4: Compose a New Pattern Entry

Write a new `###` pattern entry that:
- Uses the **exact same field names and bullet format** as the other entries in that agent's Common Patterns section
- Includes at minimum: **Symptoms** and **Algorithmic** (these appear in every agent)
- Adds optional fields (**Issue**, **Look for**, **Impact**, **Kernel**, **Expected**, **Note**) only when they add diagnostic value
- Keeps the entry concise (4-8 bullet lines)
- Do not use specific numbers from the given trace and keep it general

### Step 5: Present for Review

Show the user:
1. Which agent file will be modified
2. The exact new pattern entry to be appended
3. Where it will be inserted (end of the Common Patterns section, before the next `---` or `## Key Principles` separator)

**Do NOT write any changes until the user approves.**

### Step 6: Apply the Change

After user approval, insert the new pattern entry at the end of the `## Common Patterns` section in the target agent file. Do not modify any other section or any existing pattern entry.

---

## Constraints

1. **Append-only** -- never modify or delete existing pattern entries
2. **Common Patterns section only** -- do not touch any other section (Workflow, Error Handling, Key Principles, Efficiency Thresholds, etc.)
3. **Match format exactly** -- use the same bullet style (`- **Field:**`) and heading level (`###`) as the target agent
4. **One pattern per agent per invocation** -- if multiple agents need updates, handle them sequentially
5. **No logic changes** -- do not alter analysis scripts, workflow steps, or thresholds
6. **User confirmation required** -- always present the proposed change before writing

---

## Agent File Map

| Category | Agent File |
|----------|-----------|
| gemm | `TraceLens/Jarvis/.cursor/agents/gemm-analyzer.md` |
| sdpa_fwd | `TraceLens/Jarvis/.cursor/agents/sdpa-analyzer.md` |
| elementwise | `TraceLens/Jarvis/.cursor/agents/elementwise-analyzer.md` |
| reduce | `TraceLens/Jarvis/.cursor/agents/reduce-analyzer.md` |
| triton | `TraceLens/Jarvis/.cursor/agents/triton-analyzer.md` |
| moe_fused | `TraceLens/Jarvis/.cursor/agents/moe-analyzer.md` |
| batchnorm | `TraceLens/Jarvis/.cursor/agents/batchnorm-analyzer.md` |
| convolution | `TraceLens/Jarvis/.cursor/agents/convolution-analyzer.md` |
| cpu_idle | `TraceLens/Jarvis/.cursor/agents/cpu-idle-analyzer.md` |
| other | `TraceLens/Jarvis/.cursor/agents/generic-op-analyzer.md` |

All paths are relative to `TraceLens/` repository root.

---

## Pattern Entry Format Reference

Every agent uses this structure with `###` headings and `- **Field:**` bullets:

```markdown
### Pattern Title
- **Symptoms:** Observable indicators in the trace data
- **Issue:** What the problem is (optional but recommended)
- **Look for:** Specific operations or call patterns to search for (optional)
- **Algorithmic:** Model or framework-level recommendation
- **Kernel:** GPU kernel-level recommendation (optional)
- **Impact:** Expected improvement if addressed (optional)
```

Required fields: **Symptoms**, **Algorithmic**
Common optional fields: **Issue**, **Look for**, **Impact**, **Kernel**, **Expected**, **Note**, **Cause**

Match the specific optional fields used by the target agent -- do not introduce field names that agent does not already use.
