# TraceLens review rules

The canonical rule set for the `review-pr` skill. Do not read top to bottom during a review: use the
index to pick the rules the diff at hand can break, then read only those.

**Scope.** Rules apply to the lines a PR adds or changes. Existing debt is flagged only when the PR
touches it or adds a new copy of it.

**Each check has exactly one owner — deterministic CI or this review, never both.** These are
CI-owned; the review never flags them:

- `black==26.3.1`, `ruff` (`F401`/`F841`/`RET502`/`RET503`), `import-linter` cycles — `lint.yml`
- copyright headers (`tests/update_copyright.py`), 95% coverage + Codecov — `unit-tests.yml`
- CodeQL security scan — `codeql.yml`

**Proposed CI (follow-up).** Until these land as gates, the review flags exactly the named check on
changed lines; once a check lands, it moves to the CI-owned list above and leaves this review:
ruff `B006` (mutable defaults), `F811` (redefinition), `F403` (wildcard import), `F601` (duplicate
dict key), `A001`/`A002` (shadowed builtins), `PLC0415` (non-top-level import), `TID252` (relative
import), and an `import-linter` `layers` contract (lower layers never import higher).

**Owners.** The canonical owning layer for each concern lives in
[`references/tiers.md`](references/tiers.md) and is cited by the structure and logic rules, so the
rules stay stable as code moves.

This file is seeded from issue #1076 § 2. Every rule is `blocking` unless the review determines the
shape is correct; non-blocking observations are adjudicated and then not published (they would break
the two-part card).

---

## Index

| The diff... | Read |
|---|---|
| adds a function, class or module; copies, moves or consolidates code | S1 S2 S3 S6 |
| re-loads a trace, rebuilds the event tree, or recomputes a value another layer produces | S1 L2 |
| adds an import, an import-contract exception, or a `sys.path` edit | S4 Q3 |
| adds a model-specific or one-off script, validation scaffolding, or a file another repo owns | S5 |
| changes a signature, adds a parameter/flag, passes a raw dict across modules, or checks capability by `hasattr`/signature | L1 |
| hardcodes a path, shape, threshold, model or vendor id; classifies by name substring; adds a fallback constant | L2 |
| adds or reads an env var | L3 |
| changes output columns, JSON fields, report markers, or a public API | L4 L5 D1 |
| names a physical quantity, field or unit | L5 |
| writes rows or keys, or iterates a set/dict to produce output | L6 |
| adds class- or module-level mutable state, a cache, or work in `__init__` | L7 |
| adds a comment, TODO, dead/unreachable code, an unused parameter, or history narration | Q1 |
| adds `except`/`try`, downgrades a failure to a warning, or `print` in library code | Q2 |
| adds or changes DataFrame work | Q4 |
| adds or changes logic, or adds a test fixture | Q5 |
| adds a runtime dependency, package-data rule, or manifest entry | Q6 |
| touches docs, `README`, CLI help, a `SKILL.md`, `*.md`/`*.rst` | D1 D2 D3 D4 |

D5 applies to every PR: the title and description must match the diff at head. The review *method*
— confirm-introduced-by-diff, reproduce, name the root cause, check CI at head — lives in SKILL.md's
Steps 0–8, not in a rule.

---

## S -- Structure

### S1 -- Use the owning layer
New code calls the owner of each concern (see [`references/tiers.md`](references/tiers.md)). It never
re-loads or re-parses an input, rebuilds a structure, or recomputes a value that another layer
already produced.
**Report:** `S1 <file>:<line> -- re-does <concern> owned by <owner>; call it instead`

### S2 -- Extend, don't copy
New behavior or a new variant extends the existing module through a parameter, a hook on a shared
base, or a registry entry. Findings: a copied module; a function redefined from a sibling module; a
function or class about 80% or more similar to an existing one that neither subclasses nor calls it;
the same variation expressed twice (a flag and a subclass).
**Report:** `S2 <file>:<line> -- duplicates <existing>; extend it instead`

### S3 -- Define shared declarations once
A constant, type or size map, unit factor, column name, category string, regex, or helper used in
more than one place is defined once and imported. A new local copy or shadow, including a nested or
test helper, is a finding.
**Report:** `S3 <file>:<line> -- second copy of <declaration>; define once and import`

### S4 -- Justify import-contract exceptions
Every new exception to an import contract (`.importlinter` `ignore_imports`) states its reason.
**Report:** `S4 <file>:<line> -- new import-contract exception with no stated reason`

### S5 -- Keep the production package production-only
The library never contains validation scaffolding, files owned and run by another repo, or
model-specific or one-off scripts (those belong in `examples/` or `scripts/`).
**Report:** `S5 <file>:<line> -- non-production <thing> in the library package`

### S6 -- Don't grow large files
A change that adds a class or more than 100 lines to a file over 1,500 lines states why the code is
not in a new module. Simulation, subprocess, and I/O code stays out of pure-computation modules.
**Report:** `S6 <file>:<line> -- adds <N> lines / a class to a 1,500+ line file with no reason given`

## L -- Logic

### L1 -- Take the narrowest typed interface
Accept the object needed, not a hand-built file or a whole report when one field is enough.
Cross-module state is a dataclass or Protocol, not a raw dict. Capability is declared by an ABC or
Protocol, never found by `hasattr` or signature inspection. A new flag on a function with more than
8 parameters goes into an options object. Inputs are not mutated unless the contract says so.
**Report:** `L1 <file>:<line> -- takes <wide input / raw dict / hasattr check>; narrow it`

### L2 -- Derive, never guess
Read paths, shapes, and thresholds from the data or the owning layer; no hardcoded model or vendor
identifiers and no unnamed threshold literals. A missing or unrecognized input yields an error,
`None`, or an explicit "estimated" flag, never a fallback constant or guessed type. Classify from
structural fields, not name substrings.
**Report:** `L2 <file>:<line> -- guesses <value> via <hardcode/substring/fallback>; derive it`

### L3 -- Document env-var semantics
Each env var has a comment stating why it exists, its precedence over discovered values, and its
allowed values.
**Report:** `L3 <file>:<line> -- env var <NAME> documents no reason/precedence/allowed values`

### L4 -- Keep output contracts stable
Output columns, JSON fields, and report markers are a downstream API. Adding keys is safe; a rename
or removal is breaking and is called out. A column name states the true source of its value.
**Report:** `L4 <file>:<line> -- renames/removes <field>; breaking, undeclared`

### L5 -- Use one name and one unit per quantity
A physical quantity has one field name and one unit across the repo, stated in the name or
docstring.
**Report:** `L5 <file>:<line> -- <quantity> uses a second name/unit`

### L6 -- Emit deterministic output
Rows and keys are written in a stable order, never dependent on set or hash iteration order.
**Report:** `L6 <file>:<line> -- output order depends on set/hash iteration`

### L7 -- No hidden state, no heavy constructors
No mutable class-level or module-level state. A cache belongs to an instance, is keyed by the input
it was built from, and is bounded. No subprocess, simulation, or file I/O in `__init__`.
**Report:** `L7 <file>:<line> -- <shared mutable state / heavy __init__>`

## Q -- Code quality

### Q1 -- Remove dead and parked code
No commented-out code, inline TODOs, unused parameters, unreachable code, or guards for states the
producer never emits. Comments and docstrings describe the current code, not its history. Follow-ups
go in a ticket.
**Report:** `Q1 <file>:<line> -- <dead/parked code / history narration>`

### Q2 -- Fail loudly and specifically
No bare or broad `except` that swallows an error. A failed step is never downgraded to a warning
while continuing with partial results. Raise with context. Library code logs through `logging`,
never `print`.
**Report:** `Q2 <file>:<line> -- <swallowed error / failure-as-warning / print>`

### Q3 -- Keep Python hygiene
Module-level constants go at the top. Each module is imported once, in one fully qualified style. No
`sys.path` edits in library code. `__all__` is explicit and duplicate-free. Public functions have
type hints and a docstring that matches the signature.
**Report:** `Q3 <file>:<line> -- <import/const/__all__/signature hygiene>`

### Q4 -- Vectorize dataframe work
No row-wise loop or `apply` over a DataFrame where a vectorized form exists. No chained assignment.
**Report:** `Q4 <file>:<line> -- row-wise/apply where vectorized form exists`

### Q5 -- Test behavior, not execution
New or changed logic gets tests that assert computed values; a smoke run or golden-file refresh
alone is a finding. Shared fixtures live in the owner from the table. A fixture over 1 MB states its
reason.
**Report:** `Q5 <file>:<line> -- asserts execution, not computed values`

### Q6 -- Keep dependencies minimal
Every runtime dependency is imported by package code; everything else is an optional extra. A new
dependency states its reason and does not duplicate an existing one. Package-data rules and the
manifest agree.
**Report:** `Q6 <file>:<line> -- <unjustified/duplicate dependency / manifest disagreement>`

## D -- Docs

### D1 -- Update docs with user-visible changes
A change to a CLI flag, env var, output field, public API, analyzer, or extension updates its docs
page, README, and CLI help in the same PR. A pure internal refactor is exempt, and the review states
the exemption.
**Report:** `D1 <file>:<line> -- user-visible <change> with no doc/README/CLI-help update`

### D2 -- Write plain technical prose
Match the register of existing docs: no emojis, no conversational or first-person phrasing.
**Report:** `D2 <file>:<line> -- <emoji / conversational / first-person> prose`

### D3 -- Keep docs ROCm-render safe
No mermaid (use a list), no inline boldface between paragraphs, and links that resolve (relative
where needed).
**Report:** `D3 <file>:<line> -- <mermaid / inline boldface / broken link>`

### D4 -- Be complete and precise
Enumerate full sets, list allowed values, and define terms before use. Any CLI a docstring
advertises exists in `entry_points`.
**Report:** `D4 <file>:<line> -- <incomplete set / undefined term / advertised CLI not in entry_points>`

### D5 -- Match the PR description to the diff at head
The title and description describe what the diff does at the current head commit. A later commit
that added a change the description omits, or a narrow title over a broad diff, is a desync.
**Report:** `D5 -- description says "<claim>"; at head the diff does <actual>`
