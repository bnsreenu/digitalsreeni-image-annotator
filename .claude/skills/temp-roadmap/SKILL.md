---
name: temp-roadmap
description: Create a temporary, self-removing roadmap/backlog section in CLAUDE.md from a set of identified work items (issue triage, audit findings, review feedback). Use when the user asks to "track these items", "write this list into CLAUDE.md as a plan", "create a backlog/roadmap in CLAUDE.md", or wants a work list that cleans itself up as PRs land.
---

# Temporary Self-Removing Roadmap in CLAUDE.md

Turn a set of identified work items into a CLAUDE.md section that shrinks with
every PR and disappears entirely when the work is done — so CLAUDE.md returns
to its clean state without manual housekeeping.

## When to use

- After issue triage, a code audit, or a review produced a concrete list of
  work items that will be resolved across several future PRs.
- NOT for single-PR task tracking (use the todo list) and NOT for permanent
  guidance (write a normal CLAUDE.md section or arc42 doc instead).

## Structure to write into CLAUDE.md

Insert a section near the top of CLAUDE.md (after the docs index, before
project structure):

```markdown
## <Topic> Backlog (TEMPORARY section — self-deleting)

**Deletion hook:** When a PR resolves one of the items below, DELETE its row
from this table **in the same PR**. When the last row is gone, delete this
entire section so CLAUDE.md returns to its clean state. Never let a finished
item linger here.

Items validated <YYYY-MM-DD>; source: <issue tracker / audit / review link>.

| Item | Size | Task |
|------|------|------|
| #123 | quick win | One-line actionable description with file hints (`path/file.py`, function name, reporter-verified fix if any) |
| #124 | medium | ... |
| #125 | blocked | ... — name what it is blocked on so it can be re-checked |
```

## Rules

1. **One line per item.** Concise but self-sufficient: a future session must be
   able to start the item from the row alone — include file paths, function
   names, and known pitfalls ("do NOT use setSortingEnabled — currentRowChanged
   fires switch_image").
2. **Classify every item**: `quick win` / `medium` / `large` / `blocked`.
   For `blocked`, state the blocker.
3. **Reference the source** (issue number, ticket, review comment) so context
   can be recovered.
4. **Date the validation** — rows describe code state at a point in time;
   re-verify before implementing if the date is old.
5. **The deletion hook is mandatory** and must appear verbatim-in-spirit at the
   top of the section. It is the whole point: the section is a consumable, not
   documentation.
6. Mark items currently being worked on with *(in progress on this branch)* so
   parallel sessions don't double-pick them.

## Per-PR workflow (executing the hook)

1. Pick item(s), implement on a feature branch.
2. In the same PR: delete the resolved row(s) from the table.
3. If the table is now empty: delete the entire section, including the heading
   and the deletion-hook paragraph.
4. The PR diff thus always shows both the fix and the backlog shrinking —
   reviewers can see progress without a separate tracker.
