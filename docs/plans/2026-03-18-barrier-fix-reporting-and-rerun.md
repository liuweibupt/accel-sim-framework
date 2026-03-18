# Barrier Fix Reporting and Large Replay Rerun Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Record the barrier root-cause analysis and fix in repository documentation, then continue the large replay with the repaired trace-driven barrier mapping.

**Architecture:** Update the existing user-facing summary plus create a dedicated debug/fix report under `docs/plans/`, then launch a fresh long-running replay using the repaired simulator binary and monitor it for completion or a new failure mode.

**Tech Stack:** Markdown, shell replay workflow, Git.

---

### Task 1: Save the reporting plan

**Files:**
- Create: `docs/plans/2026-03-18-barrier-fix-reporting-and-rerun.md`

**Step 1: Write the plan document**
- Save the implementation steps for reporting + rerun.

**Step 2: Verify it exists**

Run: `ls docs/plans/2026-03-18-barrier-fix-reporting-and-rerun.md`
Expected: file is listed.

### Task 2: Document the debug findings and fix

**Files:**
- Modify: `modal/results/summary.md`
- Create: `docs/plans/2026-03-18-barrier-root-cause-report.md`

**Step 1: Write a dedicated root-cause report**
- Capture the failing assertion, the evidence from the barrier diagnostics, the `OP_BAR` placeholder behavior, and the implemented PC-derived barrier-id fix.

**Step 2: Update the user-facing summary**
- Add a short section describing:
  - the root cause hypothesis now supported by evidence,
  - the implemented fix,
  - the fact that a fresh long replay is being attempted with the repaired binary.

### Task 3: Launch the repaired large replay

**Files:**
- Reuse trace: `modal/artifacts/fp16-2048x12288x12288-gemm3/raw_traces/kernelslist.g`
- Write log: `modal/artifacts/fp16-2048x12288x12288-gemm3/fix_full_run/sim.out`

**Step 1: Start a fresh long replay**
- Use the repaired simulator binary.
- Capture stdout and stderr together.

**Step 2: Record the launch status**
- Save PID/session details or otherwise prove the replay is active.

### Task 4: Verify and commit docs/fix metadata

**Files:**
- Verify: `docs/plans/2026-03-18-barrier-root-cause-report.md`
- Verify: `modal/results/summary.md`

**Step 1: Inspect updated docs**

Run: `sed -n '1,240p' docs/plans/2026-03-18-barrier-root-cause-report.md && echo '---' && grep -n 'barrier\|OP_BAR\|2048x12288x12288' modal/results/summary.md`
Expected: docs clearly describe the root cause and fix.

**Step 2: Check git status**

Run: `git status --short`
Expected: intended code/docs changes only, excluding large runtime artifacts.

**Step 3: Commit**

```bash
git add docs/plans/2026-03-18-barrier-fix-reporting-and-rerun.md \
        docs/plans/2026-03-18-barrier-root-cause-report.md \
        modal/results/summary.md
git commit -m "docs: record barrier root cause and fix"
```

**Step 4: Push**

```bash
git push
```
