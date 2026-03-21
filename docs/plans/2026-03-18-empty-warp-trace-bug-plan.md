# Empty Warp Trace Bug Follow-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Record the successful barrier-fix progress, then diagnose and fix the new `trace_driven.cc:82` empty-warp-trace assertion exposed by the next replay stage.

**Architecture:** Add one new progress report documenting the transition from the barrier bug to the new empty-warp-trace bug, then instrument and repair the trace-driven warp lifecycle so replay no longer accesses an empty `warp_traces` queue during execution.

**Tech Stack:** Markdown, C++, Accel-Sim trace-driven frontend/runtime, shell replay workflow, Git.

---

### Task 1: Record the progress handoff

**Files:**
- Create: `docs/plans/2026-03-18-empty-warp-trace-root-cause-report.md`

**Step 1: Write the phase-2 report**
- Capture that the barrier bug was partially fixed.
- Record that the new terminal failure is `trace_driven.cc:82` with `warp_traces.size() > 0`.
- Explain that the replay now progresses past the original barrier teardown issue.

**Step 2: Verify the doc exists**

Run: `ls docs/plans/2026-03-18-empty-warp-trace-root-cause-report.md`
Expected: file is listed.

### Task 2: Diagnose the empty-warp-trace assertion

**Files:**
- Read/modify as needed:
  - `gpu-simulator/trace-driven/trace_driven.cc`
  - `gpu-simulator/trace-driven/trace_driven.h`
  - `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc` (only if runtime context is needed)

**Step 1: Inspect `trace_shd_warp_t::get_start_trace_pc()` and its callers**
- Determine when the simulator expects a non-empty `warp_traces` vector.

**Step 2: Trace warp lifecycle assumptions**
- Identify whether a warp is being queried after its traces are exhausted/cleared.
- Determine whether CTA teardown / relaunch / scheduler state is reusing a trace warp object incorrectly.

**Step 3: Add minimal diagnostics if needed**
- Log warp id / CTA / trace queue size only where necessary.

### Task 3: Implement the smallest safe fix

**Files:**
- Modify only the files required by the diagnosed root cause.

**Step 1: Patch the empty-warp-trace bug**
- Ensure replay never dereferences a warp trace start-PC from an empty trace queue.
- Prefer preserving lifecycle correctness over masking the assert blindly.

**Step 2: Rebuild Accel-Sim**

Run the normal rebuild command.
Expected: `accel-sim.out` rebuild succeeds.

### Task 4: Rerun and validate

**Files:**
- Reuse the same large trace and `fix_full_run2`/new run directory.

**Step 1: Rerun the large replay**
- Capture stdout/stderr together.

**Step 2: Check terminal state**
- If it succeeds, collect final metrics.
- If it fails, capture the new failure as the next debugging target.

**Step 3: Run one small-case sanity replay if the code path changed broadly**
- Ensure no regression for the verified small case.

### Task 5: Commit and push the new fix/report

**Files:**
- Add the new report doc.
- Add the minimal code fix.

**Step 1: Check git status**

Run: `git status --short`
Expected: intended code/docs changes only, excluding large runtime artifacts.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-18-empty-warp-trace-root-cause-report.md <fixed-files>
git commit -m "fix: handle empty trace warp lifecycle"
```

**Step 3: Push**

```bash
git push
```
