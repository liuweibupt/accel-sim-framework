# Large GEMM Barrier Debug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collect decisive evidence for the large-GEMM barrier assertion failure in one rerun, so we can identify the exact broken barrier state and then implement the correct fix.

**Architecture:** Keep the large trace fixed and instrument only the Accel-Sim barrier state machine. Capture barrier state at warp entry, warp exit, and CTA teardown, then rerun the exact same replay with both stdout and stderr logged. Use the resulting log to determine whether the trace-driven `OP_BAR` semantics are too coarse for this SM80 kernel.

**Tech Stack:** C++, Accel-Sim/GPGPU-Sim shader pipeline code, shell logging, Git.

---

### Task 1: Save the design and plan docs

**Files:**
- Create: `docs/plans/2026-03-18-large-gemm-barrier-debug-design.md`
- Create: `docs/plans/2026-03-18-large-gemm-barrier-debug.md`

**Step 1: Write the design document**
- Save the approved debugging design and rationale.

**Step 2: Write the implementation plan**
- Save the task-by-task execution plan.

**Step 3: Verify docs exist**

Run: `ls docs/plans/2026-03-18-large-gemm-barrier-debug*.md`
Expected: both files listed.

### Task 2: Add minimal barrier diagnostics

**Files:**
- Modify: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc`
- Optional context: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h`

**Step 1: Add a compact barrier dump helper**
- Print CTA-local warp state in a compact, grep-friendly format.

**Step 2: Instrument warp barrier entry**
- In `barrier_set_t::warp_reaches_barrier`, print `cta_id`, `warp_id`, `bar_id`, `bar_count`, and the CTA-local active/barrier masks before and after updates.

**Step 3: Instrument warp exit**
- In `barrier_set_t::warp_exit`, print the CTA-local active/barrier masks before and after release logic.

**Step 4: Instrument CTA deallocation**
- In `barrier_set_t::deallocate_barrier`, print the CTA-local state immediately before the failing assertions.

**Step 5: Keep output bounded**
- Only print for barrier-related events, not every instruction.

### Task 3: Rebuild and rerun the exact failing replay

**Files:**
- Reuse trace: `modal/artifacts/fp16-2048x12288x12288-gemm3/raw_traces/kernelslist.g`
- Write logs under: `modal/artifacts/fp16-2048x12288x12288-gemm3/debug_barrier_run/`

**Step 1: Rebuild Accel-Sim**

Run the repository’s normal build command for the simulator binary.
Expected: rebuilt `accel-sim.out` succeeds.

**Step 2: Create a dedicated debug run directory**
- Keep the original logs untouched.

**Step 3: Rerun with combined stdout/stderr logging**

Run: exact replay command with `> debug.log 2>&1`
Expected: either the same assertion reproduces with extra context or the replay unexpectedly completes.

**Step 4: Verify the failure signature**
- Confirm whether the assertion is the same.
- Extract the final barrier-state dump around the crash.

### Task 4: Analyze the captured evidence

**Files:**
- Read: `modal/artifacts/fp16-2048x12288x12288-gemm3/debug_barrier_run/debug.log`
- Read: `gpu-simulator/trace-driven/trace_driven.cc`

**Step 1: Identify the failing CTA and warp set**
- Determine which CTA is deallocating with residual `m_warp_at_barrier` bits.

**Step 2: Classify the failure mode**
- Decide whether the stuck state came from:
  - incorrect barrier entry semantics,
  - failed barrier release,
  - incorrect warp-exit cleanup,
  - or a mismatch between `OP_BAR` trace decoding and actual hardware behavior.

**Step 3: Write a concise root-cause summary**
- Save the evidence and likely fix direction in the final response.

### Task 5: Commit and push the diagnostic instrumentation

**Files:**
- Modify: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc`
- Create: `docs/plans/2026-03-18-large-gemm-barrier-debug-design.md`
- Create: `docs/plans/2026-03-18-large-gemm-barrier-debug.md`

**Step 1: Check git status**

Run: `git status --short`
Expected: only intended debug files changed, plus untracked large artifact logs if any.

**Step 2: Commit code/docs only**

```bash
git add gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc \
        docs/plans/2026-03-18-large-gemm-barrier-debug-design.md \
        docs/plans/2026-03-18-large-gemm-barrier-debug.md
git commit -m "debug: instrument large GEMM barrier failure"
```

**Step 3: Push**

```bash
git push
```
