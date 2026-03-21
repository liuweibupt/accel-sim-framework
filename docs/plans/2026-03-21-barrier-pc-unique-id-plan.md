# Barrier-PC Unique-ID Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current hashed `OP_BAR` mapping with a per-kernel barrier-PC unique-id mapping so large SM80 GEMM replay no longer aliases distinct barrier phases.

**Architecture:** Introduce a tiny barrier-id mapper helper with deterministic first-seen assignment and explicit overflow checking. Store the mapper in per-kernel trace metadata and use it from `trace_driven.cc` when decoding `OP_BAR`.

**Tech Stack:** C++, Accel-Sim trace-driven frontend, shell build workflow, Git.

---

### Task 1: Save the approved design and execution plan

**Files:**
- Create: `docs/plans/2026-03-21-barrier-pc-unique-id-design.md`
- Create: `docs/plans/2026-03-21-barrier-pc-unique-id-plan.md`

**Step 1: Write the design doc**
- Capture the observed `% 64` collision evidence and the chosen exact-mapping fix.

**Step 2: Write the implementation plan**
- Capture the files, regression test, integration points, and validation workflow.

**Step 3: Verify the docs exist**

Run: `ls docs/plans/2026-03-21-barrier-pc-unique-id-*.md`
Expected: both files listed.

### Task 2: Add the failing regression test first

**Files:**
- Create: `gpu-simulator/trace-driven/tests/test_barrier_id_map.cpp`

**Step 1: Write a test that expects exact PC identity preservation**
- Use the known collision pairs from the large GEMM trace.
- Assert:
  - same PC -> same id
  - different collision PCs -> different ids
  - too many unique PCs -> explicit failure

**Step 2: Compile the test before implementation**

Run: a direct `g++` compile command for the test.
Expected: fail because the new barrier-id mapping API does not exist yet.

### Task 3: Implement the minimal barrier-id mapper

**Files:**
- Create: `gpu-simulator/trace-driven/barrier_id_map.h`
- Modify: `gpu-simulator/trace-parser/trace_parser.h`
- Modify: `gpu-simulator/trace-driven/trace_driven.cc`

**Step 1: Add a small helper**
- Provide `get_or_assign(pc, max_slots)` with deterministic first-seen assignment.

**Step 2: Store mapper state per kernel**
- Extend kernel trace metadata with one mapper instance so IDs stay stable within a kernel replay.

**Step 3: Replace `% MAX_BARRIERS_PER_CTA`**
- Use the per-kernel mapper in `case OP_BAR`.

### Task 4: Verify the new behavior

**Files:**
- Reuse: `gpu-simulator/trace-driven/tests/test_barrier_id_map.cpp`

**Step 1: Recompile the regression test**
- Expected: compile succeeds.

**Step 2: Run the regression test**
- Expected: pass with the real collision-PC pairs.

**Step 3: Rebuild the trace-driven binary**
- Expected: `accel-sim.out` rebuild succeeds.

### Task 5: Record, commit, and push

**Files:**
- Create: `docs/plans/2026-03-21-barrier-pc-unique-id-design.md`
- Create: `docs/plans/2026-03-21-barrier-pc-unique-id-plan.md`
- Create: `gpu-simulator/trace-driven/barrier_id_map.h`
- Create: `gpu-simulator/trace-driven/tests/test_barrier_id_map.cpp`
- Modify: `gpu-simulator/trace-parser/trace_parser.h`
- Modify: `gpu-simulator/trace-driven/trace_driven.cc`

**Step 1: Check git status**

Run: `git status --short`
Expected: only the intended code/docs/test files are staged.

**Step 2: Commit**

```bash
git add docs/plans/2026-03-21-barrier-pc-unique-id-design.md \
        docs/plans/2026-03-21-barrier-pc-unique-id-plan.md \
        gpu-simulator/trace-driven/barrier_id_map.h \
        gpu-simulator/trace-driven/tests/test_barrier_id_map.cpp \
        gpu-simulator/trace-parser/trace_parser.h \
        gpu-simulator/trace-driven/trace_driven.cc
git commit -m "fix: assign unique barrier ids per trace pc"
```

**Step 3: Push**

```bash
git push origin modal-a100-cutlass-trace
```
