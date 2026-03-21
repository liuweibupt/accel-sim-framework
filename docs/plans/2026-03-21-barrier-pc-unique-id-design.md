# Large GEMM Barrier-PC Unique-ID Fix Design

## Goal
Fix the current large A100 GEMM replay deadlock by replacing the lossy `OP_BAR` hash mapping with a per-kernel one-to-one mapping from barrier PC to simulator barrier id.

## Problem Summary
The current replay no longer fails in the earlier malformed-trace and empty-warp-trace paths, but it still deadlocks on the large `2048x12288x12288` FP16 GEMM replay.

The strongest new evidence is:
- `OP_BAR` is currently mapped with `bar_id = (trace.m_pc >> 4) % MAX_BARRIERS_PER_CTA`
- this kernel has `22` unique `BAR.SYNC` PCs
- after `% 64`, only `18` unique barrier ids remain
- real collisions exist, for example:
  - `0x1df0` and `0x71f0`
  - `0x5a30` and `0x6e30`
  - `0x2270` and `0x6a70`
  - `0x5b30` and `0x5f30`

So the current fix is still collapsing distinct SM80 software-pipeline barrier phases.

## Approach Options

### Option 1: Increase `MAX_BARRIERS_PER_CTA` again
Pros:
- Very small change.

Cons:
- Does not remove hash collisions in principle.
- Just postpones the same bug.

### Option 2: Keep hash mapping but use a stronger hash
Pros:
- Still local.

Cons:
- Still probabilistic.
- A replay bug caused by identity aliasing should not depend on hash luck.

### Option 3: Assign a unique barrier id per barrier PC within each kernel (recommended)
Pros:
- Removes collisions entirely as long as the kernel uses no more than the configured barrier-slot budget.
- Preserves barrier phase identity instead of approximating it.
- Easy to validate with a focused regression test.

Cons:
- Slightly more stateful than the current one-line hash.
- Needs an explicit overflow failure if a kernel exceeds the barrier-slot budget.

## Recommended Design
Use Option 3.

Implementation shape:
- add a small reusable barrier-id mapper helper in the trace-driven frontend
- keep a per-kernel map from `BAR` PC to assigned barrier id
- assign ids in first-seen order: `0, 1, 2, ...`
- reuse the same id whenever the same PC appears again
- fail loudly if unique barrier PCs exceed `MAX_BARRIERS_PER_CTA`

This preserves semantic separation between the kernel’s distinct `BAR.SYNC` phases without relying on `%`.

## Validation Plan
1. Add a focused regression test covering the collision PCs already observed in the large GEMM trace.
2. Verify the test fails before the new helper exists.
3. Implement the helper and wire `OP_BAR` to it.
4. Re-run the focused regression test and confirm unique IDs are assigned.
5. Rebuild Accel-Sim.
6. If compilation succeeds, continue with replay verification in the next iteration.

## Success Criteria
- The known collision PC pairs now receive different barrier ids.
- Repeated occurrences of the same PC reuse the same id.
- Overflow beyond the configured barrier-slot budget fails explicitly instead of silently aliasing.
- The code path is documented in repo docs for later replay debugging.
