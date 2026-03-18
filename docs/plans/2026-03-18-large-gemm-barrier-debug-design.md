# Large GEMM Barrier Debug Design

## Goal
Diagnose the Accel-Sim replay crash for the large FP16 `2048x12288x12288` GEMM trace by collecting the minimum additional evidence needed to identify the exact CTA/warp/barrier state that triggers the assertion in `barrier_set_t::deallocate_barrier()`.

## Problem Summary
The replay no longer runs. It aborts in:
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc:3833`

The failing assertion is:
- `assert(at_barrier.any() == false)`

This means a CTA is being deallocated while Accel-Sim still believes one or more warps in that CTA are waiting at a barrier.

## Constraints
- Re-running the large replay is expensive in wall-clock time, so the first new run must maximize diagnostic value.
- We should avoid speculative fixes before collecting evidence.
- We should not modify the trace-generation pipeline or large trace artifacts.
- We should keep all changes local to Accel-Sim debugging support unless root cause is proven.

## Current Leading Hypothesis
The most likely root cause is incomplete barrier semantics in trace-driven mode for SM80 async-copy style kernels.

Evidence supporting this hypothesis:
- The large trace contains repeated `LDGSTS`, `LDGDEPBAR`, `DEPBAR.LE`, and `BAR.SYNC.DEFER_BLOCKING` sequences.
- The replay fails specifically in CTA barrier bookkeeping.
- `gpu-simulator/trace-driven/trace_driven.cc` still handles `OP_BAR` with a placeholder implementation:
  - `bar_id = 0`
  - `bar_count = (unsigned)-1`
  - `bar_type = SYNC`
- That placeholder may be good enough for smaller kernels but wrong for this larger pipelined GEMM.

## Approaches Considered

### Option 1: Minimal diagnostic instrumentation + one rerun (recommended)
Add targeted logging around barrier entry, warp exit, and CTA deallocation so the next replay tells us exactly which CTA/warp set is inconsistent.

Pros:
- Highest probability of yielding actionable root cause in one rerun.
- Keeps code changes small and reversible.
- Avoids speculative semantic changes.

Cons:
- Requires another long replay.

### Option 2: Static-code-only reasoning
Do not rerun; infer likely cause from trace and source inspection.

Pros:
- Fastest.

Cons:
- Likely stops at a plausible theory rather than a proven root cause.
- Insufficient for a reliable fix.

### Option 3: Immediate semantic fix to `OP_BAR`
Patch trace-driven barrier decoding and rerun.

Pros:
- Could solve the problem directly.

Cons:
- High risk of blind fixing.
- Could introduce incorrect semantics without proving the real failure mode.

## Recommended Design
Use Option 1.

Add diagnostic logging only at the barrier state transitions that matter:
- `barrier_set_t::warp_reaches_barrier`
- `barrier_set_t::warp_exit`
- `barrier_set_t::deallocate_barrier`
- optionally `shader_core_ctx::register_cta_thread_exit`

The log must print:
- `cta_id`
- `warp_id`
- `bar_id`
- `bar_count`
- CTA warp mask
- active warp mask
- `m_warp_at_barrier`
- per-barrier warp masks for that CTA

Also rerun with both stdout and stderr captured to a dedicated debug log so assertion output is preserved.

## Success Criteria
The next debug replay should give enough evidence to answer all of these:
1. Which CTA is being deallocated?
2. Which warp(s) are still marked at barrier?
3. Which barrier ID is involved?
4. Did the warp reach barrier without being released, or did warp exit bookkeeping fail to clear it?
5. Is the failure consistent with incorrect trace-driven `OP_BAR` semantics?

## Likely Next Step After Evidence Collection
If the diagnostic run confirms that barrier state is being tracked with the wrong semantics, the next change will likely be to correct `OP_BAR` handling in trace-driven mode or special-case the relevant SM80 barrier sequence more accurately.
