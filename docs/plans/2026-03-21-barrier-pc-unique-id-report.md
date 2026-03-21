# Barrier-PC Unique-ID Fix Report

## What changed
To address the latest large-GEMM replay deadlock investigation, the trace-driven `OP_BAR` path was changed from a hashed barrier-id scheme to an exact per-kernel barrier-PC mapping.

Files added/updated:
- `gpu-simulator/trace-driven/barrier_id_map.h`
- `gpu-simulator/trace-parser/trace_parser.h`
- `gpu-simulator/trace-driven/trace_driven.cc`
- `gpu-simulator/trace-driven/tests/test_barrier_id_map.cpp`

## Root-cause evidence
The previous mapping used:

```cpp
bar_id = (trace.m_pc >> 4) % MAX_BARRIERS_PER_CTA;
```

For the large FP16 `2048x12288x12288` GEMM trace, that was still lossy:
- unique `BAR.SYNC` PCs observed: `22`
- unique ids after `% 64`: `18`
- confirmed collisions:
  - `0x1df0` vs `0x71f0`
  - `0x5a30` vs `0x6e30`
  - `0x2270` vs `0x6a70`
  - `0x5b30` vs `0x5f30`

This meant distinct SM80 software-pipeline barrier phases could still alias in replay.

## New behavior
The new helper assigns barrier ids in first-seen order per kernel:
- same barrier PC -> same id
- different barrier PCs -> different ids
- if a kernel exceeds the configured barrier-slot budget, the code now fails explicitly instead of silently aliasing

## Verification performed

### 1. TDD regression test
Added:
- `gpu-simulator/trace-driven/tests/test_barrier_id_map.cpp`

Sequence:
1. test was compiled before implementation
2. expected failure occurred because `barrier_id_map.h` did not exist
3. helper was implemented
4. test compiled and passed

Pass condition covered:
- stable reuse for repeated PC
- distinct ids for known collision pairs
- overflow throws instead of aliasing

### 2. Trace-driven compile
Verified:
- `make trace-driven` succeeds for the updated code path

## Current blocker
A full `make` of `accel-sim.out` is currently blocked by pre-existing build issues in the main `gpgpu-sim` tree used for linking on this machine, including errors in:
- `libcuda/cuda_runtime_api.cc`
- `src/gpgpu-sim/gpu-sim.cc`

These errors are unrelated to the new barrier-id mapping and appeared during the downstream full build stage after the trace-driven object itself compiled successfully.

## Recommended next step
Once the main-tree full-build environment is back to a good state, rerun the large replay with:
- the reprocessed trace
- the current barrier fixes
- the new per-kernel barrier-PC mapping

That is the next high-value validation for whether the remaining deadlock was primarily caused by barrier-id aliasing.
