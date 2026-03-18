# Large GEMM Barrier Root Cause Report

## Summary
The large FP16 `2048x12288x12288` GEMM trace replay on Accel-Sim failed because trace-driven mode collapsed distinct SM80 barrier phases into a single simulated barrier identity. The failure surfaced as an assertion in CTA barrier teardown:

- file: `gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc`
- function: `barrier_set_t::deallocate_barrier()`
- assertion: `assert(at_barrier.any() == false)`

This means a CTA was being deallocated while one or more warps were still marked as waiting at a barrier.

## Failing symptom
Observed failure during replay:

```text
accel-sim.out: shader.cc:3833: void barrier_set_t::deallocate_barrier(unsigned int): Assertion `at_barrier.any() == false' failed.
```

## Evidence collected

### 1. The large trace contains modern SM80 async-copy synchronization patterns
The large trace includes repeated sequences of:
- `LDGSTS.E.BYPASS.LTC128B.128`
- `LDGDEPBAR`
- `DEPBAR.LE`
- `BAR.SYNC.DEFER_BLOCKING`

This is consistent with a software-pipelined Ampere tensor-core GEMM kernel.

### 2. Trace-driven `OP_BAR` handling was still a placeholder
In `gpu-simulator/trace-driven/trace_driven.cc`, the replay frontend handled every barrier instruction as:

```cpp
bar_id = 0;
bar_count = (unsigned)-1;
bar_type = SYNC;
```

So all barrier phases in the kernel were forced into the same simulated barrier slot.

### 3. Runtime barrier instrumentation confirmed the collapse
A debug rerun with barrier instrumentation showed that many barrier events in the large replay were repeatedly entering the same slot:
- `bar=0`
- `bar_slot=0`
- `count=-1`

This matched the placeholder decoding above and strongly suggested that distinct barrier phases were being merged incorrectly.

### 4. The small 512x512x512 replay succeeded because it is less demanding
The smaller verified GEMM also contains barrier/dependency instructions, but the simpler execution pattern did not trigger the same teardown assertion. The larger GPT-style GEMM exercised many more synchronization phases and exposed the decoding weakness.

## Root cause judgment
The most likely root cause is:

> Trace-driven replay did not preserve barrier phase identity for `OP_BAR`. Distinct `BAR.SYNC.DEFER_BLOCKING` phases in the large SM80 GEMM were collapsed into one barrier id (`0`), corrupting barrier bookkeeping and eventually causing CTA deallocation to observe warps still marked at barrier.

## Fix implemented
A minimal repair was applied in:
- `gpu-simulator/trace-driven/trace_driven.cc`

Instead of forcing all `OP_BAR` instructions to `bar_id = 0`, the replay frontend now derives a stable synthetic barrier id from the barrier instruction PC:

```cpp
bar_id = (trace.m_pc >> 4) % MAX_BARRIERS_PER_CTA;
```

The rest of the semantics remain unchanged for this iteration:
- `bar_count = (unsigned)-1`
- `bar_type = SYNC`

## Why this fix is reasonable
- It is minimal and local.
- It preserves separation between distinct barrier PCs/phases.
- It directly targets the behavior observed in the failing replay.
- It avoids larger speculative changes to barrier semantics until more evidence is needed.

## Validation results so far

### Large replay probe
With the fix applied, a fresh large replay probe showed:
- barrier ids are no longer all zero
- example observed ids: `bar=7`, `bar=10`
- the replay progressed for the probe window without immediately reproducing the original `deallocate_barrier()` assertion

### Small replay sanity check
The verified small FP16 `512x512x512` replay still completed successfully after the fix:
- `gpu_tot_sim_cycle = 35980`
- `gpu_tot_sim_insn = 6150144`
- `gpu_tot_ipc = 170.9323`
- `GPGPU-Sim: *** exit detected ***`

## Remaining caveat
This is still a synthetic mapping. If the kernel uses more distinct barrier PCs than the simulator's barrier-slot budget, collisions are still possible. But the previous catastrophic collapse of all barriers into slot 0 has been removed, and the new behavior is much closer to the intended phase separation.

## Current status
A fresh long-running large replay is being attempted with the repaired binary to determine whether the full `2048x12288x12288` trace now completes or exposes a later-stage issue.
