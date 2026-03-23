# Large Trace Replay Debug Recap

## Scope
This document records the current debugging status for the large A100-like trace replay task in Accel-Sim, focused on the FP16 GEMM:

- shape: `2048 x 12288 x 12288`
- trace source: Modal A100 retrace
- replay input: reprocessed `traceg`
- target config: `SM80_A100`

## Goal
Get the large GEMM trace to replay successfully in Accel-Sim and understand the remaining replay deadlock precisely enough to fix it.

---

## Issues already solved

### 1. Large trace was originally incomplete
An earlier large trace replay failed because the trace itself was incomplete:
- grid expected `1536` CTAs
- actual trace only contained `569` completed thread blocks

That trace was discarded and replaced by a newly generated complete trace.

### 2. `post-traces-processing` corrupted a correct raw trace
The raw `.trace` file was structurally correct, but the generated `.traceg` dropped warp records in some thread blocks.

Observed example:
- `thread block = 53,4,0`
- original `.traceg` missing `warp = 5`
- raw `.trace` still contained the corresponding data

Root cause:
- reused `stringstream` in `post-traces-processing.cpp` without `ss.clear()`

Status:
- fixed
- raw trace reprocessed locally
- regenerated `traceg` validated structurally

### 3. `OP_BAR` in trace-driven replay collapsed all barrier phases
Original trace-driven decoding used:

```cpp
bar_id = 0;
```

This was too coarse for SM80 software-pipelined GEMM kernels and caused distinct barrier phases to alias.

### 4. PC-hash barrier IDs were still insufficient
A first repair changed `bar_id` to a PC-derived hash, but `% MAX_BARRIERS_PER_CTA` still collided for distinct barrier PCs.

That was not enough for the large trace.

### 5. Exact per-kernel barrier identity mapping implemented
The current trace-driven barrier mapping assigns:
- one unique barrier id per unique barrier PC within the kernel
- stable reuse for repeated barrier PCs
- explicit failure if barrier-slot capacity is exceeded

Status:
- implemented
- regression-tested
- committed and pushed

### 6. Empty warp trace lifecycle bug
Replay previously failed on:

```text
trace_driven.cc:82: assert(warp_traces.size() > 0)
```

This was caused by empty trace warps being reused incorrectly.

Status:
- fixed so empty trace warps are completed and exited cleanly

### 7. Barrier cleanup on warp exit
Replay previously exposed residual barrier state when the last active warp exited.

Status:
- fixed so barrier waiters are released correctly when active warps become empty

### 8. Local standalone binary / ABI mismatch
After several fixes, the local replay binary still crashed because:
- some simulator objects were compiled with old layout assumptions
- newer objects were compiled after `MAX_BARRIERS_PER_CTA` had changed from `16` to `64`

This produced mixed-ABI relinking and startup crashes.

Status:
- relevant simulator objects rebuilt
- standalone relink repaired
- small replay restored

### 9. Small replay is healthy again
Verified working smoke replay:
- FP16 `512x512x512`

Observed successful completion:
- `gpu_tot_sim_cycle = 35980`
- `gpu_tot_sim_insn = 6150144`
- `gpu_tot_ipc = 170.9323`
- `GPGPU-Sim: *** exit detected ***`

---

## Current large-trace symptom

The large replay now runs deeply enough to reproduce a stable late-stage deadlock rather than failing immediately.

Stable deadlock signature:

```text
GPGPU-Sim uArch: ERROR ** deadlock detected:
last writeback core 35 @ gpu_sim_cycle 7689340
(+ gpu_tot_sim_cycle 4287217296) (60660 cycles ago)
```

This deadlock point has reproduced consistently.

---

## Important observations about the current deadlock

### 1. CTA 0 barrier cleanup is not the final root cause
At the end of the captured deadlock log:
- CTA 0 barrier phases complete
- `warp_exit_*` completes
- `deallocate_pre` shows a clean CTA teardown

So the remaining deadlock is **not** simply “CTA 0 barrier never released”.

### 2. Core 35 is not the real stuck core
Detailed gdb pipeline dump showed:
- core `35` had `0 threads running`
- pipeline was empty / bubble

So:
- core 35 is only the **last writeback core**
- not the core currently holding forward progress hostage

### 3. Real problematic cores are still active
Example dumps from cores `0` and `2` showed:
- `256 threads running`
- warps still alive
- ibuffers still contain instructions
- pipelines, scoreboards, operand collectors mostly empty
- no obvious memory-response backlog

This means:
- some warps remain alive
- but the front-end / scheduler / issue path no longer makes forward progress

### 4. Suspicious PCs at deadlock
Deadlock-time warp PCs observed repeatedly:
- `0x1e00`
- `0x2280`

Mapped back to trace instructions:

- `0x1e00` -> `IADD3.X`
- `0x1e10` -> `HMMA.16816.F32`
- `0x2280` -> `LDG.E` with **zero active mask**
- `0x2290` -> `LDG.E` with **zero active mask**

This is currently one of the strongest clues.

### 5. Why zero-mask loads matter
The deadlocked warps still appear alive and still have ibuffer entries, but they are not issuing useful work.

The zero-mask `LDG.E` instructions are suspicious because they may interact badly with:
- warp progress bookkeeping
- issue / completion logic
- scoreboard visibility
- front-end “warp still has work” decisions

Current hypothesis:
- the replay is no longer dominated by barrier identity corruption
- instead it is more likely dominated by **warp forward-progress logic** on alive-but-not-advancing warps
- zero-mask memory instructions are a prime candidate

---

## Additional debugging work already performed

### gdb deadlock capture
A scripted gdb replay was used to capture:
- deadlock breakpoint
- backtrace
- selected shader pipeline dumps
- memory partition 0 dump

That debugging established:
- deadlock is real and stable
- core 35 is not the root stuck core
- active cores still have live warps and pending front-end state

### Instrumentation added
Additional deadlock-oriented instrumentation has been added locally in nested `gpgpu-sim` to support the next debugging round:
- deadlock-time selected pipeline dumps
- extended warp-state printing intended to show waiting reasons such as:
  - `wait=barrier`
  - `wait=mem_barrier`
  - `wait=ldgsts`
  - `wait=atomic(...)`

This instrumentation has been compiled into the latest standalone replay binary.

---

## Current best interpretation

The replay chain has already moved past the earlier classes of failures:
- malformed trace structure
- barrier identity collapse
- empty warp lifecycle failure
- ABI/relink breakage

The remaining bug is more likely:

> active warps remain logically alive, with ibuffer-visible instructions, but do not continue issuing or committing, leading to global deadlock.

This looks more like:
- warp/scheduler forward-progress failure
- or incorrect handling of certain trace-driven instruction states

than like:
- a simple final CTA barrier cleanup bug

---

## Current debugging direction

The next debugging target is to determine whether deadlocked active warps are stalled by:

1. zero-mask `LDG.E` handling
2. `LDGSTS / DEPBAR`-related waiting state
3. scheduler visibility / issue eligibility logic
4. another trace-driven front-end lifecycle bug

At this point, the strongest concrete lead is still:
- deadlocked active warps near PCs `0x1e00 / 0x2280`
- especially the zero-mask `LDG.E` instructions

---

## Branch / artifact note
This document is a status recap only.

Relevant implementation work has already been recorded on branch:
- `modal-a100-cutlass-trace`

Additional local nested-repo debug instrumentation exists in:
- `gpu-simulator/gpgpu-sim`

and should be treated carefully when finalizing the eventual root-cause fix.
