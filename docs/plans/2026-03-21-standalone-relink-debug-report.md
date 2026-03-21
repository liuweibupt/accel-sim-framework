# Standalone Relink Debug Report

## Context
After the exact barrier-PC mapping fix was implemented, the normal full `make` path was still blocked by unrelated local build failures in:
- `libcuda/cuda_runtime_api.cc`
- `src/gpgpu-sim/gpu-sim.cc`

To keep replay debugging moving, a standalone `accel-sim.out` relink path was attempted using:
- updated worktree `trace-driven` objects
- previously built `gpgpu-sim` objects from the main tree

## First failure: startup segmentation fault
The first relinked binary crashed immediately at startup in:

```text
trace_shader_core_ctx::create_shd_warp()
```

and later, after one partial rebuild, in:

```text
shader_core_ctx::can_issue_1block()
```

## Root cause
This was an ABI mismatch, not a trace bug:
- the main nested `gpgpu-sim` repository has local source changes:
  - `src/abstract_hardware_model.h`
  - `src/gpgpu-sim/shader.cc`
- specifically, `MAX_BARRIERS_PER_CTA` had been changed from `16` to `64`
- but many already-built simulator objects were still from the older layout

That meant the relinked executable mixed:
- new objects compiled against the 64-barrier class layout
- old objects compiled against the 16-barrier class layout

This invalidated object layout assumptions in constructors and member accesses.

## Evidence
- old `shader.o` timestamp was from `2026-03-17`
- rebuilt `abstract_hardware_model.o` was from `2026-03-21`
- rebuilding only `shader.o` moved the segfault from `create_shd_warp()` to `can_issue_1block()`

That strongly confirmed progressive removal of ABI-mismatched objects.

## Standalone relink repair

### 1. Rebuilt root support objects
Recompiled:
- `src/abstract_hardware_model.cc`
- `src/debug.cc`
- `src/gpgpusim_entrypoint.cc`
- `src/option_parser.cc`
- `src/statwrapper.cc`
- `src/stream_manager.cc`
- `src/trace.cc`

### 2. Rebuilt `intersim2`
Built `src/intersim2` in `CREATE_LIBRARY=1` mode so interconnect objects were available for linking.

### 3. Added trace-driven runtime compatibility shim
Added:
- `gpu-simulator/trace-driven/trace_runtime_compat.cc`

This provides weak definitions for the minimal runtime entrypoints needed by the standalone trace-driven binary when the local `libcuda/cuda_runtime_api.cc` path is temporarily unavailable:
- `register_ptx_function`
- `GPGPU_Context`
- `gpgpu_context::GPGPUSim_Init`
- `ptxinfo_data::ptxinfo_addinfo`

The definitions are marked weak so a future successful full `libcuda` build can override them with the canonical strong definitions.

### 4. Rebuilt key mismatched uarch object
Recompiled:
- `src/gpgpu-sim/shader.cc`
- `src/gpgpu-sim/gpu-sim.cc`

For `gpu-sim.cc`, compilation succeeded once the power-model include path / define were supplied, matching how the source expects `gpgpu_sim_wrapper` and `MAX(...)` support.

## Verification

### Smoke replay
The rebuilt standalone binary successfully replayed the small FP16 `512x512x512` trace:

```text
gpu_tot_sim_cycle = 35980
gpu_tot_sim_insn = 6150144
gpu_tot_ipc = 170.9323
GPGPU-Sim: *** exit detected ***
```

### Large replay
A new large replay was launched with:
- reprocessed `2048x12288x12288` FP16 trace
- exact barrier-PC mapping
- `-gpgpu_num_cta_barriers 64`

At launch time it progressed through:
- config load
- kernel header parse
- kernel bind
- initial CTA issue

which is already beyond the earlier startup/link failure modes.

## Current status
- standalone trace-driven binary is usable again
- small smoke replay passes
- large replay is currently running with the new exact barrier-id logic
