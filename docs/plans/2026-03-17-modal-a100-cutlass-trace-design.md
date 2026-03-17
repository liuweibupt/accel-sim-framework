# Modal A100 CUTLASS Trace Design

**Date:** 2026-03-17
**Status:** Approved

## Goal
Build a cost-conscious workflow under `modal/` that runs CUTLASS GEMM kernels on Modal A100, captures Accel-Sim-compatible traces, and replays them locally with the repository's `SM80_A100` configuration.

## Constraints
- Prefer true A100 hardware over proxy traces from local Ada hardware.
- Minimize GPU billable runtime; target an initial end-to-end GPU execution window of about 2 minutes or less.
- Support both:
  - FP16 inputs with FP32 accumulation
  - BF16 inputs with FP32 accumulation
- Include both:
  - small sanity GEMM: `512 x 512 x 512`
  - GPT-3-style GEMM family with `N = 12288`, `K = 12288`, and small `M` sweeps.
- Produce trace outputs that can be consumed by local Accel-Sim without manual format surgery.

## A100 Version Alignment
### Modal side
Modal's GPU guide documents explicit selectors for `A100-40GB` and `A100-80GB`, and notes that generic `A100` may be upgraded automatically. For reproducibility we will request `A100-80GB` explicitly.

### Accel-Sim side
This repository exposes one SM80 A100 configuration:
- `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config`
- `gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config`

The repository does not explicitly label this configuration as 40GB or 80GB. Based on the SM80 A100 memory partitioning and clock parameters, we infer that it is closer to an 80GB-class A100 profile. This is an inference from config values, not an explicit statement from the repository.

## Considered Approaches
### Approach A: Modal A100 + custom minimal CUTLASS driver + direct NVBit tracing
Clone CUTLASS into `modal/`, build a tiny driver that launches only the target GEMMs, run that driver on Modal A100-80GB, and attach the existing NVBit tracer tool to generate traces.

**Pros**
- Best control over which kernels execute.
- Cheapest path to clean traces.
- Best match to the user's goal and to Accel-Sim replay needs.

**Cons**
- Requires custom runner code and container setup.

### Approach B: Modal A100 + CUTLASS profiler
Use `cutlass_profiler` to invoke GEMMs directly on Modal and trace that process.

**Pros**
- Less custom C++ code.

**Cons**
- More extra kernels and setup noise in traces.
- Harder to keep runtime and trace volume low.

### Approach C: cuBLASLt-based GEMM on Modal
Use a library call instead of CUTLASS.

**Pros**
- Small user code surface.

**Cons**
- Less control over kernel mix and selection.
- Less aligned with the explicit request to run CUTLASS.

## Selected Approach
Use **Approach A**.

## High-Level Architecture
1. Create a new `modal/` workspace in this repository.
2. Vendor or clone CUTLASS under `modal/extern/`.
3. Build a tiny GEMM runner binary with two data-type families:
   - FP16 -> FP32 accumulate
   - BF16 -> FP32 accumulate
4. Build a Modal app that:
   - boots an A100-80GB container,
   - compiles the GEMM runner and the existing NVBit tracer,
   - executes one requested GEMM configuration,
   - runs post-processing to produce `kernelslist.g` and `.traceg` outputs,
   - stores outputs in a deterministic artifact directory.
5. Pull artifacts back locally and replay with local `accel-sim.out` using the repository's A100 config files.

## Data Flow
1. User invokes a local Modal launcher with dtype and matrix shape.
2. Modal job starts on `A100-80GB`.
3. Container compiles CUTLASS target and tracer if needed.
4. NVBit traces only the target process.
5. Trace post-processing emits:
   - `kernelslist.g`
   - `kernel-*.traceg[.xz]`
   - stats metadata
6. Artifacts are downloaded locally under `modal/artifacts/...`.
7. Local helper script replays those traces with `SM80_A100` configs and archives simulator logs.

## Runtime / Cost Control Strategy
- Run only one GEMM shape per Modal invocation.
- Default to no benchmark sweep, no warmup loop, and deterministic fixed launch counts.
- Start with the smallest useful set:
  - `fp16-fp32acc 512x512x512`
  - `bf16-fp32acc 512x512x512`
  - `fp16-fp32acc 512x12288x12288`
  - `bf16-fp32acc 512x12288x12288`
- Add larger GPT-3-style `M` values only after timing evidence shows sub-2-minute runs.
- Prefer raw or lightly compressed traces only if compression is reliable; avoid xz corruption paths discovered locally.

## Error Handling
- Fail fast if Modal authentication is unavailable.
- Fail fast if CUTLASS BF16 kernels cannot compile for the requested CUDA version.
- Record exact container stdout/stderr and preserve partial artifacts for debugging.
- If BF16 support is unavailable for the chosen kernel path, keep FP16 working and report BF16 as a bounded blocker.

## Verification Strategy
Successful first milestone means all of the following are true:
1. Modal A100-80GB job completes.
2. CUTLASS runner finishes for one selected GEMM.
3. Tracer emits valid `kernelslist.g` and traceg files.
4. Local Accel-Sim replay completes with `SM80_A100` configs.
5. Replay log contains the expected `binary version = 80` and simulator exit markers.

## Expected Repository Additions
- `modal/README.md`
- `modal/app.py`
- `modal/requirements.txt` or equivalent Python dependency pinning
- `modal/cutlass_runner/` source files
- `modal/scripts/` for artifact management and local replay
- `docs/plans/2026-03-17-modal-a100-cutlass-trace.md`

