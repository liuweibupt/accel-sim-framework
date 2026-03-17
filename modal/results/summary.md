# Modal A100 CUTLASS Trace Run Summary

## Scope
This summary captures the first successful end-to-end runs of the Modal-based A100 trace workflow using CUTLASS GEMM kernels and local Accel-Sim replay with the repository's `SM80_A100` configuration.

## A100 target selection
- Modal GPU requested: `A100-80GB`
- Accel-Sim replay config:
  - `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config`
  - `gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config`

The repository does not explicitly label its A100 config as 40GB or 80GB, but this workflow intentionally used Modal's `A100-80GB` to align with the earlier config-based inference.

## Successful runs

### 1. FP16 input, FP32 accumulate, 512x512x512
- Modal job name: `fp16-512x512x512`
- Trace directory: `modal/artifacts/fp16-512x512x512/traces/`
- Replay log: `modal/artifacts/fp16-512x512x512/sim_run/sim.out`
- Trace files include:
  - `kernelslist.g`
  - `kernel-1-ctx_0x56201b8d3c00.trace`
  - `kernel-1-ctx_0x56201b8d3c00.traceg`
- Replay evidence:
  - `binary version = 80`
  - `gpu_tot_sim_cycle = 35980`
  - `gpu_tot_sim_insn = 6150144`
  - `gpu_tot_ipc = 170.9323`
  - `GPGPU-Sim: *** exit detected ***`

### 2. BF16 input, FP32 accumulate, 512x512x512
- Modal job name: `bf16-512x512x512`
- Trace directory: `modal/artifacts/bf16-512x512x512/traces/`
- Replay log: `modal/artifacts/bf16-512x512x512/sim_run/sim.out`
- Trace files include:
  - `kernelslist.g`
  - `kernel-1-ctx_0x56431e79dc80.traceg`
- Replay evidence:
  - `binary version = 80`
  - `gpu_tot_sim_cycle = 35895`
  - `gpu_tot_sim_insn = 6150144`
  - `gpu_tot_ipc = 171.3371`
  - `GPGPU-Sim: *** exit detected ***`

## Notes
- Both runs exercised real Modal A100 hardware and produced Accel-Sim-compatible traces.
- The small `512x512x512` kernels were chosen first to keep cost and turnaround low.
- Initial attempts to expand directly to GPT-3-style `Mx12288x12288` shapes were not finalized in this round; the current preserved, verified outputs are the two successful 512-cube runs above.
- The remote CUTLASS runner printed unstable checksums (`nan`/`inf`) in some cases, but trace generation and Accel-Sim replay completed successfully. Functional/numerical validation of the CUTLASS runner remains a separate follow-up concern from trace pipeline validation.
