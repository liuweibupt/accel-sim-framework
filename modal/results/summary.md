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

## A100-LPDDR5X experiment

A new config variant was added as `A100_LPDDR5X` with the compute side held constant and only the memory-side bandwidth model changed to approximately 819.2 GB/s using:
- `-gpgpu_n_mem 12`
- `-gpgpu_dram_buswidth 8`
- `-dram_data_command_freq_ratio 2`
- `-gpgpu_clock_domains 1410:1410:1410:4266.667`

This corresponds to a 12-channel, 768-bit LPDDR5X-style bandwidth model.

### Baseline vs LPDDR5X replay comparison (1410 MHz core clock)

| Case | Baseline cycles | LPDDR5X cycles | Baseline time (us) | LPDDR5X time (us) | Slowdown |
|---|---:|---:|---:|---:|---:|
| FP16 / FP32 acc / 512x512x512 | 35980 | 35936 | 25.518 | 25.487 | 0.999x |
| BF16 / FP32 acc / 512x512x512 | 35895 | 36265 | 25.457 | 25.720 | 1.010x |

### Interpretation
For this CUTLASS tensor-core GEMM at `512x512x512`, reducing modeled peak DRAM bandwidth from ~1935 GB/s to ~819.2 GB/s caused only negligible timing change in Accel-Sim. That strongly suggests this kernel is compute-dominated / tensor-core-dominated at this problem size rather than bandwidth-limited by off-chip memory.
