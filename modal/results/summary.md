# Modal A100 CUTLASS Trace Run Summary

## Scope
This summary captures the verified end-to-end Modal A100 trace workflow runs for small tensor-core GEMMs, plus the completed large GPT-3-style GEMM replay comparison between the baseline A100 memory system and the LPDDR5X-style memory variant.

## A100 target selection
- Modal GPU requested: `A100-80GB`
- Observed Modal device: `NVIDIA A100 80GB PCIe, 81920 MiB`
- Accel-Sim replay config:
  - `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config`
  - `gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config`

The repository does not explicitly label its A100 config as 40GB or 80GB, but this workflow intentionally used Modal's `A100-80GB` and confirmed the runtime device as A100 80GB PCIe.

## Verified small-kernel runs

### 1. FP16 input, FP32 accumulate, 512x512x512
- Modal job name: `fp16-512x512x512`
- Trace directory: `modal/artifacts/fp16-512x512x512/traces/`
- Replay log: `modal/artifacts/fp16-512x512x512/sim_run/sim.out`
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
- Replay evidence:
  - `binary version = 80`
  - `gpu_tot_sim_cycle = 35895`
  - `gpu_tot_sim_insn = 6150144`
  - `gpu_tot_ipc = 171.3371`
  - `GPGPU-Sim: *** exit detected ***`

## Tensor Core evidence
The two verified `512x512x512` traces both exercised tensor-core MMA instructions:
- FP16 trace: `65536 HMMA`
- BF16 trace: `65536 HMMA`

## Saved small GEMM results table

| dtype | accumulate | shape | config | modeled BW (GB/s) | cycles | sim insn | IPC | time (us @ 1410 MHz) | tensor-core evidence | trace | replay log |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| FP16 | FP32 | 512x512x512 | SM80_A100 | 1935.36 | 35980 | 6150144 | 170.9323 | 25.518 | HMMA x65536 | `modal/artifacts/fp16-512x512x512/traces/` | `modal/artifacts/fp16-512x512x512/sim_run/sim.out` |
| BF16 | FP32 | 512x512x512 | SM80_A100 | 1935.36 | 35895 | 6150144 | 171.3371 | 25.457 | HMMA x65536 | `modal/artifacts/bf16-512x512x512/traces/` | `modal/artifacts/bf16-512x512x512/sim_run/sim.out` |
| FP16 | FP32 | 512x512x512 | SM80_A100_LPDDR5X | 819.2 | 35936 | 6150144 | 171.1416 | 25.487 | same kernel / same trace replayed under LPDDR5X config | `modal/artifacts/fp16-512x512x512/traces/` | `modal/artifacts/fp16-512x512x512/sim_run_lpddr5x/sim.out` |
| BF16 | FP32 | 512x512x512 | SM80_A100_LPDDR5X | 819.2 | 36265 | 6150144 | 169.5890 | 25.720 | same kernel / same trace replayed under LPDDR5X config | `modal/artifacts/bf16-512x512x512/traces/` | `modal/artifacts/bf16-512x512x512/sim_run_lpddr5x/sim.out` |

A machine-readable export of the same table is saved at `modal/results/small_gemm_results.csv`.

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

## Large GPT-3-style GEMM replay comparison

### Captured large trace
A larger FP16 GEMM was captured on Modal A100-80GB using cuBLASLt/CUTLASS-generated SM80 tensor-core code:
- Shape: `2048x12288x12288`
- Kernel observed in trace:
  - `_ZN7cutlass7Kernel2I57cutlass_80_tensorop_s16816gemm_f16_128x256_32x3_nn_align8EEvNT_6ParamsE`
- Kernel header:
  - `grid dim = (64,12,2)`
  - `block dim = (256,1,1)`
  - `nregs = 220`
  - `shmem = 73728`
  - `binary version = 80`
- Processed trace inputs:
  - `modal/artifacts/fp16-2048x12288x12288-gemm3/raw_traces/kernelslist.g`
  - `modal/artifacts/fp16-2048x12288x12288-gemm3/raw_traces/kernel-3-ctx_0x55962a553360.traceg`

### Final replay evidence

#### Baseline `SM80_A100`
- Replay log:
  - `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_bar0_20260327-010117_pty/sim_live.out`
- Verified final metrics:
  - `gpu_tot_sim_cycle = 4848187`
  - `gpu_tot_sim_insn = 10244247315`
  - `gpu_tot_ipc = 2113.0059`
  - `gpgpu_simulation_time = 0 days, 2 hrs, 29 min, 13 sec (8953 sec)`
  - `GPGPU-Sim: *** exit detected ***`

#### LPDDR5X-style `SM80_A100_LPDDR5X`
- Replay log:
  - `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_lpddr5x_quiet_fg_20260327-105500/sim_live.out`
- Verified final metrics:
  - `gpu_tot_sim_cycle = 14992260`
  - `gpu_tot_sim_insn = 10244247315`
  - `gpu_tot_ipc = 683.3024`
  - `gpgpu_simulation_time = 0 days, 6 hrs, 25 min, 32 sec (23132 sec)`
  - `GPGPU-Sim: *** exit detected ***`

### Large GEMM baseline vs LPDDR5X replay comparison (1410 MHz core clock)

| Case | Config | Modeled BW (GB/s) | Cycles | Sim insn | IPC | Time (us @ 1410 MHz) | Slowdown vs A100 | Perf drop vs A100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 / FP32 acc / 2048x12288x12288 | SM80_A100 | 1935.36 | 4848187 | 10244247315 | 2113.0059 | 3438.430 | 1.000x | 0.00% |
| FP16 / FP32 acc / 2048x12288x12288 | SM80_A100_LPDDR5X | 819.2 | 14992260 | 10244247315 | 683.3024 | 10632.809 | 3.092x | 67.66% |

### Interpretation
For this large CUTLASS tensor-core GEMM, replacing the A100 HBM-like memory system with the 12-channel LPDDR5X-style 819.2 GB/s configuration causes a substantial modeled slowdown:
- slowdown factor: `3.092x`
- relative performance drop: `67.66%`

This large problem size is therefore much more sensitive to off-chip memory bandwidth than the previously verified `512x512x512` small GEMMs.

## Barrier root cause and repair update

The large `2048x12288x12288` replay failure was traced to trace-driven `OP_BAR` decoding collapsing distinct `BAR.SYNC.DEFER_BLOCKING` phases into a single barrier id (`bar_id = 0`). Runtime barrier instrumentation on the failing kernel showed repeated barrier traffic all landing in the same simulator slot, which is consistent with the placeholder logic in `gpu-simulator/trace-driven/trace_driven.cc`.

A minimal repair has now been implemented: `BAR.SYNC.DEFER_BLOCKING` is mapped to the default CTA barrier while other `OP_BAR` forms still use a stable per-kernel mapping. With that repair in place, the large replay now completes successfully for both the baseline A100 config and the LPDDR5X-style config.

## Notes
- The small `512x512x512` kernels were chosen first to keep Modal cost and turnaround low.
- The large trace artifacts are intentionally kept out of git because of their size.
- The summary tables in this file are the currently verified completed replays.
