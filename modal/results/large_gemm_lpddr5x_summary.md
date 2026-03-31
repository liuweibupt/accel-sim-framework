# Large GEMM A100 vs LPDDR5X Replay Summary

## Workload
- dtype: `FP16`
- accumulate: `FP32`
- shape: `2048x12288x12288`
- trace:
  - `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/traces/reprocessed/`

## Verified replay results

| Config | Modeled BW (GB/s) | Cycles | Sim insn | IPC | Time (us @ 1410 MHz) | Slowdown vs A100 | Perf drop vs A100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `SM80_A100` | 1935.36 | 4848187 | 10244247315 | 2113.0059 | 3438.430 | 1.000x | 0.00% |
| `SM80_A100_LPDDR5X` | 819.2 | 14992260 | 10244247315 | 683.3024 | 10632.809 | 3.092x | 67.66% |

## Replay logs
- A100:
  - `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_bar0_20260327-010117_pty/sim_live.out`
- LPDDR5X:
  - `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_lpddr5x_quiet_fg_20260327-105500/sim_live.out`

## Conclusion
For this large CUTLASS tensor-core GEMM, reducing the modeled off-chip bandwidth from the A100 baseline to the 12-channel LPDDR5X-style `819.2 GB/s` configuration increases replay cycles by `3.092x`, corresponding to a `67.66%` relative performance drop.
