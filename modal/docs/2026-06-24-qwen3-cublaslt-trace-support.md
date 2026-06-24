# Qwen3 FFN cuBLASLt Trace Runner Support Validation

Date: 2026-06-24
Worktree: `/work/accel-sim-framework/.worktrees/modal-a100-cutlass-trace`

## Scope and boundary

This note validates the local `modal/cublaslt_runner` support for Qwen3 FFN projection shapes in fp16:

- Gate/up projection: `Mx12288x4096`
- Down projection: `Mx4096x12288`

Boundary: this covers 4090-captured NVBit traces that are replayable with the A100 Accel-Sim SM80 flow. It is not a validation of a full PyTorch request, model serving path, or end-to-end Qwen3 inference workload.

## Build verification

Command:

```bash
bash modal/scripts/build_cublaslt_runner.sh
```

Output:

```text
-- Configuring done (0.1s)
-- Generating done (0.4s)
-- Build files have been written to: /work/accel-sim-framework/.worktrees/modal-a100-cutlass-trace/modal/cublaslt_runner/build
[100%] Built target cublaslt_runner
[build_cublaslt_runner] Built /work/accel-sim-framework/.worktrees/modal-a100-cutlass-trace/modal/cublaslt_runner/build/cublaslt_runner
```

Result: exit code `0`.

## Runner shape verification

All commands below used the freshly built binary at `modal/cublaslt_runner/build/cublaslt_runner` and completed within `timeout 60`.

### fp16, M=1, N=12288, K=4096

Command:

```bash
timeout 60 modal/cublaslt_runner/build/cublaslt_runner --dtype fp16 --m 1 --n 12288 --k 4096
```

Output:

```text
GEMM complete with cuBLASLt: M=1 N=12288 K=4096 sample=[0.5625, 0.375, 0.1875, 0]
```

Result: exit code `0`.

### fp16, M=1, N=4096, K=12288

Command:

```bash
timeout 60 modal/cublaslt_runner/build/cublaslt_runner --dtype fp16 --m 1 --n 4096 --k 12288
```

Output:

```text
GEMM complete with cuBLASLt: M=1 N=4096 K=12288 sample=[1, 0.53125, 0.0625, -0.40625]
```

Result: exit code `0`.

### fp16, M=2, N=12288, K=4096

Command:

```bash
timeout 60 modal/cublaslt_runner/build/cublaslt_runner --dtype fp16 --m 2 --n 12288 --k 4096
```

Output:

```text
GEMM complete with cuBLASLt: M=2 N=12288 K=4096 sample=[0.5625, 0.375, 0.1875, 0]
```

Result: exit code `0`.

### fp16, M=4, N=4096, K=12288

Command:

```bash
timeout 60 modal/cublaslt_runner/build/cublaslt_runner --dtype fp16 --m 4 --n 4096 --k 12288
```

Output:

```text
GEMM complete with cuBLASLt: M=4 N=4096 K=12288 sample=[1, 0.53125, 0.0625, -0.40625]
```

Result: exit code `0`.

## Existing trace directories

The Qwen3 FFN fp16 4090-captured, SM80-header trace directories are present locally.

Command:

```bash
ls -la \
  modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr \
  modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr
```

Output:

```text
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr:
total 0
drwxr-xr-x 1 liuwei liuwei   12 Jun 24 15:07 .
drwxr-xr-x 1 liuwei liuwei 6010 Jun 24 15:26 ..
drwxr-xr-x 1 liuwei liuwei  716 Jun 24 15:15 traces

modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr:
total 0
drwxr-xr-x 1 liuwei liuwei   12 Jun 24 15:11 .
drwxr-xr-x 1 liuwei liuwei 6010 Jun 24 15:26 ..
drwxr-xr-x 1 liuwei liuwei  582 Jun 24 15:15 traces
```

Trace file inventory command:

```bash
find \
  modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr \
  modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr \
  -maxdepth 2 -type f | sort
```

Output:

```text
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-1-ctx_0x603d9cc9e520.trace
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-1-ctx_0x603d9cc9e520.traceg
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-2-ctx_0x603d9cc9e520.trace
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-2-ctx_0x603d9cc9e520.traceg
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-3-ctx_0x603d9cc9e520.trace
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-3-ctx_0x603d9cc9e520.traceg
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-4-ctx_0x603d9cc9e520.trace
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernel-4-ctx_0x603d9cc9e520.traceg
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernelslist.g
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernelslist.gemm_only.g
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/kernelslist_ctx_0x603d9cc9e520
modal/artifacts/local-fp16-1x12288x4096-cublaslt-qwen3-gateup-sm80hdr/traces/stats_ctx_0x603d9cc9e520
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernel-1-ctx_0x5e2018586520.trace
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernel-1-ctx_0x5e2018586520.traceg
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernel-2-ctx_0x5e2018586520.trace
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernel-2-ctx_0x5e2018586520.traceg
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernel-3-ctx_0x5e2018586520.trace
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernel-3-ctx_0x5e2018586520.traceg
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernelslist.g
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernelslist.gemm_only.g
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/kernelslist_ctx_0x5e2018586520
modal/artifacts/local-fp16-1x4096x12288-cublaslt-qwen3-down-sm80hdr/traces/stats_ctx_0x5e2018586520
```

## Summary

- `modal/scripts/build_cublaslt_runner.sh` built the cuBLASLt runner successfully.
- The runner accepts and completes the requested Qwen3 FFN fp16 gate/up and down shapes for `M=1`, `M=2`, and `M=4` coverage listed above.
- The two existing 4090-captured NVBit trace directories are present and contain SM80-header trace artifacts suitable for the A100 Accel-Sim replay workflow.
- This validation does not claim full PyTorch request or end-to-end model-serving support.
