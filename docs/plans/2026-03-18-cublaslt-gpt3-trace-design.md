# cuBLASLt GPT-3 GEMM Trace Design

**Date:** 2026-03-18
**Status:** Approved

## Goal
Generate a real A100 trace for a GPT-3-style large GEMM using cuBLASLt instead of CUTLASS, because the current minimal CUTLASS runner does not successfully launch the larger `Mx12288x12288` shapes.

## Selected approach
Use a dedicated cuBLASLt runner on Modal A100-80GB, attach the existing NVBit tracer, and store the resulting trace artifacts locally without committing the large trace files to git.

## Scope
- Support `fp16` and `bf16` inputs
- Use `fp32` accumulation
- Target shape: `2048 x 12288 x 12288`
- Reuse the existing Modal image/tracer workflow where possible
- Replay resulting traces locally with `SM80_A100`

## Cost control
- Trace only one GEMM launch per run
- Do not commit large raw traces to git
- Download only the required `traceg` and `kernelslist.g` if needed for replay
