# Large GEMM LPDDR5X Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replay the already-captured FP16 2048x12288x12288 GEMM trace on both `SM80_A100` and `SM80_A100_LPDDR5X`, then report the modeled slowdown after reducing memory bandwidth from A100-class HBM2e to the existing 12-channel LPDDR5X-like config.

**Architecture:** Reuse the already-validated large GEMM trace and the successful `SM80_A100` replay as the baseline. Run the same `kernelslist.g` through the existing `SM80_A100_LPDDR5X` config without regenerating traces or changing compute-side parameters, then compare `gpu_tot_sim_cycle` and key memory statistics from the resulting logs.

**Tech Stack:** Accel-Sim standalone trace replay, GPGPU-Sim config files, shell log parsing.

---

### Task 1: Confirm baseline inputs and config mapping

**Files:**
- Read: `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_bar0_20260327-010117_pty/sim_live.out`
- Read: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config`
- Read: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100_LPDDR5X/gpgpusim.config`

**Step 1:** Verify the successful baseline replay log still contains `gpu_tot_sim_cycle`, `gpu_tot_sim_insn`, and `exit detected`.

Run: `grep -nE 'gpu_tot_sim_cycle =|gpu_tot_sim_insn =|gpu_tot_ipc =|exit detected' modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_bar0_20260327-010117_pty/sim_live.out`

Expected: lines showing the successful baseline metrics.

**Step 2:** Verify the LPDDR5X config is the intended memory-only variant.

Run: `grep -nE 'gpgpu_n_mem|gpgpu_clock_domains|gpgpu_dram_buswidth|gpgpu_dram_burst_length' gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100_LPDDR5X/gpgpusim.config`

Expected: 12 memory channels and the existing LPDDR5X-like memory parameters.

### Task 2: Replay the same large GEMM trace under `SM80_A100_LPDDR5X`

**Files:**
- Create: `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_lpddr5x_<timestamp>/run.sh`
- Create: `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_lpddr5x_<timestamp>/sim_live.out`
- Create: `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_lpddr5x_<timestamp>/meta.txt`

**Step 1:** Create a wrapper script that reuses the same binary and trace path but swaps the config pair to `SM80_A100_LPDDR5X`.

**Step 2:** Launch the replay and wait for completion.

Run pattern:
`stdbuf -oL -eL gpu-simulator/bin/release/accel-sim.out -trace <kernelslist.g> -config <lpddr_gpgpusim.config> -config <lpddr_trace.config> -gpgpu_num_cta_barriers 64`

Expected: final log contains `gpu_tot_sim_cycle` and `exit detected`.

### Task 3: Summarize slowdown against baseline

**Files:**
- Create or update: `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/lpddr5x_compare_2026-03-27.md`

**Step 1:** Extract baseline and LPDDR5X metrics.

Run: `grep -nE 'gpu_tot_sim_cycle =|gpu_tot_ipc =|gpgpu_simulation_time =|exit detected' <baseline_log> <lpddr_log>`

**Step 2:** Compute slowdown and relative performance.

Formulae:
- `slowdown = lpddr_cycles / a100_cycles`
- `perf_drop = 1 - a100_cycles / lpddr_cycles`

**Step 3:** Write a short markdown summary with:
- config name
- modeled bandwidth
- `gpu_tot_sim_cycle`
- slowdown vs A100
- relative performance drop

### Task 4: Verify and report

**Files:**
- Read: the two replay logs
- Read: the markdown summary

**Step 1:** Re-run the grep commands to confirm the numbers used in the report.

**Step 2:** Report the final comparison in the chat with explicit cycles and percentages.
