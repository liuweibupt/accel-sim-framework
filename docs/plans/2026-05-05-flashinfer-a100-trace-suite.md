# FlashInfer A100 Trace Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collect real Modal A100 shared-prefix paged-KV address traces and feed them into the existing LLMCompass + ChampSim L3/compression evaluation flow.

**Architecture:** Add a Modal runner path that can execute a FlashInfer-like paged decode/shared-prefix kernel on A100 under NVBit, starting with the existing `agent_kv` runner if FlashInfer Python packaging blocks trace collection. Convert generated `.trace` files into LLMCompass bridge entries, run ChampSim L3 sweeps, and publish a report that clearly labels evidence level.

**Tech Stack:** Modal A100, CUDA/NVBit/Accel-Sim trace format, Python unittest, LLMCompass `software_model.champsim_bridge`, ChampSim `llmcompass_replay`, matplotlib reports.

---

### Task 1: Validate Modal auth and existing A100 runner

**Files:**
- Read: `modal/app.py`
- Read: `modal/scripts/run_trace_job.sh`
- Read: `modal/agent_kv_runner/main.cu`

**Steps:**
1. Run `modal token info` without printing secrets.
2. Run a small A100 agent-KV trace job if no current artifact exists.
3. Confirm `kernelslist.g` and `kernel-2*.trace` exist.

### Task 2: Add FlashInfer/paged-KV suite execution wrapper

**Files:**
- Modify: `modal/app.py`
- Modify: `modal/scripts/run_trace_job.sh` if needed
- Modify/Create: `modal/scripts/run_flashinfer_trace_suite.sh` if Python FlashInfer is viable

**Steps:**
1. Add failing tests or smoke checks for the new runner-kind/suite arguments.
2. Implement minimal runner support.
3. Run a small smoke job: batch=4, kvlen=4096, shared_prefix=3072.
4. If FlashInfer package/kernel cannot run under NVBit within time, explicitly record blocker and use the existing A100 paged-KV CUDA runner as the traceable kernel baseline.

### Task 3: Run trace suite v1

**Matrix:**
- batch: `1,4`
- kvlen: `4096,16384,65536` where trace size permits
- shared_prefix_ratio: `0,0.75`
- memory instruction caps in LLMCompass replay: `4096` for fast rows; larger caps only for selected rows.

**Artifacts:**
- Modal volume: `accelsim-cutlass-traces`
- Local: `modal/artifacts/<job-name>.download/`

### Task 4: Extend LLMCompass parser and report

**Files:**
- Modify: `/work/LLMCompass-private/.worktrees/champsim-llc-bridge/L3cache/trace_backed_champsim_experiments.py`
- Modify: `/work/LLMCompass-private/.worktrees/champsim-llc-bridge/L3cache/l3_decode_extended_experiments.py`
- Test: `/work/LLMCompass-private/.worktrees/champsim-llc-bridge/tests/test_trace_backed_champsim_experiments.py`

**Steps:**
1. Write failing tests for suite metadata, trace discovery, and report rows.
2. Implement trace bundle generation for multiple A100 agent/FlashInfer-style traces.
3. Run L3 sweep `0,64,320,512` and compression overlay `kv4`/`act2_weight4_kv4`.
4. Update Markdown/CSV/JSON/figures.

### Task 5: Verify, commit, push

**Commands:**
```bash
cd /work/LLMCompass-private/.worktrees/champsim-llc-bridge
timeout 60s .venv/bin/python -m unittest -v tests.test_trace_backed_champsim_experiments tests.test_l3_decode_extended_experiments tests.test_champsim_bridge_invocation
```

**Commit:**
- Commit LLMCompass changes and push.
- Commit Accel-Sim/Modal changes and push if runner files changed.
