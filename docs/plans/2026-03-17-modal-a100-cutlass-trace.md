# Modal A100 CUTLASS Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Modal-based A100-80GB CUTLASS GEMM trace workflow that generates Accel-Sim-compatible traces for FP16/BF16 GEMMs and replays them locally with the repository's `SM80_A100` configuration.

**Architecture:** The implementation adds a new `modal/` subtree that owns the remote execution flow, CUTLASS checkout/build orchestration, NVBit trace capture, artifact collection, and local replay helpers. The first milestone focuses on one clean end-to-end GEMM path with explicit `A100-80GB` selection and small shape defaults before expanding shape coverage.

**Tech Stack:** Python 3, Modal CLI/API, CUDA 12.8-compatible toolchain, CUTLASS, existing NVBit tracer from `util/tracer_nvbit`, local Accel-Sim `SM80_A100` configs.

---

### Task 1: Scaffold Modal workspace and documentation

**Files:**
- Create: `modal/README.md`
- Create: `modal/requirements.txt`
- Create: `modal/__init__.py`
- Create: `modal/app.py`
- Modify: `docs/plans/2026-03-17-modal-a100-cutlass-trace-design.md`

**Step 1: Write the failing test**

Create a simple smoke script description in `modal/README.md` that references `python -m modal` commands which do not yet exist in `modal/app.py`.

**Step 2: Run test to verify it fails**

Run:
```bash
python -m py_compile modal/app.py
```
Expected: FAIL because `modal/app.py` does not exist.

**Step 3: Write minimal implementation**

Create `modal/app.py` with:
- a `modal.App("accelsim-cutlass-trace")`
- one stub function for environment validation
- image definition pinned to CUDA-compatible base tooling

Also create `modal/README.md`, `modal/requirements.txt`, and `modal/__init__.py`.

**Step 4: Run test to verify it passes**

Run:
```bash
python -m py_compile modal/app.py
```
Expected: PASS.

**Step 5: Commit**

```bash
git add modal/README.md modal/requirements.txt modal/__init__.py modal/app.py
git commit -m "feat: scaffold modal trace workspace"
```

### Task 2: Add CUTLASS acquisition and remote build flow

**Files:**
- Create: `modal/scripts/fetch_cutlass.sh`
- Create: `modal/scripts/build_cutlass_runner.sh`
- Create: `modal/cutlass_runner/CMakeLists.txt`
- Create: `modal/cutlass_runner/main.cu`
- Modify: `modal/app.py`

**Step 1: Write the failing test**

Define the expected local validation command:
```bash
bash modal/scripts/build_cutlass_runner.sh
```
Expected initially: FAIL because scripts and runner sources do not exist.

**Step 2: Run test to verify it fails**

Run the command above and confirm missing-file failure.

**Step 3: Write minimal implementation**

Implement:
- `fetch_cutlass.sh` to clone a pinned CUTLASS revision into `modal/extern/cutlass`
- `build_cutlass_runner.sh` to configure/build a minimal runner
- `main.cu` to accept dtype and shape arguments and launch one CUTLASS GEMM for:
  - FP16 input / FP32 accumulate
  - BF16 input / FP32 accumulate

Shapes to support at minimum:
- `512 512 512`
- `512 12288 12288`

**Step 4: Run test to verify it passes**

Run:
```bash
bash modal/scripts/fetch_cutlass.sh
bash modal/scripts/build_cutlass_runner.sh
```
Expected: PASS and runner binary produced.

**Step 5: Commit**

```bash
git add modal/scripts/fetch_cutlass.sh modal/scripts/build_cutlass_runner.sh modal/cutlass_runner/CMakeLists.txt modal/cutlass_runner/main.cu modal/app.py
git commit -m "feat: add cutlass gemm runner build flow"
```

### Task 3: Add NVBit tracer integration for Modal jobs

**Files:**
- Create: `modal/scripts/build_tracer.sh`
- Create: `modal/scripts/run_trace_job.sh`
- Modify: `modal/app.py`
- Reference: `util/tracer_nvbit/install_nvbit.sh`
- Reference: `util/tracer_nvbit/tracer_tool/`

**Step 1: Write the failing test**

Define the expected command:
```bash
bash modal/scripts/run_trace_job.sh --binary /tmp/fake-runner --m 512 --n 512 --k 512 --dtype fp16
```
Expected initially: FAIL because script does not exist.

**Step 2: Run test to verify it fails**

Run the command above and confirm missing-file failure.

**Step 3: Write minimal implementation**

Implement:
- `build_tracer.sh` to build the repository's tracer stack inside the Modal image using CUDA 12.8
- `run_trace_job.sh` to:
  - export `TRACES_FOLDER`
  - disable problematic compression by default
  - preload tracer tool
  - execute the CUTLASS runner once
  - post-process traces into `kernelslist.g`

Update `modal/app.py` so the remote function calls the build and trace scripts.

**Step 4: Run test to verify it passes**

Run a local dry-run check:
```bash
bash -n modal/scripts/build_tracer.sh
bash -n modal/scripts/run_trace_job.sh
python -m py_compile modal/app.py
```
Expected: PASS.

**Step 5: Commit**

```bash
git add modal/scripts/build_tracer.sh modal/scripts/run_trace_job.sh modal/app.py
git commit -m "feat: integrate nvbit tracing into modal workflow"
```

### Task 4: Add artifact export and local replay helpers

**Files:**
- Create: `modal/scripts/download_artifacts.py`
- Create: `modal/scripts/replay_with_accelsim.sh`
- Modify: `modal/README.md`
- Modify: `modal/app.py`

**Step 1: Write the failing test**

Define the expected replay command:
```bash
bash modal/scripts/replay_with_accelsim.sh /tmp/nonexistent-trace-dir
```
Expected initially: FAIL because helper does not exist.

**Step 2: Run test to verify it fails**

Run the command above and confirm missing-file failure.

**Step 3: Write minimal implementation**

Implement:
- artifact download helper that stores Modal outputs under `modal/artifacts/<job-name>/`
- replay helper that invokes:
```bash
gpu-simulator/bin/release/accel-sim.out \
  -trace <trace_dir>/kernelslist.g \
  -config gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config \
  -config gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config
```
- README usage documenting exact commands.

**Step 4: Run test to verify it passes**

Run:
```bash
bash -n modal/scripts/replay_with_accelsim.sh
python -m py_compile modal/scripts/download_artifacts.py
```
Expected: PASS.

**Step 5: Commit**

```bash
git add modal/scripts/download_artifacts.py modal/scripts/replay_with_accelsim.sh modal/README.md modal/app.py
git commit -m "feat: add artifact retrieval and local replay helpers"
```

### Task 5: Execute first real Modal A100-80GB trace run

**Files:**
- Modify: `modal/README.md`
- Output: `modal/artifacts/fp16-512x512x512/`
- Output: `modal/artifacts/bf16-512x512x512/`
- Output: `modal/artifacts/fp16-512x12288x12288/` or `modal/artifacts/bf16-512x12288x12288/`

**Step 1: Write the failing test**

The failing condition is absence of real artifacts and replay logs.

**Step 2: Run test to verify it fails**

Run:
```bash
ls modal/artifacts/fp16-512x512x512
```
Expected: FAIL because no artifacts exist yet.

**Step 3: Write minimal implementation**

Use the new Modal app to run at least:
- FP16->FP32 accumulate, `512x512x512`
- BF16->FP32 accumulate, `512x512x512`

Then, if runtime is acceptable, add one GPT-3-style run with:
- `512x12288x12288`
for FP16 and/or BF16.

**Step 4: Run test to verify it passes**

Run:
```bash
ls modal/artifacts/fp16-512x512x512
bash modal/scripts/replay_with_accelsim.sh modal/artifacts/fp16-512x512x512/traces
```
Expected: PASS with produced trace artifacts and successful local replay log.

**Step 5: Commit**

```bash
git add modal/README.md modal/artifacts
git commit -m "feat: capture first modal a100 cutlass traces"
```

### Task 6: Verify BF16/FP16 trace replay outputs and summarize cost envelope

**Files:**
- Create: `modal/results/summary.md`
- Modify: `modal/README.md`

**Step 1: Write the failing test**

Define summary expectations:
- each recorded run must list GPU type, dtype, shape, remote runtime, artifact path, replay status

**Step 2: Run test to verify it fails**

Run:
```bash
test -f modal/results/summary.md
```
Expected: FAIL because summary file does not exist.

**Step 3: Write minimal implementation**

Create `modal/results/summary.md` capturing:
- requested Modal GPU: `A100-80GB`
- rationale for 80GB alignment with Accel-Sim `SM80_A100`
- actual runtime per run
- whether each trace replayed successfully
- whether the initial 2-minute cost target was met

**Step 4: Run test to verify it passes**

Run:
```bash
test -f modal/results/summary.md && grep -E 'A100-80GB|512x512x512|12288' modal/results/summary.md
```
Expected: PASS.

**Step 5: Commit**

```bash
git add modal/results/summary.md modal/README.md
git commit -m "docs: summarize modal a100 cutlass trace runs"
```
