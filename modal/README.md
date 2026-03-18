# Modal accelsim-cutlass-trace scaffold

This directory contains the initial Modal scaffold for the `accelsim-cutlass-trace` workflow.

## Usage

Create a local virtual environment at the repository root (or adapt the Python path below to your own environment):

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run Modal from inside this directory so the local `modal/` folder does not shadow the installed Modal package:

```sh
cd modal
../.venv/bin/python -m modal run app.py::validate_environment
```

The current app only exposes a lightweight environment validation helper.

## Local artifact helpers (Task 4)

Stage local trace outputs into a job-specific artifact directory:

```sh
python3 modal/scripts/download_artifacts.py <job-name> \
  --source-dir modal/artifacts/trace_job
```

This writes to:

```text
modal/artifacts/<job-name>/
```

Replay a trace directory locally with the repository's `SM80_A100` configs:

```sh
bash modal/scripts/replay_with_accelsim.sh modal/artifacts/<job-name>/traces
```

## Verified A100 trace runs

The following end-to-end runs have been completed and replayed locally with Accel-Sim's `SM80_A100` configuration:

- `fp16-512x512x512`
  - trace: `modal/artifacts/fp16-512x512x512/traces/`
  - replay log: `modal/artifacts/fp16-512x512x512/sim_run/sim.out`
- `bf16-512x512x512`
  - trace: `modal/artifacts/bf16-512x512x512/traces/`
  - replay log: `modal/artifacts/bf16-512x512x512/sim_run/sim.out`

See `modal/results/summary.md` for the replay metrics and current limitations.

## A100-LPDDR5X config variant

An `A100_LPDDR5X` config variant is available in this worktree. It keeps the `SM80_A100` compute side fixed and changes the memory-side bandwidth model to roughly 819.2 GB/s.

Replayed 512x512x512 traces show negligible timing change versus the baseline A100 config, indicating the current CUTLASS kernel is not strongly off-chip-bandwidth-bound at this size.
