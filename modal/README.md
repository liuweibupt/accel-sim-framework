# Modal accelsim-cutlass-trace scaffold

This directory contains the initial Modal scaffold for the `accelsim-cutlass-trace` workflow.

## Usage

Run Modal from inside this directory so the local `modal/` folder does not shadow the installed Modal package:

```sh
cd modal
python -m pip install -r requirements.txt
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
