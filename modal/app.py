"""Minimal scaffold for the accelsim-cutlass-trace Modal app."""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

import modal

app = modal.App(name="accelsim-cutlass-trace")

_MODAL_ROOT = Path(__file__).resolve().parent
_FETCH_CUTLASS_SCRIPT = _MODAL_ROOT / "scripts" / "fetch_cutlass.sh"
_BUILD_RUNNER_SCRIPT = _MODAL_ROOT / "scripts" / "build_cutlass_runner.sh"
_BUILD_TRACER_SCRIPT = _MODAL_ROOT / "scripts" / "build_tracer.sh"
_RUN_TRACE_JOB_SCRIPT = _MODAL_ROOT / "scripts" / "run_trace_job.sh"


@app.function()
def validate_environment() -> dict[str, object]:
    """Return lightweight environment information without raising."""
    nvcc_path = shutil.which("nvcc")
    cuda_12_8 = Path("/usr/local/cuda-12.8/bin/nvcc")
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "nvcc_in_path": nvcc_path is not None,
        "nvcc_path": nvcc_path,
        "cuda_12_8_present": cuda_12_8.exists(),
        "fetch_cutlass_script": str(_FETCH_CUTLASS_SCRIPT),
        "build_cutlass_runner_script": str(_BUILD_RUNNER_SCRIPT),
        "build_tracer_script": str(_BUILD_TRACER_SCRIPT),
        "run_trace_job_script": str(_RUN_TRACE_JOB_SCRIPT),
        "tracer_root": str(_MODAL_ROOT.parent / "util" / "tracer_nvbit"),
        "default_trace_env": {
            "TRACES_FOLDER": str(_MODAL_ROOT / "artifacts" / "trace_job"),
            "TOOL_COMPRESS": "0",
            "TRACE_FILE_COMPRESS": "0",
        },
        "repo_root": str(_MODAL_ROOT.parent),
        "cwd": os.getcwd(),
    }
