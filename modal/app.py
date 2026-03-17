"""Minimal scaffold for the accelsim-cutlass-trace Modal app."""
from __future__ import annotations

import platform
import shutil
import sys

import modal

app = modal.App(name="accelsim-cutlass-trace")


@modal.function()
def validate_environment() -> dict[str, object]:
    """Return lightweight environment information without raising."""
    nvcc_path = shutil.which("nvcc")
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "nvcc_in_path": nvcc_path is not None,
        "nvcc_path": nvcc_path,
    }
