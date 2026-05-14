#!/usr/bin/env python3
"""Integration test for Accel-Sim NVBit trace header metadata."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

UTF8 = "utf-8"
TRACER_TOOL_DIR = Path(__file__).resolve().parents[1]
TRACER_SO = TRACER_TOOL_DIR / "tracer_tool.so"
CUDA_ROOT = Path("/usr/local/cuda")
NVCC = CUDA_ROOT / "bin" / "nvcc"


def run_checked(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout + completed.stderr


def write_cuda_app(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r"""
            #include <cuda_runtime.h>

            __global__ void metadata_kernel(int *out) {
              __shared__ int scratch[64];
              int tid = threadIdx.x;
              scratch[tid] = tid;
              __syncthreads();
              out[tid] = scratch[tid] + 1;
            }

            int main() {
              int *device_output = nullptr;
              cudaMalloc(&device_output, 64 * sizeof(int));
              metadata_kernel<<<1, 64>>>(device_output);
              cudaDeviceSynchronize();
              cudaFree(device_output);
              return 0;
            }
            """
        ).strip()
        + "\n",
        encoding=UTF8,
    )


def parse_trace_header(trace_file: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in trace_file.read_text(encoding=UTF8).splitlines():
        if not line.startswith("-"):
            break
        key, value = line[1:].split(" = ", 1)
        header[key.strip()] = value.strip()
    return header


def main() -> int:
    if not TRACER_SO.exists():
        raise FileNotFoundError(f"missing tracer shared object: {TRACER_SO}")
    if not NVCC.exists():
        raise FileNotFoundError(f"missing nvcc: {NVCC}")

    with tempfile.TemporaryDirectory(prefix="accelsim-tracer-metadata-") as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        source = tmpdir / "metadata_kernel.cu"
        binary = tmpdir / "metadata_kernel"
        traces_root = tmpdir / "trace_root"
        write_cuda_app(source)

        run_checked([str(NVCC), "-arch=sm_80", str(source), "-o", str(binary)], cwd=tmpdir)
        env = os.environ.copy()
        env.update(
            {
                "CUDA_INJECTION64_PATH": str(TRACER_SO),
                "DYNAMIC_KERNEL_RANGE": "1-1",
                "TRACE_FILE_COMPRESS": "0",
                "TRACES_FOLDER": str(traces_root),
            }
        )
        env["PATH"] = str(CUDA_ROOT / "bin") + os.pathsep + env.get("PATH", "")
        traces_root.mkdir()
        run_checked([str(binary)], cwd=tmpdir, env=env)

        trace_files = sorted((traces_root / "traces").glob("kernel-*.trace"))
        if len(trace_files) != 1:
            raise AssertionError(f"expected one trace file, found {trace_files}")

        header = parse_trace_header(trace_files[0])
        binary_version = header.get("binary version")
        if binary_version != "80":
            raise AssertionError(f"expected binary version 80, got {binary_version!r}; header={header}")

        nregs = int(header.get("nregs", "0"))
        if not 0 < nregs < 256:
            raise AssertionError(f"expected sane nregs, got {nregs}; header={header}")

        shmem = int(header.get("shmem", "0"))
        if shmem != 256:
            raise AssertionError(f"expected 256 bytes static shmem, got {shmem}; header={header}")

        kernel_name = header.get("kernel name", "")
        if not re.search(r"metadata_kernel", kernel_name):
            raise AssertionError(f"unexpected kernel name {kernel_name!r}; header={header}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
