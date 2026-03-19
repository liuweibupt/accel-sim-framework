"""Modal app for GEMM trace capture on A100."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

app = modal.App(name="accelsim-cutlass-trace")

_MODAL_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MODAL_ROOT.parent
_FETCH_CUTLASS_SCRIPT = Path("/root/project/modal/scripts/fetch_cutlass.sh")
_BUILD_RUNNER_SCRIPT = Path("/root/project/modal/scripts/build_cutlass_runner.sh")
_BUILD_CUBLASLT_RUNNER_SCRIPT = Path("/root/project/modal/scripts/build_cublaslt_runner.sh")
_BUILD_TRACER_SCRIPT = Path("/root/project/modal/scripts/build_tracer.sh")
_RUN_TRACE_JOB_SCRIPT = Path("/root/project/modal/scripts/run_trace_job.sh")
_DOWNLOAD_ARTIFACTS_SCRIPT = _MODAL_ROOT / "scripts" / "download_artifacts.py"
_REPLAY_WITH_ACCELSIM_SCRIPT = _MODAL_ROOT / "scripts" / "replay_with_accelsim.sh"
_ARTIFACT_VOLUME_NAME = "accelsim-cutlass-traces"
_ARTIFACTS_ROOT = _MODAL_ROOT / "artifacts"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "wget",
        "xz-utils",
        "bc",
    )
    .add_local_dir(str(_MODAL_ROOT / "scripts"), remote_path="/root/project/modal/scripts", copy=True)
    .add_local_dir(str(_MODAL_ROOT / "cutlass_runner"), remote_path="/root/project/modal/cutlass_runner", copy=True)
    .add_local_dir(str(_MODAL_ROOT / "cublaslt_runner"), remote_path="/root/project/modal/cublaslt_runner", copy=True)
    .add_local_dir(str(_REPO_ROOT / "util" / "tracer_nvbit"), remote_path="/root/project/util/tracer_nvbit", copy=True)
    .run_commands(
        "cd /root/project && bash modal/scripts/fetch_cutlass.sh",
        "cd /root/project && bash modal/scripts/build_cutlass_runner.sh",
        "cd /root/project && bash modal/scripts/build_cublaslt_runner.sh",
        "cd /root/project && ARCH=sm_80 bash modal/scripts/build_tracer.sh",
    )
)

trace_volume = modal.Volume.from_name(_ARTIFACT_VOLUME_NAME, create_if_missing=True)


def _validate_shape(m: int, n: int, k: int, runner_kind: str) -> None:
    if runner_kind == "cutlass":
        allowed = {(512, 512, 512), (512, 12288, 12288)}
    elif runner_kind == "cublaslt":
        allowed = {(2048, 12288, 12288)}
    else:
        raise ValueError("runner_kind must be cutlass or cublaslt")
    if (m, n, k) not in allowed:
        raise ValueError(f"unsupported shape {(m, n, k)} for runner_kind={runner_kind}; allowed shapes: {sorted(allowed)}")


def _default_job_name(dtype: str, m: int, n: int, k: int) -> str:
    return f"{dtype}-{m}x{n}x{k}-{int(time.time())}"


@app.function()
def validate_environment() -> dict[str, object]:
    """Return lightweight local environment information without raising."""
    nvcc_path = shutil.which("nvcc")
    cuda_12_8 = Path("/usr/local/cuda-12.8/bin/nvcc")
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "nvcc_in_path": nvcc_path is not None,
        "nvcc_path": nvcc_path,
        "cuda_12_8_present": cuda_12_8.exists(),
        "download_artifacts_script": str(_DOWNLOAD_ARTIFACTS_SCRIPT),
        "replay_with_accelsim_script": str(_REPLAY_WITH_ACCELSIM_SCRIPT),
        "artifact_volume_name": _ARTIFACT_VOLUME_NAME,
        "artifacts_root": str(_ARTIFACTS_ROOT),
        "repo_root": str(_REPO_ROOT),
        "cwd": os.getcwd(),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 4,
    ephemeral_disk=524288,
    volumes={"/artifacts": trace_volume},
)
def run_trace(
    dtype: str = "fp16",
    m: int = 512,
    n: int = 512,
    k: int = 512,
    job_name: str = "",
    runner_kind: str = "cutlass",
    runner_bin: str = "",
    dynamic_kernel_range: str = "",
) -> dict[str, object]:
    _validate_shape(m, n, k, runner_kind=runner_kind)
    if dtype not in {"fp16", "bf16"}:
        raise ValueError("dtype must be fp16 or bf16")
    if not job_name:
        job_name = _default_job_name(dtype, m, n, k)

    trace_root = Path("/artifacts") / job_name
    env = os.environ.copy()
    env["TRACES_FOLDER"] = str(trace_root)
    env.setdefault("TOOL_COMPRESS", "0")
    env.setdefault("TRACE_FILE_COMPRESS", "0")
    if dynamic_kernel_range:
        env["DYNAMIC_KERNEL_RANGE"] = dynamic_kernel_range

    cmd = [
        str(_RUN_TRACE_JOB_SCRIPT),
        "--runner-kind",
        runner_kind,
        "--dtype",
        dtype,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
    ]
    if runner_bin:
        cmd.extend(["--runner-bin", runner_bin])
    subprocess.run(cmd, cwd="/root/project", env=env, check=True)
    trace_volume.commit()

    trace_dir = trace_root / "traces"
    files = []
    if trace_dir.exists():
        files = sorted(str(p.relative_to(trace_root)) for p in trace_dir.rglob("*") if p.is_file())

    return {
        "job_name": job_name,
        "dtype": dtype,
        "shape": [m, n, k],
        "runner_kind": runner_kind,
        "runner_bin": runner_bin or "(default)",
        "dynamic_kernel_range": dynamic_kernel_range or "(all kernels)",
        "gpu": "A100-80GB",
        "artifact_volume": _ARTIFACT_VOLUME_NAME,
        "remote_artifact_root": str(trace_root),
        "remote_trace_dir": str(trace_dir),
        "files": files,
    }


@app.local_entrypoint()
def main(
    dtype: str = "fp16",
    m: int = 512,
    n: int = 512,
    k: int = 512,
    job_name: str = "",
    runner_kind: str = "cutlass",
    runner_bin: str = "",
    dynamic_kernel_range: str = "",
    skip_download: bool = False,
):
    result = run_trace.remote(
        dtype=dtype,
        m=m,
        n=n,
        k=k,
        job_name=job_name,
        runner_kind=runner_kind,
        runner_bin=runner_bin,
        dynamic_kernel_range=dynamic_kernel_range,
    )
    print(result)

    if skip_download:
        return

    local_dest = _ARTIFACTS_ROOT / result["job_name"]
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            _ARTIFACT_VOLUME_NAME,
            f"/{result['job_name']}",
            str(local_dest),
            "--force",
        ],
        cwd="/tmp",
        check=True,
    )
    print(f"downloaded_artifacts={local_dest}")
