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
_BUILD_AGENT_KV_RUNNER_SCRIPT = Path("/root/project/modal/scripts/build_agent_kv_runner.sh")
_BUILD_DEEPSEEK_V4_RUNNER_SCRIPT = Path("/root/project/modal/scripts/build_deepseek_v4_runner.sh")
_BUILD_TRACER_SCRIPT = Path("/root/project/modal/scripts/build_tracer.sh")
_RUN_TRACE_JOB_SCRIPT = Path("/root/project/modal/scripts/run_trace_job.sh")
_DOWNLOAD_ARTIFACTS_SCRIPT = _MODAL_ROOT / "scripts" / "download_artifacts.py"
_REPLAY_WITH_ACCELSIM_SCRIPT = _MODAL_ROOT / "scripts" / "replay_with_accelsim.sh"
_ARTIFACT_VOLUME_NAME = "accelsim-cutlass-traces"
_ARTIFACTS_ROOT = _MODAL_ROOT / "artifacts"


def _prepare_artifact_download_destination(artifacts_root: Path, job_name: str) -> Path:
    local_dest = artifacts_root / job_name
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    if local_dest.exists():
        if local_dest.is_dir():
            shutil.rmtree(local_dest)
        else:
            local_dest.unlink()
    return local_dest


_COPY_IGNORED_DIRS = {"build", "__pycache__"}
_A100_EPHEMERAL_DISK_MIB = 524288
_NCU_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
)
_NCU_RUNNERS = {
    "cutlass": Path("/root/project/modal/cutlass_runner/build/cutlass_runner"),
    "cublaslt": Path("/root/project/modal/cublaslt_runner/build/cublaslt_runner"),
}


def _ignore_modal_copy_path(path: Path) -> bool:
    return any(part in _COPY_IGNORED_DIRS for part in path.parts)


def _build_ncu_command(dtype: str, m: int, n: int, k: int, runner_kind: str) -> tuple[list[str], Path]:
    binary = _NCU_RUNNERS.get(runner_kind)
    if binary is None:
        raise ValueError("runner_kind must be cutlass or cublaslt for Nsight Compute collection")
    cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--csv",
        "--metrics",
        ",".join(_NCU_METRICS),
        str(binary),
        "--dtype",
        dtype,
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
    ]
    return cmd, binary


def _read_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def _raise_for_ncu_failure(returncode: int, stderr_path: Path) -> None:
    if returncode == 0:
        return
    stderr_tail = _read_tail(stderr_path)
    raise RuntimeError(f"ncu failed with rc={returncode}; stderr tail:\n{stderr_tail}")


def _build_gpu_diagnostic_command() -> list[str]:
    return [
        "bash",
        "-lc",
        "set -x; "
        "nvidia-smi; "
        "nvidia-smi -q | sed -n '1,220p'; "
        "ncu --version; "
        "ldconfig -p | grep -E 'libcuda|libnvidia-ml|libcupti|libnvidia-.*perf' || true; "
        "find /usr/local -maxdepth 5 -name 'libcupti.so*' -o -name 'libnvidia-ml.so*' -o -name 'libcuda.so*' | sort",
    ]


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
    .add_local_dir(
        str(_MODAL_ROOT / "cutlass_runner"),
        remote_path="/root/project/modal/cutlass_runner",
        copy=True,
        ignore=_ignore_modal_copy_path,
    )
    .add_local_dir(
        str(_MODAL_ROOT / "cublaslt_runner"),
        remote_path="/root/project/modal/cublaslt_runner",
        copy=True,
        ignore=_ignore_modal_copy_path,
    )
    .add_local_dir(
        str(_MODAL_ROOT / "agent_kv_runner"),
        remote_path="/root/project/modal/agent_kv_runner",
        copy=True,
        ignore=_ignore_modal_copy_path,
    )
    .add_local_dir(
        str(_MODAL_ROOT / "deepseek_v4_runner"),
        remote_path="/root/project/modal/deepseek_v4_runner",
        copy=True,
        ignore=_ignore_modal_copy_path,
    )
    .add_local_dir(str(_REPO_ROOT / "util" / "tracer_nvbit"), remote_path="/root/project/util/tracer_nvbit", copy=True)
    .run_commands(
        "cd /root/project && bash modal/scripts/fetch_cutlass.sh",
        "cd /root/project && bash modal/scripts/build_cutlass_runner.sh",
        "cd /root/project && bash modal/scripts/build_cublaslt_runner.sh",
        "cd /root/project && bash modal/scripts/build_agent_kv_runner.sh",
        "cd /root/project && bash modal/scripts/build_deepseek_v4_runner.sh",
        "cd /root/project && ARCH=sm_80 bash modal/scripts/build_tracer.sh",
    )
)

trace_volume = modal.Volume.from_name(_ARTIFACT_VOLUME_NAME, create_if_missing=True)


def _validate_shape(m: int, n: int, k: int, runner_kind: str) -> None:
    if runner_kind in {"agent_kv", "deepseek_v4"}:
        return
    if runner_kind == "cutlass":
        allowed = {(512, 512, 512), (512, 12288, 12288)}
    elif runner_kind == "cublaslt":
        allowed = {
            (2048, 12288, 12288),
            # LLaMA-3.1-8B-Instruct projection shapes: q/o, k/v, gate/up, down.
            (2048, 4096, 4096),
            (2048, 1024, 4096),
            (2048, 14336, 4096),
            (2048, 4096, 14336),
        }
    else:
        raise ValueError("runner_kind must be cutlass, cublaslt, agent_kv, or deepseek_v4")
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
    ephemeral_disk=_A100_EPHEMERAL_DISK_MIB,
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
    if runner_kind not in {"agent_kv", "deepseek_v4"} and dtype not in {"fp16", "bf16"}:
        raise ValueError("dtype must be fp16 or bf16")
    if not job_name:
        if runner_kind == "agent_kv":
            job_name = f"agent-kv-bs{m}-kv{n}-prefix{k}-{int(time.time())}"
        elif runner_kind == "deepseek_v4":
            job_name = f"deepseek-v4-bs{m}-kv{n}-prefix{k}-{int(time.time())}"
        else:
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
    if runner_kind == "agent_kv":
        cmd = [
            str(_RUN_TRACE_JOB_SCRIPT),
            "--runner-kind",
            "agent_kv",
            "--batch",
            str(m),
            "--kvlen",
            str(n),
            "--shared-prefix",
            str(k),
            "--samples-per-page",
            "32",
        ]
    elif runner_kind == "deepseek_v4":
        selected_blocks = max(64, min(512, n // 32))
        selected_shared_blocks = int(selected_blocks * (k / n)) if n else 0
        cmd = [
            str(_RUN_TRACE_JOB_SCRIPT),
            "--runner-kind",
            "deepseek_v4",
            "--batch",
            str(m),
            "--kvlen",
            str(n),
            "--shared-prefix",
            str(k),
            "--selected-blocks",
            str(selected_blocks),
            "--selected-shared-blocks",
            str(selected_shared_blocks),
            "--samples-per-block",
            "32",
        ]
        env.setdefault("DYNAMIC_KERNEL_RANGE", "7-11")
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


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    ephemeral_disk=_A100_EPHEMERAL_DISK_MIB,
    volumes={"/artifacts": trace_volume},
)
def run_ncu_gemm(
    dtype: str = "fp16",
    m: int = 512,
    n: int = 512,
    k: int = 512,
    job_name: str = "",
    runner_kind: str = "cutlass",
) -> dict[str, object]:
    _validate_shape(m, n, k, runner_kind=runner_kind)
    if dtype not in {"fp16", "bf16"}:
        raise ValueError("dtype must be fp16 or bf16")
    cmd, binary = _build_ncu_command(dtype, m, n, k, runner_kind)
    if not binary.exists():
        raise FileNotFoundError(binary)
    if shutil.which("ncu") is None:
        raise FileNotFoundError("ncu")
    if not job_name:
        job_name = f"ncu-{runner_kind}-{dtype}-{m}x{n}x{k}-{int(time.time())}"

    out_dir = Path("/artifacts") / job_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = out_dir / "ncu.csv"
    stderr_path = out_dir / "ncu.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(cmd, cwd="/root/project", stdout=stdout, stderr=stderr, text=True, check=False)
    trace_volume.commit()
    _raise_for_ncu_failure(completed.returncode, stderr_path)
    return {
        "job_name": job_name,
        "dtype": dtype,
        "shape": [m, n, k],
        "runner_kind": runner_kind,
        "gpu": "A100-80GB",
        "artifact_volume": _ARTIFACT_VOLUME_NAME,
        "remote_artifact_root": str(out_dir),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "metrics": list(_NCU_METRICS),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=2,
    memory=8192,
    timeout=15 * 60,
    ephemeral_disk=_A100_EPHEMERAL_DISK_MIB,
)
def gpu_diagnostics() -> dict[str, object]:
    completed = subprocess.run(
        _build_gpu_diagnostic_command(),
        cwd="/root/project",
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
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

    local_dest = _prepare_artifact_download_destination(_ARTIFACTS_ROOT, result["job_name"])
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
