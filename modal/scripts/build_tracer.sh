#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRACER_ROOT="${REPO_ROOT}/util/tracer_nvbit"
TRACER_TOOL_DIR="${TRACER_ROOT}/tracer_tool"
POSTPROC_DIR="${TRACER_TOOL_DIR}/traces-processing"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "[build_tracer] ERROR: expected CUDA 12.8 nvcc at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
export CUDA_HOME
export CUDAToolkit_ROOT="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${TRACER_ROOT}/nvbit_release/core/libnvbit.a" ]]; then
  echo "[build_tracer] NVBit release missing; installing via install_nvbit.sh"
  (cd "${TRACER_ROOT}" && bash ./install_nvbit.sh)
fi

ARCH="${ARCH:-sm_80}"
echo "[build_tracer] Building tracer_tool for ARCH=${ARCH}"
make -C "${TRACER_TOOL_DIR}" clean
make -C "${TRACER_TOOL_DIR}" ARCH="${ARCH}"

make -C "${POSTPROC_DIR}" clean
make -C "${POSTPROC_DIR}"

echo "[build_tracer] Built: ${TRACER_TOOL_DIR}/tracer_tool.so"
echo "[build_tracer] Built: ${POSTPROC_DIR}/post-traces-processing"
