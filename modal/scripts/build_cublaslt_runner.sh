#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER_SOURCE_DIR="${REPO_ROOT}/modal/cublaslt_runner"
BUILD_DIR="${RUNNER_SOURCE_DIR}/build"

if [[ -d "/usr/local/cuda-12.8" ]]; then
  export CUDA_HOME="/usr/local/cuda-12.8"
  export CUDAToolkit_ROOT="/usr/local/cuda-12.8"
fi

if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]] && command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  export CUDA_HOME
  export CUDAToolkit_ROOT="${CUDA_HOME}"
fi

if [[ -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME}}"
fi

if [[ ! -x "${CUDA_HOME:-}/bin/nvcc" ]]; then
  echo "[build_cublaslt_runner] ERROR: nvcc not found (looked in /usr/local/cuda-12.8 and PATH)" >&2
  exit 1
fi

cmake -S "${RUNNER_SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

cmake --build "${BUILD_DIR}" --parallel

echo "[build_cublaslt_runner] Built ${BUILD_DIR}/cublaslt_runner"
