#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRACER_TOOL_SO="${REPO_ROOT}/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POST_PROCESSOR="${REPO_ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"
DEFAULT_CUTLASS_RUNNER="${REPO_ROOT}/modal/cutlass_runner/build/cutlass_runner"
DEFAULT_CUBLASLT_RUNNER="${REPO_ROOT}/modal/cublaslt_runner/build/cublaslt_runner"
DEFAULT_AGENT_KV_RUNNER="${REPO_ROOT}/modal/agent_kv_runner/build/agent_kv_runner"
RUNNER_KIND="${RUNNER_KIND:-cutlass}"
RUNNER_BIN="${RUNNER_BIN:-}"

runner_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runner-kind)
      if [[ $# -lt 2 ]]; then
        echo "[run_trace_job] ERROR: --runner-kind requires a value" >&2
        exit 1
      fi
      RUNNER_KIND="$2"
      shift 2
      ;;
    --runner-bin)
      if [[ $# -lt 2 ]]; then
        echo "[run_trace_job] ERROR: --runner-bin requires a value" >&2
        exit 1
      fi
      RUNNER_BIN="$2"
      shift 2
      ;;
    *)
      runner_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${RUNNER_BIN}" ]]; then
  case "${RUNNER_KIND}" in
    cutlass)
      RUNNER_BIN="${DEFAULT_CUTLASS_RUNNER}"
      ;;
    cublaslt)
      RUNNER_BIN="${DEFAULT_CUBLASLT_RUNNER}"
      ;;
    agent_kv)
      RUNNER_BIN="${DEFAULT_AGENT_KV_RUNNER}"
      ;;
    *)
      echo "[run_trace_job] ERROR: unsupported runner kind '${RUNNER_KIND}' (use cutlass, cublaslt, or agent_kv)" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -f "${TRACER_TOOL_SO}" ]]; then
  echo "[run_trace_job] ERROR: tracer tool not found at ${TRACER_TOOL_SO}. Run modal/scripts/build_tracer.sh first." >&2
  exit 1
fi
if [[ ! -x "${POST_PROCESSOR}" ]]; then
  echo "[run_trace_job] ERROR: post processor not found at ${POST_PROCESSOR}. Run modal/scripts/build_tracer.sh first." >&2
  exit 1
fi
if [[ ! -x "${RUNNER_BIN}" ]]; then
  if [[ "${RUNNER_KIND}" == "cublaslt" ]]; then
    echo "[run_trace_job] ERROR: cublaslt runner not found at ${RUNNER_BIN}. Run modal/scripts/build_cublaslt_runner.sh first." >&2
  elif [[ "${RUNNER_KIND}" == "agent_kv" ]]; then
    echo "[run_trace_job] ERROR: agent_kv runner not found at ${RUNNER_BIN}. Run modal/scripts/build_agent_kv_runner.sh first." >&2
  else
    echo "[run_trace_job] ERROR: runner not found at ${RUNNER_BIN}. Run modal/scripts/build_cutlass_runner.sh first or pass --runner-kind cublaslt." >&2
  fi
  exit 1
fi

export TRACES_FOLDER="${TRACES_FOLDER:-${REPO_ROOT}/modal/artifacts/trace_job}"
TRACE_OUTPUT_DIR="${TRACES_FOLDER}/traces"
mkdir -p "${TRACE_OUTPUT_DIR}"
rm -f "${TRACE_OUTPUT_DIR}"/kernel-*.trace "${TRACE_OUTPUT_DIR}"/kernel-*.trace.xz \
      "${TRACE_OUTPUT_DIR}"/kernel-*.traceg "${TRACE_OUTPUT_DIR}"/kernel-*.traceg.xz \
      "${TRACE_OUTPUT_DIR}"/kernelslist "${TRACE_OUTPUT_DIR}"/kernelslist.g

# Compression is disabled by default to avoid known xz reliability/runtime issues.
export TOOL_COMPRESS="${TOOL_COMPRESS:-0}"
export TRACE_FILE_COMPRESS="${TRACE_FILE_COMPRESS:-0}"

# NVBit preloads/injects the tracer through CUDA_INJECTION64_PATH.
export CUDA_INJECTION64_PATH="${TRACER_TOOL_SO}"

"${RUNNER_BIN}" "${runner_args[@]}"
"${POST_PROCESSOR}" "${TRACE_OUTPUT_DIR}"

if [[ ! -f "${TRACE_OUTPUT_DIR}/kernelslist.g" ]]; then
  echo "[run_trace_job] ERROR: kernelslist.g not generated in ${TRACE_OUTPUT_DIR}" >&2
  exit 1
fi

echo "[run_trace_job] Trace complete"
echo "[run_trace_job] TRACES_FOLDER=${TRACES_FOLDER}"
echo "[run_trace_job] kernelslist.g=${TRACE_OUTPUT_DIR}/kernelslist.g"
