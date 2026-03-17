#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <trace_dir> [accel-sim-args...]" >&2
  exit 1
fi

TRACE_DIR="$1"
shift

KERNELSLIST="${TRACE_DIR}/kernelslist.g"
ACCEL_SIM_BIN="${REPO_ROOT}/gpu-simulator/bin/release/accel-sim.out"
GPGPUSIM_CONFIG="${REPO_ROOT}/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config"
TRACE_CONFIG="${REPO_ROOT}/gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config"

if [[ ! -f "${KERNELSLIST}" ]]; then
  echo "[replay_with_accelsim] ERROR: missing kernelslist.g at ${KERNELSLIST}" >&2
  exit 1
fi
if [[ ! -x "${ACCEL_SIM_BIN}" ]]; then
  echo "[replay_with_accelsim] ERROR: accel-sim binary not executable: ${ACCEL_SIM_BIN}" >&2
  exit 1
fi
if [[ ! -f "${GPGPUSIM_CONFIG}" ]]; then
  echo "[replay_with_accelsim] ERROR: missing config: ${GPGPUSIM_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRACE_CONFIG}" ]]; then
  echo "[replay_with_accelsim] ERROR: missing config: ${TRACE_CONFIG}" >&2
  exit 1
fi

exec "${ACCEL_SIM_BIN}" \
  -trace "${KERNELSLIST}" \
  -config "${GPGPUSIM_CONFIG}" \
  -config "${TRACE_CONFIG}" \
  "$@"
