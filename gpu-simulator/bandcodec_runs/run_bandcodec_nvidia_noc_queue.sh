#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/work/accel-sim-framework/gpu-simulator
RUNNER=$ROOT/bandcodec_runs/run_bandcodec_nvidia_noc_case.sh

run_case() {
  local case_name=$1
  shift
  echo "[queue] start ${case_name} at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  "$RUNNER" "$case_name" "$@"
  echo "[queue] done ${case_name} at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
}

# Paper minimum queue: calibrated native baseline and BandCodec main point.
BANDCODEC_ENABLE=0 run_case large_lpddr5x_native_nvidia_noc
BANDCODEC_ENABLE=1 DECODER_COUNT=100 DECODER_CYCLES_PER_RECORD=4 run_case large_lpddr5x_bandcodec_nvidia_noc

# Compact decoder-throughput sensitivity. These keep the same trace and NoC.
BANDCODEC_ENABLE=1 DECODER_COUNT=80 DECODER_CYCLES_PER_RECORD=4 run_case large_lpddr5x_bandcodec_nvidia_noc_dec80
BANDCODEC_ENABLE=1 DECODER_COUNT=100 DECODER_CYCLES_PER_RECORD=8 run_case large_lpddr5x_bandcodec_nvidia_noc_dec100_lat8
