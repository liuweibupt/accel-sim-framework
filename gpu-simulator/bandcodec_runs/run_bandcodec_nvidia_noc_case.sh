#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CASE_NAME [accel-sim option ...]" >&2
  exit 2
fi

CASE_NAME=$1
shift

export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=/tmp/gpgpu-local-tools:${PATH}
export LIBRARY_PATH=/tmp/gpgpu-local-lib:${LIBRARY_PATH:-}
set +u
source /work/accel-sim-framework/gpu-simulator/setup_environment.sh >/tmp/accelsim_setup_bandcodec_nvidia_noc_case.log 2>&1
set -Eeuo pipefail

TRACE=${TRACE:-/work/accel-sim-framework/.worktrees/modal-a100-cutlass-trace/modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/traces/reprocessed/kernelslist.g}
CFG1=${CFG1:-/work/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100_LPDDR5X_NVIDIA_NOC/gpgpusim.config}
CFG2=${CFG2:-/work/accel-sim-framework/gpu-simulator/configs/tested-cfgs/SM80_A100_LPDDR5X_NVIDIA_NOC/trace.config}
BIN=${BIN:-/work/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out}
WEIGHT_BASE=${WEIGHT_BASE:-0x00002ae461000000}
WEIGHT_BYTES=${WEIGHT_BYTES:-301989888}
WEIGHT_BYTES_HEX=${WEIGHT_BYTES_HEX:-0x12000000}
COMPRESSION_RATIO=${COMPRESSION_RATIO:-4}
DECODER_COUNT=${DECODER_COUNT:-100}
DECODER_CYCLES_PER_RECORD=${DECODER_CYCLES_PER_RECORD:-4}
RECORD_BYTES=${RECORD_BYTES:-256}
BANDCODEC_ENABLE=${BANDCODEC_ENABLE:-1}

OUTDIR=/work/accel-sim-framework/gpu-simulator/bandcodec_runs/${CASE_NAME}_$(date -u '+%Y%m%d-%H%M%S')
mkdir -p "$OUTDIR"

RC_RECORDED=0
record_exit() {
  local rc=$?
  if [[ "$RC_RECORDED" == "0" ]]; then
    {
      echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo "rc=$rc"
      echo "failing_command=${BASH_COMMAND}"
      echo "failing_line=${BASH_LINENO[0]:-unknown}"
    } >> "$OUTDIR/manifest.txt"
    RC_RECORDED=1
  fi
  exit "$rc"
}
trap record_exit EXIT

cat > "$OUTDIR/manifest.txt" <<MANIFEST
start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
case_name=$CASE_NAME
trace=$TRACE
binary=$BIN
gpgpusim_config=$CFG1
trace_config=$CFG2
bandcodec_enable=$BANDCODEC_ENABLE
weight_base=$WEIGHT_BASE
weight_bytes=$WEIGHT_BYTES
weight_bytes_hex=$WEIGHT_BYTES_HEX
weight_range_method=B-matrix base inferred from large CUTLASS trace LDGSTS address stream; size=N*K*2 for 12288x12288 FP16.
compression_ratio=$COMPRESSION_RATIO
decoder_count=$DECODER_COUNT
decoder_cycles_per_record=$DECODER_CYCLES_PER_RECORD
record_bytes=$RECORD_BYTES
noc_model=A100-like hierarchical local interconnect; network_mode=2; icnt_arbiter_algo=3; 108 SM nodes; 40 memory modules; 7 GPC groups; 10 FBP groups; 2 coarse partitions; far-partition extra latency 94 cycles per direction calibrated to MICRO'24 real-GPU-NoC A100 near/far L2 latency.
stderr=kept_in_sim.err
extra_args=$*
MANIFEST

stdbuf -oL -eL "$BIN" \
  -trace "$TRACE" \
  -config "$CFG1" \
  -config "$CFG2" \
  -gpgpu_num_cta_barriers 64 \
  -gpgpu_bandcodec_enable "$BANDCODEC_ENABLE" \
  -gpgpu_bandcodec_weight_base "$WEIGHT_BASE" \
  -gpgpu_bandcodec_weight_bytes "$WEIGHT_BYTES" \
  -gpgpu_bandcodec_compression_ratio "$COMPRESSION_RATIO" \
  -gpgpu_bandcodec_record_bytes "$RECORD_BYTES" \
  -gpgpu_bandcodec_decoder_count "$DECODER_COUNT" \
  -gpgpu_bandcodec_decoder_cycles_per_record "$DECODER_CYCLES_PER_RECORD" \
  "$@" \
  > "$OUTDIR/sim_live.out" 2> "$OUTDIR/sim.err"
rc=$?
{
  echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "rc=$rc"
} >> "$OUTDIR/manifest.txt"
RC_RECORDED=1
exit "$rc"
