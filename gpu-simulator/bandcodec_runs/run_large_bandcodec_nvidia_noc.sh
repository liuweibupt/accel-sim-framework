#!/usr/bin/env bash
set -Eeuo pipefail
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=/tmp/gpgpu-local-tools:${PATH}
export LIBRARY_PATH=/tmp/gpgpu-local-lib:${LIBRARY_PATH:-}
set +u
source /work/accel-sim-framework/gpu-simulator/setup_environment.sh >/tmp/accelsim_setup_large_bandcodec_nvidia_noc.log 2>&1
set -Eeuo pipefail

OUTDIR=/work/accel-sim-framework/gpu-simulator/bandcodec_runs/large_lpddr5x_bandcodec_nvidia_noc_$(date -u '+%Y%m%d-%H%M%S')
mkdir -p "$OUTDIR"
TRACE=/work/accel-sim-framework/.worktrees/modal-a100-cutlass-trace/modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/traces/reprocessed/kernelslist.g
CFG1=/work/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100_LPDDR5X_NVIDIA_NOC/gpgpusim.config
CFG2=/work/accel-sim-framework/gpu-simulator/configs/tested-cfgs/SM80_A100_LPDDR5X_NVIDIA_NOC/trace.config
BIN=/work/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out
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
trace=$TRACE
binary=$BIN
gpgpusim_config=$CFG1
trace_config=$CFG2
weight_base=0x00002ae461000000
weight_bytes=301989888
weight_bytes_hex=0x12000000
weight_range_method=B-matrix base inferred from large CUTLASS trace LDGSTS address stream; size=N*K*2 for 12288x12288 FP16.
compression_ratio=4
decoder_count=100
decoder_cycles_per_record=4
record_bytes=256
noc_model=A100-like hierarchical local interconnect; network_mode=2; icnt_arbiter_algo=3; 108 SM nodes; 40 memory modules; 7 GPC groups; 10 FBP groups; 2 coarse partitions; far-partition extra latency 94 cycles per direction calibrated to MICRO'24 real-GPU-NoC A100 near/far L2 latency.
stderr=kept_in_sim.err
MANIFEST
stdbuf -oL -eL "$BIN" \
  -trace "$TRACE" \
  -config "$CFG1" \
  -config "$CFG2" \
  -gpgpu_num_cta_barriers 64 \
  -gpgpu_bandcodec_enable 1 \
  -gpgpu_bandcodec_weight_base 0x00002ae461000000 \
  -gpgpu_bandcodec_weight_bytes 301989888 \
  -gpgpu_bandcodec_compression_ratio 4 \
  -gpgpu_bandcodec_record_bytes 256 \
  -gpgpu_bandcodec_decoder_count 100 \
  -gpgpu_bandcodec_decoder_cycles_per_record 4 \
  > "$OUTDIR/sim_live.out" 2> "$OUTDIR/sim.err"
rc=$?
{
  echo "end_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "rc=$rc"
} >> "$OUTDIR/manifest.txt"
RC_RECORDED=1
exit $rc
