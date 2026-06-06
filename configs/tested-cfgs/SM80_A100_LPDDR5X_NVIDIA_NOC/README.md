# SM80_A100_LPDDR5X_NVIDIA_NOC

This configuration keeps the CUTLASS/LLM traces unchanged and replaces the
flat local interconnect arbitration with an A100-like hierarchical local
interconnect model for BandCodec sensitivity runs.

Modeling scope:

- 108 SM-side nodes, matching A100 SM count used by the SM80_A100 config.
- 40 logical memory/channel groups and 160 L2 subpartitions, matching the
  existing Accel-Sim A100 memory-side organization.
- 7 GPC groups and 10 FBP groups.
- Two coarse GPU partitions.
- Per-cycle GPC/FBP grant limits approximate hierarchical input/output
  speedup instead of a flat all-to-all xbar.
- Far-partition packets receive 94 extra cycles per request/reply network
  direction.  This calibrates the round-trip near/far L2-hit gap to the
  MICRO'24 real-GPU-NoC A100 observation: near accesses are V100-like
  (~212 cycles), while far-partition A100 accesses are around 400 cycles.

This is not the default Accel-Sim mesh/Intersim mode.  It is selected by
`-network_mode 2` and `-icnt_arbiter_algo 3`.
