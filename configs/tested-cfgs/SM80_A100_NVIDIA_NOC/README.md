# SM80_A100_NVIDIA_NOC

This configuration keeps the SM80_A100 HBM-side bandwidth settings and replaces the default local interconnect arbitration with the same A100-like hierarchical local interconnect model used by `SM80_A100_LPDDR5X_NVIDIA_NOC`.

Modeling scope:

- 108 SM-side nodes, 40 logical memory/channel groups, and 160 L2 subpartitions, matching the base SM80_A100 organization.
- 7 GPC groups, 10 FBP groups, and two coarse GPU partitions.
- Per-cycle GPC/FBP grant limits approximate hierarchical input/output speedup instead of a flat all-to-all xbar.
- Far-partition packets receive 94 extra cycles per request/reply network direction, matching the calibration note in the LPDDR5X_NVIDIA_NOC config.
- DRAM-side settings are restored to the base A100 profile (`1512 MHz` memory clock and 16-byte partition bus width).

This config is used as an HBM-bandwidth calibrated-NoC counterpart for BandCodec sensitivity runs.
