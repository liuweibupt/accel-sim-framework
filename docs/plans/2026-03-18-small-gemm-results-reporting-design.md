# Small GEMM Results Reporting Design

## Goal
Persist the already-verified small GEMM replay results in both human-readable Markdown and machine-readable CSV, while keeping the ongoing large replay status visible in the existing Modal results summary.

## Scope
- Reuse the existing verified 512x512x512 FP16/BF16 baseline and LPDDR5X replay outputs.
- Keep the canonical narrative summary in `modal/results/summary.md`.
- Add a compact tabular export at `modal/results/small_gemm_results.csv`.
- Optionally refresh `modal/README.md` so it points to the CSV in addition to the Markdown summary.
- Do not add large raw trace artifacts to git.

## Approach Options

### Option 1: Update the existing summary + add one CSV (recommended)
- Pros: minimal churn, consistent with current layout, easy for humans and scripts.
- Cons: `summary.md` keeps growing.

### Option 2: Add a dedicated small-results Markdown file + CSV
- Pros: cleaner separation between narrative and tabular data.
- Cons: duplicates context and adds another entry point.

### Option 3: CSV only
- Pros: smallest change.
- Cons: worse for quick inspection and loses already-written narrative context.

## Recommended Design
Use Option 1. Keep `modal/results/summary.md` as the main human-readable report, add a concise "saved artifacts" style CSV for the four verified small-kernel replay cases, and update `modal/README.md` to advertise both outputs.

## Data Model
Each CSV row should capture:
- precision mode (`fp16` / `bf16`)
- accumulation mode (`fp32`)
- shape (`512x512x512`)
- config (`SM80_A100` / `SM80_A100_LPDDR5X`)
- modeled bandwidth (`1935.36` / `819.2` GB/s)
- replay metrics (`gpu_tot_sim_cycle`, `gpu_tot_sim_insn`, `gpu_tot_ipc`)
- derived time in microseconds at 1410 MHz core clock
- tensor-core evidence (`HMMA`, count 65536 for baseline traces)
- artifact paths

## Verification
- Re-read the replay logs and trace files rather than transcribing from memory.
- Confirm the CSV matches the values already reported in `summary.md`.
- Keep the large replay status marked as in progress until `gpu_tot_sim_cycle` and exit markers are available.
