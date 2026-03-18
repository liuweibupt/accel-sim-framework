# Small GEMM Results Reporting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Save the verified small GEMM results into stable Markdown and CSV outputs while preserving visibility into the ongoing large replay.

**Architecture:** Reuse existing replay logs and trace artifacts as the source of truth, extend the existing summary document, and add a single CSV export in `modal/results/`. Keep large-run status narrative-only until replay finishes.

**Tech Stack:** Markdown, CSV, shell/grep verification, Git.

---

### Task 1: Record the reporting design

**Files:**
- Create: `docs/plans/2026-03-18-small-gemm-results-reporting-design.md`
- Create: `docs/plans/2026-03-18-small-gemm-results-reporting.md`

**Step 1: Save the approved design**
- Write the design rationale, chosen approach, and data fields.

**Step 2: Save the implementation plan**
- Write the concrete file-level implementation steps.

**Step 3: Verify the docs exist**

Run: `ls docs/plans/2026-03-18-small-gemm-results-reporting*.md`
Expected: both files listed.

### Task 2: Capture the verified small-run metrics

**Files:**
- Read: `modal/artifacts/fp16-512x512x512/sim_run/sim.out`
- Read: `modal/artifacts/bf16-512x512x512/sim_run/sim.out`
- Read: `modal/artifacts/fp16-512x512x512/sim_run_lpddr5x/sim.out`
- Read: `modal/artifacts/bf16-512x512x512/sim_run_lpddr5x/sim.out`
- Read: `modal/artifacts/fp16-512x512x512/traces/*.traceg`
- Read: `modal/artifacts/bf16-512x512x512/traces/*.traceg`

**Step 1: Extract replay metrics**
- Read cycles, instructions, IPC, and successful exit markers from the four replay logs.

**Step 2: Extract tensor-core evidence**
- Read HMMA opcode counts from the small baseline traces.

**Step 3: Derive replay time**
- Compute time in microseconds from cycles at 1410 MHz core clock.

### Task 3: Update the stable result outputs

**Files:**
- Modify: `modal/results/summary.md`
- Modify: `modal/README.md`
- Create: `modal/results/small_gemm_results.csv`

**Step 1: Extend the Markdown summary**
- Add a compact saved-results section/table for the four small cases.
- Add a brief status note that the large `2048x12288x12288` replay is still running / in progress.

**Step 2: Write the CSV export**
- Add one row per verified small case with config, metrics, and artifact paths.

**Step 3: Refresh the README pointers**
- Mention the new CSV path alongside the summary.

### Task 4: Verify and commit

**Files:**
- Verify: `modal/results/summary.md`
- Verify: `modal/results/small_gemm_results.csv`
- Verify: `modal/README.md`

**Step 1: Inspect the generated outputs**

Run: `sed -n '1,220p' modal/results/summary.md && echo '---' && cat modal/results/small_gemm_results.csv`
Expected: values match the replay logs.

**Step 2: Check git status**

Run: `git status --short`
Expected: only intended files changed.

**Step 3: Commit**

```bash
git add docs/plans/2026-03-18-small-gemm-results-reporting-design.md \
        docs/plans/2026-03-18-small-gemm-results-reporting.md \
        modal/results/summary.md modal/results/small_gemm_results.csv modal/README.md
git commit -m "docs: save small GEMM replay results"
```

**Step 4: Push**

```bash
git push
```
