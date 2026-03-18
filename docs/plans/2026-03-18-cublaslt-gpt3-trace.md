# cuBLASLt GPT-3 GEMM Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a cuBLASLt-based large GEMM runner and use it on Modal A100-80GB to collect a GPT-3-style GEMM trace for local Accel-Sim replay.

**Architecture:** A new `modal/cublaslt_runner/` target will launch one large GEMM through cuBLASLt. Existing Modal tracer scripts will be extended minimally to accept an alternate binary path and output location. Resulting traces will be downloaded locally and replayed with the existing A100 config without committing oversized traces to git.

**Tech Stack:** CUDA 12.8, cuBLASLt, Modal, NVBit tracer, Accel-Sim.

---
