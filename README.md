# 🧮 Sparse Matrix, Very Thin Sparse Matrix
GPU Computing Project – CMPS 224 (AUB)

**Authors:** Mohammad Khaled Charaf · Mohammad Ayman Charaf · Jad Abou Hawili  
**Supervisor:** Dr. Izzat El Hajj

---

## Overview
This project implements and progressively optimizes **sparse × thin-sparse matrix multiplication** on **NVIDIA GPUs (CUDA)**. We move from a sequential baseline to warp-level kernels that improve **parallelism, memory locality, and synchronization**, culminating in strong speedups over the baseline.

**Why it matters:** Sparse ops are ubiquitous in graph analytics, PDE solvers, ML (e.g., NLP/recommenders), and scientific computing. Efficient sparse multiplication reduces both compute and memory overhead.

---

## Objectives
- Implement a correct baseline for sparse × thin-sparse multiplication.
- Design **five CUDA kernels** with increasing sophistication:
  - Thread-per-row → Block-per-row → Shared-memory privatization → Warp-level coalescing → Multi-warp write-back.
- Benchmark and analyze performance, occupancy, and bottlenecks.

---

## Methods & Milestones

### 1) Sequential Baseline
- Start with **COO × CSR → COO**, then switch to **CSR × CSR** for faster row access and reduced memory overhead.
- Per-row temporary accumulators replace hashmap merges.

### 2) `kernel0.cu` — Thread per Row
- Each thread computes a row of **A (CSR)** against **B (CSR)**.
- Column-chunking controls memory usage; uses `atomicAdd` to write out.
- **Limitation:** Under-parallelization and load imbalance on uneven sparsity.

### 3) `kernel1.cu` — Block per Row + Shared Accumulation
- One **thread block per row**; threads collaborate via **shared memory** to accumulate column chunks.
- Fewer global memory reads; still uses `atomicAdd` on a global output counter.

### 4) `kernel2.cu` — Shared Counter (Privatization)
- Add a **block-local shared counter** to batch output reservation:
  - Threads increment a **shared** counter.
  - **One** global `atomicAdd` per block reserves a contiguous region.
  - Threads write directly into their reserved slice.
- **Result:** Drastically lower contention on the global counter.

### 5) `kernel3.cu` & `kernel4.cu` — Warp-Level Optimization
- **Warp-per-element**: a warp collaboratively processes one nonzero of A’s row, traversing B’s corresponding row with **coalesced** loads.
- Use warp intrinsics: `__ballot_sync`, `__popc`, `__shfl_sync` to:
  - Compute compacted write positions.
  - Perform **one atomicAdd per warp per sub-chunk**.
- `kernel3.cu`: first warp writes back (cap at 32 columns per chunk).  
- `kernel4.cu`: **multi-warp write-back** scales to larger column chunks (e.g., 512), utilizing all warps in the block.

---

## Results (Summary)
- **Kernel1–4** substantially outperform the baseline; occupancy increases from ~29% (baseline) to ~68–72% (best).
- **Kernel3** provides the strongest balance of compute utilization and memory throughput.
- **Kernel4** improves small/medium cases further; for very large datasets, global memory traffic to **B** becomes the main bottleneck.

> **Key bottleneck:** non-coalesced, repeated global reads from the second matrix (CSR).  
> **Future work:** data reordering/tiling for better locality, hybrid sparse formats, cache-aware traversal.

---

## Build & Run

### Prerequisites
- CUDA Toolkit (nvcc), NVIDIA GPU (≥ 8 GB recommended), C++17 compiler.


---

---
> 📘 For detailed explanations, implementation notes, and performance analysis of all milestones, please refer to the [Final Report](./Final-Report.pdf).


