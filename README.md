# CUDA Fundamentals
 
Learning CUDA from first principles on an **NVIDIA RTX 3050 (Ampere, sm_86)**, building toward GPU microarchitecture and side-channel research. This repo tracks that progression: each kernel is written from scratch, benchmarked against a CPU baseline, and verified for correctness — with an emphasis on *measuring honestly*, not just getting code to run.
 
## Contents
 
| File | What it does |
|------|--------------|
| `addition_basics/vector_addition.cu` | 1D vector addition (10M elements), CPU vs GPU benchmark with correctness verification |
| `addition_basics/vector_addition3d.cu` | Same workload under 1D vs 3D thread indexing — tests whether index arithmetic costs anything |
| `matmul/naive_method.cu` | Naive matrix multiplication, CPU vs GPU — the compute-bound contrast to vector addition |
 
More kernels (tiled / shared-memory matrix multiplication, memory-coalescing and bank-conflict microbenchmarks) will be added as the work progresses.
 
## Build & run
 
```bash
nvcc -o ./01_vector_addition addition_basics/vector_addition.cu && ./01_vector_addition
nvcc -o ./02_vector_add_1d_vs_3d addition_basics/vector_addition3d.cu && ./02_vector_add_1d_vs_3d
nvcc -o ./03_matmul_naive matmul/naive_method.cu && ./03_matmul_naive
```
 
Requires the CUDA Toolkit and an NVIDIA GPU. Tested on an RTX 3050 (sm_86).
 
## 01 — Vector addition: a lesson in measuring honestly
 
Vector addition is the "hello world" of CUDA, but the interesting part here was **not** the kernel — it was catching a measurement bug in my own benchmark.
 
### The bug: a physically impossible speedup
 
My first benchmark reported a GPU time of ~0.004 ms and a **~9,000× speedup** over the CPU. That number is wrong, and a quick sanity check proves it: adding 10 million floats requires reading two input arrays and writing one output — roughly 120 MB of memory traffic. At the RTX 3050's memory bandwidth (~170–220 GB/s), moving that much data *must* take on the order of 0.5–0.7 ms. A measurement claiming 0.004 ms is faster than the hardware can physically move the bytes, so the measurement — not the GPU — is what's extraordinary.
 
The cause: CUDA kernel launches are **asynchronous**. `vector_add_gpu<<<...>>>` returns to the host immediately, before the GPU finishes. My timer stopped while the GPU was still computing, so I was timing the *launch*, not the *work*.
 
### The fix
 
Adding `cudaDeviceSynchronize()` **inside the timed region**, so the clock stops only after the GPU actually finishes:
 
```cuda
double start = get_time();
vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
cudaDeviceSynchronize();   // wait for the GPU before stopping the clock
double end = get_time();
```
 
After the fix, the GPU time rose to a realistic ~0.66 ms — right in the range the bandwidth estimate predicted, which confirms the corrected measurement is trustworthy.
 
### Results (RTX 3050, N = 10,000,000)
 
| Version | GPU time | Reported speedup | Valid? |
|---------|----------|------------------|--------|
| Buggy (no sync in timed region) | ~0.004 ms | ~9,000× | No — physically impossible |
| Fixed (sync inside timed region) | ~0.66 ms | ~62× | Yes — matches bandwidth estimate |
 
*(Numbers from my RTX 3050; they will vary by machine.)*
 
### The real lesson: vector addition is memory-bound
 
Even the correct ~62× is close to the ceiling for this kernel, and that's the point. Vector addition does **one** arithmetic operation per **three** global-memory accesses (read `a`, read `b`, write `c`). It is almost pure memory movement with negligible compute — a *memory-bound* kernel. The GPU can't show off because there's no arithmetic to parallelize; it's limited by how fast it can move bytes, not how fast it can calculate.
 
This is the key intuition to carry into matrix multiplication (added next), which is the opposite: each output element performs many multiply-accumulates on reused data — *compute-bound* — where the GPU's parallelism produces a far larger, legitimate speedup. The contrast between the two is the single most important performance lesson in GPU computing.
 
### Notes / honest limitations
- The benchmark times **compute only**; host↔device `cudaMemcpy` is done once before the loop and excluded. For an end-to-end "is the GPU worth it" question on a simple kernel, transfer cost often erases the win — worth keeping in mind.
- Timing uses `clock_gettime(CLOCK_MONOTONIC)` on the host around a synchronized kernel. Finer-grained on-device timing (CUDA events, `clock64()`) comes in later work.
---
 
## 02 — 1D vs 3D thread indexing: testing whether index arithmetic costs anything
 
Having established that vector addition is memory-bound (see 01), I wanted to test a specific hypothesis: **does the extra index arithmetic in a 3D thread layout make the kernel measurably slower than a 1D layout, for the same work?**
 
### Hypothesis
 
A 1D kernel computes its element index with one multiply-add (`blockIdx.x * blockDim.x + threadIdx.x`). A 3D kernel computes three indices `(i, j, k)` and then flattens them (`i + j*nx + k*nx*ny`) — several more integer multiplies and adds per thread. My prediction was that this would make the 3D version noticeably slower.
 
### Method (and two measurement bugs I had to fix first)
 
Getting a *fair* comparison took three iterations, and the mistakes are worth recording because they're easy to make and easy to miss:
 
1. **Wrong kernel in the timed loop.** My first 3D benchmark accidentally called the 1D kernel (copy-paste), so it was silently comparing 1D against itself. The reported "speedup" was meaningless until I fixed the call.
2. **Mismatched workload.** I initially ran the 1D kernel over 10M elements but the 3D kernel over a 100×100×100 = 1M grid — one-tenth the work. Any "speedup" there just measured the size difference, not the kernels. Fixed by running both over the same 1M elements.
3. **Unsound verification.** Both kernels wrote to the same output buffer, and I verified only once at the end — so the check only ever validated whichever kernel ran last; a broken first kernel would have passed unnoticed. Fixed by verifying each kernel independently and zeroing the buffer (`cudaMemset`) between them.
All timing is done with `cudaDeviceSynchronize()` inside the timed region (see 01 for why), same 1M-element workload on both kernels.
 
### Result (RTX 3050, N = 1,000,000)
 
| Kernel | Time | Speedup vs CPU |
|--------|------|----------------|
| 1D | ~0.078 ms | ~51× |
| 3D | ~0.090 ms | ~44× |
 
1D-vs-3D ratio: **~0.86×** — i.e. effectively identical, within run-to-run noise.
 
### Interpretation
 
The hypothesis was wrong, and *why* it's wrong is the useful part. Because vector addition is memory-bound, the kernel is limited by how fast the GPU moves bytes, not by arithmetic. The extra integer operations in the 3D indexing are hidden behind memory latency and barely affect runtime. Index-arithmetic cost would only show up in a *compute-bound* kernel — which is exactly what matrix multiplication (next) will demonstrate from the other direction.
 
### Takeaway
The recurring lesson across 01 and 02 is not about CUDA syntax — it's about measurement discipline: match the workload on both sides, verify what you actually measured, and sanity-check results against physical expectation. These habits matter more as the measurements get noisier.
 
---
 
## 03 — Naive matrix multiplication: compute-bound parallelism
 
After vector addition (memory-bound), matrix multiplication is the contrast case: it is **compute-bound**, so the GPU's massive parallelism produces a far larger speedup.
 
### Hypothesis
 
Matmul does K multiply-accumulates per output element while reading only ~2K values and writing one, so arithmetic intensity (~K/2) is high — the opposite of vector addition, which did one add per three memory accesses. Prediction: the GPU-vs-CPU speedup should be dramatically larger than the ~50× seen for vector addition.
 
**Confirmed:** naive matmul (256×512 @ 512×256) runs **~486× faster** than the CPU triple-loop on an RTX 3050, versus ~50× for vector addition. The contrast between the two — memory-bound vs compute-bound — is the core lesson.
 
### Why the GPU wins here
 
Each output element of C is an independent dot product of a row of A and a column of B. The GPU assigns one thread per output element, so all M×N dot products are computed concurrently across its cores, instead of sequentially as on the CPU.
 
A note on complexity, since it's easy to state wrong: matmul is O(n³) **work** — n² output elements, each an O(n) dot product. The GPU does **not** reduce this work; it runs the n² independent dot products in parallel, cutting wall-clock **time** while the total number of operations stays O(n³). Work and time are different things; parallelism improves the second, not the first.
 
This kernel is intentionally naive: every thread re-reads its row of A and column of B from global memory with no reuse. That wasted bandwidth is what the tiled / shared-memory version (next) reclaims. Baseline first, on purpose.
 
### Verifying correctness
 
Verification uses a `1e-3` tolerance, not the `1e-5` used for earlier kernels. Matmul sums K=512 floating-point products, and CPU and GPU accumulate them in different orders, so rounding differs slightly even when both results are correct — a `1e-5` tolerance would report false mismatches. A small but real lesson in floating-point non-determinism.
 
The kernel was also stress-tested with **M ≠ N**: equal dimensions can mask a row/column transposition bug in the indexing, so forcing them unequal confirms the indexing is genuinely correct rather than coincidentally correct on a square matrix.
 
### Takeaway
 
Two kernels, two regimes: vector addition is memory-bound (bandwidth-limited, ~50× speedup) and matrix multiplication is compute-bound (parallelism-limited, ~486× speedup). Recognising which regime a kernel is in is the first step in reasoning about GPU performance — and the naive matmul here is the baseline the optimised, shared-memory version is measured against next.