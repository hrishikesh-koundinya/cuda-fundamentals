## 01 - 1D vector addition: comparing speeds of cpu and gpu for vector addition

// Benchmark CPU implementation
    printf("Benchmarking CPU implementation....\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c, N);
        double end_time = get_time();          // no cudaDeviceSynchronize here — CPU work is already synchronous
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation....\n");
    double gpu_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();               // <-- THE FIX: wait for GPU inside the timed region
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

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

