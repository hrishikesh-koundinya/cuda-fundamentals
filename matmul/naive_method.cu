#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 // Number of rows in A and C matrices
#define K 512 // Number of columns in A and rows in B matrices
#define N 128 // Number of columns in B and C matrices
#define BLOCK_SiZE 32
/*
Example 3x2 @ 2x4 = 3x4 -> (M x K) @ (K x N) = (M x N)
A = [[1, 2], 
     [3, 4], 
     [5, 6
B = [[7, 8, 9, 10],
     [11, 12, 13, 14
C = A * B = [[1*7 + 2*11, 1*8 + 2*12, 1*9 + 2*13, 1*10 + 2*14],
             [3*7 + 4*11, 3*8 + 4*12, 3*9 + 4*13, 3*10 + 4*14],
             [5*7 + 6*11, 5*8 + 6*12, 5*9 + 6*13, 5*10 + 6*14
C = [[29, 32, 35, 38],
     [65, 72, 79, 86],
     [101, 112, 123, 134]]
*/

// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
            {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
/*
Hypothesis (before measuring): matrix multiplication should show a FAR larger GPU
speedup than vector addition did (~50x), because it is compute-bound rather than
memory-bound. Each output element performs K multiply-accumulates while reading
only ~2K values and writing one, so arithmetic intensity (~K/2) is high and the
GPU's parallel cores are the bottleneck-breaker, not memory bandwidth.

Note on complexity: matmul is O(n^3) *work* -- n^2 output elements, each an O(n)
dot product. The GPU does NOT reduce this work. It runs the n^2 independent dot
products concurrently, cutting wall-clock TIME while total operations stay O(n^3).
Work (total ops) and time (wall-clock given parallelism) are different things;
the GPU improves the second, not the first.

This naive kernel is intentionally unoptimized: every thread re-reads its row of A
and column of B from global memory, with no reuse. That wasted bandwidth is what
the tiled / shared-memory version (next) will reclaim. Baseline first, on purpose.
*/
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int l = 0; l < k; l++)
        {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        mat[i] = (float)rand()/RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *h_a, *h_b, *h_c, *h_c_gpu;
    float *d_a, *d_b, *d_c;

    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);


    // Allocate host memory
    h_a =(float*)malloc(size_a);
    h_b =(float*)malloc(size_b);
    h_c =(float*)malloc(size_c);
    h_c_gpu = (float*)malloc(size_c);

    srand(time(NULL));
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    // Allocate device memory
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SiZE, BLOCK_SiZE);
    dim3 gridDim(
        (N + BLOCK_SiZE - 1 ) / BLOCK_SiZE,
        (M + BLOCK_SiZE - 1 ) / BLOCK_SiZE
    );
    
    printf("Performing warm up runs...\n");
    for (int i = 0; i < 3; i ++){
        matmul_cpu(h_a, h_b, h_c, M, N, K);
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        cudaDeviceSynchronize(); // wait till the device has completed all the calculation
    }


    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation....\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        matmul_cpu(h_a, h_b, h_c, M, N, K);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation....\n");
    double gpu_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU average time: %f miliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f miliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    // Verify correctness
    cudaMemcpy(h_c_gpu, d_c, size_c, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++)
    {
        if (fabs(h_c[i] - h_c_gpu[i]) > 1e-3)
        {
            correct = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c[i], h_c_gpu[i]);
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    
    free(h_a);
    free(h_b);  
    free(h_c);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}