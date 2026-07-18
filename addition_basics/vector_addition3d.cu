#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define N (100 * 100 * 100) // vector size 1 million
#define BLOCK_SIZE_1D 1024 // number of threads per block for 1D grid
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8
// 16 * 8 * 8 = 1024 threads per block for 3D grid

// CPU function to add two vectors
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU function to add two vectors using 1D grid
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // one add, one multiple, one compare, one branch, one store
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        // one add, one store
    }
}
/*
 * Hypothesis (before measuring): the 3D kernel would be noticeably slower than 1D,
 * because each thread does more index arithmetic — 3 multiplies + 3 adds to compute
 * (i, j, k) and the linear index, versus one multiply-add in the 1D case.
 *
 * Result (same 1M-element workload on both): they are effectively identical.
 *   1D: ~0.078 ms   3D: ~0.090 ms   (ratio ~0.86x — within run-to-run noise)
 *
 * Why the hypothesis was wrong: vector addition is memory-bound. Each element does
 * one add against three global-memory accesses (read a, read b, write c), so the
 * kernel is limited by memory bandwidth, not compute. The extra integer ops in the
 * 3D indexing are hidden behind memory latency and barely register. Compute-side
 * differences only matter once a kernel is compute-bound — which vector add is not,
 * but matrix multiplication (next) is.
 */
__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    // 3 adds, 3 multiplies, 3 stores

    if (i < nx && j < ny && k < nz)
    {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz)
        {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void init_vector(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX;
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

    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a =(float*)malloc(size);
    h_b =(float*)malloc(size);
    h_c =(float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded
    
    // Define grid and block dimensions for 3d
    int nx = 100, ny = 100, nz = 100;
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z 
    );



    
    printf("Performing warm up runs...\n");
    for (int i = 0; i < 3; i ++){
        vector_add_cpu(h_a, h_b, h_c, N);
        vector_add_gpu_1d<<<num_blocks, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize(); // wait till the device has completed all the calculation
    }


    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation....\n");
    double cpu_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;
    
    // Benchmark GPU implementation
    printf("Benchmarking GPU 1D implementation....\n");
    double gpu_1d_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks, BLOCK_SIZE_1D>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 20.0;

    // verify
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++)
    {
        if(fabs(h_c[i] - h_c_gpu[i]) > 1e-5){
            correct = false;
            break;
        }
    }
    
    printf("Results are %s for 1d\n", correct ? "correct" : "incorrect");
    cudaMemset(d_c, 0, size);

    // Benchmark GPU implementation
    printf("Benchmarking GPU 3D implementation....\n");
    double gpu_3d_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 20.0;

    // verify
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    correct = true;
    for (int i = 0; i < N; i++)
    {
        if(fabs(h_c[i] - h_c_gpu[i]) > 1e-5){
            correct = false;
            break;
        }
    }
    
    printf("Results are %s for 3d\n", correct ? "correct" : "incorrect");

    
    
    printf("CPU average time: %f miliseconds\n", cpu_avg_time*1000);
    printf("GPU average 1D time: %f miliseconds\n", gpu_1d_avg_time*1000);
    printf("GPU average 3D time: %f miliseconds\n", gpu_3d_avg_time*1000);
    printf("Speedup(cpu vs gpu 1d): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup(cpu vs gpu 3d): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup(gpu 1d vs gpu 3d): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);
    
    free(h_a);
    free(h_b);  
    free(h_c);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}