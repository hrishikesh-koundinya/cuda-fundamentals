#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


#define N 10000000 // vector size 10 million
#define BLOCK_SIZE 256 // number of threads per block

// Example
// A = [1, 2, 3, 4, 5]
// B = [10, 20, 30, 40, 50]
// C = A + B = [11, 22, 33, 44, 55]

void init_vector(float *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX;
    }
}


// CPU function to add two vectors
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU functon to add two vectors
/*
the thread id obtained using threadIdx.x or such is local, 
meaning its the local position of that thread to the block.
this can be only useful when there is only one block which typically can hold up to 1024 threads.
In cases where there are more than one block this is useless, 
therefore we need to calculate the block's position first then add the local thread index
this gives the global thread id meaning the absolute position of the thread location in the whole memory layour
*/

__global__ void vector_add_gpu(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
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
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 
    
    printf("Performing warm up runs...\n");
    for (int i = 0; i < 3; i ++){
        vector_add_cpu(h_a, h_b, h_c, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
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
    printf("Benchmarking GPU implementation....\n");
    double gpu_total_time = 0.0;
    for(int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    
    
    printf("CPU average time: %f miliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f miliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);
    
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
