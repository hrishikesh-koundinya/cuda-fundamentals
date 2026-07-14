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
