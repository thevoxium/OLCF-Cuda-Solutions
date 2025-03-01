#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define N 10000000
#define threads_per_block 256

// GPU kernels
__global__ void add(int *a, int *b, int *c, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n){
        c[idx] = a[idx] + b[idx]; 
    }
}

__global__ void mul(int *a, int *b, int *d, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n){
        d[idx] = a[idx] * b[idx]; 
    }
}

__global__ void scale(int *a, int *e, int scalar, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n){
        e[idx] = a[idx] * scalar; 
    }
}

// CPU implementations for comparison
void cpu_add(int *a, int *b, int *c, int n) {
    for(int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void cpu_mul(int *a, int *b, int *d, int n) {
    for(int i = 0; i < n; i++) {
        d[i] = a[i] * b[i];
    }
}

void cpu_scale(int *a, int *e, int scalar, int n) {
    for(int i = 0; i < n; i++) {
        e[i] = a[i] * scalar;
    }
}

// Function to verify results
bool verify_results(int *a, int *b, int n) {
    for(int i = 0; i < n; i++) {
  if(a[i] != b[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main(){
  int *a, *b, *c_gpu, *d_gpu, *e_gpu;
    int *c_cpu, *d_cpu, *e_cpu;
    int scalar = 5;
    int *d_a, *d_b, *d_c, *d_d, *d_e;
    clock_t start, end;
    double cpu_time, gpu_time;
    
    // Allocate host memory
    a = new int[N];
    b = new int[N];
    c_gpu = new int[N];
    d_gpu = new int[N];
    e_gpu = new int[N];
    c_cpu = new int[N];
    d_cpu = new int[N];
    e_cpu = new int[N];
    
    // Initialize data
    for(int i=0; i<N; i++){
        a[i] = i % 100; // Using modulo to keep values small
        b[i] = (i + 5) % 100;
    }
    
    // CPU implementation timing
    start = clock();
    cpu_add(a, b, c_cpu, N);
    cpu_mul(a, b, d_cpu, N);
    cpu_scale(a, e_cpu, scalar, N);
    end = clock();
    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", cpu_time);
    
    // Allocate device memory
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));
    cudaMalloc(&d_d, N*sizeof(int));
    cudaMalloc(&d_e, N*sizeof(int));
    cudaCheckErrors("malloc failure");

    // GPU implementation timing
    start = clock();
    
    // Copy data to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernels
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);
    mul<<<blocks, threads_per_block>>>(d_a, d_b, d_d, N);
    scale<<<blocks, threads_per_block>>>(d_a, d_e, scalar, N);
    
    // Copy results back
    cudaMemcpy(c_gpu, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_gpu, d_d, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(e_gpu, d_e, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    end = clock();
    gpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("GPU time: %f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);

    // Verify results
    printf("Vector addition results match: %s\n", 
           verify_results(c_cpu, c_gpu, N) ? "Yes" : "No");
    printf("Vector multiplication results match: %s\n", 
           verify_results(d_cpu, d_gpu, N) ? "Yes" : "No");
    printf("Vector scaling results match: %s\n", 
           verify_results(e_cpu, e_gpu, N) ? "Yes" : "No");

    // Print first few elements for visual confirmation
    printf("\nFirst 5 elements:\n");
    printf("Index\tA\tB\tA+B\tA*B\t%d*A\n", scalar);
    for (int i = 0; i < 5 && i < N; i++) {
        printf("%d\t%d\t%d\t%d\t%d\t%d\n", 
               i, a[i], b[i], c_gpu[i], d_gpu[i], e_gpu[i]);
    }

    // Free memory
    delete[] a;
    delete[] b;
    delete[] c_gpu;
    delete[] d_gpu;
    delete[] e_gpu;
    delete[] c_cpu;
    delete[] d_cpu;
    delete[] e_cpu;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
    
    return 0;
}
