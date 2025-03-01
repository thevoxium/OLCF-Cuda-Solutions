#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define img_w 1024
#define img_h 1024
#define kernel_size 3
#define kernel_radius (kernel_size / 2)

#define block_size_x 8
#define block_size_y 8


#define cudaCheckErrors(call){ \
cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

void init_img(float *img, int w, int h){
  for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = x - w/2;
            float dy = y - h/2;
            float distance = sqrtf(dx*dx + dy*dy);
            img[y*w + x] = 128.0f + 127.0f * sinf(distance/10.0f);
        }
    }
}

void initializeGaussianKernel(float* kernel, int kernelSize) {
    float sigma = kernelSize/6.0f;
    float sum = 0.0f;
    int radius = kernelSize / 2;
    
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            int idx = (y + radius) * kernelSize + (x + radius);
            kernel[idx] = expf(-(x*x + y*y) / (2*sigma*sigma));
            sum += kernel[idx];
        }
    }
    
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel[i] /= sum;
    }
}



__global__ void globalMemoryConv(float *img, float *out, float *kernel, int w, int h, int ks){
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x < w && y < h){
    float sum = 0.0f;
    int radius = kernel_size / 2;
    for (int ky = -radius; ky<=radius; ky++){
      for (int kx = -radius; kx <= radius; kx++){
        int ix = kx + x;
        int iy = ky + y;

        ix = max(0, min(w-1, ix));
        iy = max(0, min(h-1, iy));

        float pixelValue = img[iy * w + ix];
        float kernelValue = kernel[(ky + radius) * ks + (kx + radius)];
                
        sum += pixelValue * kernelValue;
      }
    }
    out[y*w + x] = sum;
  }
}


__global__ void sharedMemoryConv(float *img, float *out, float *kernel, int w, int h, int ks){

  extern __shared__ float sharedMem[];
  float (*sharedData)[block_size_x + kernel_size - 1] = 
        (float (*)[block_size_x + kernel_size - 1])sharedMem;
    
  int radius = ks / 2;
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int lx = threadIdx.x;
  int ly = threadIdx.y;
  
  int x, y;
  for (int j = ly; j < blockDim.y + 2*radius; j += blockDim.y) {
      for (int i = lx; i < blockDim.x + 2*radius; i += blockDim.x) {
          y = blockIdx.y * blockDim.y + j - radius;
          x = blockIdx.x * blockDim.x + i - radius;
          
          y = max(0, min(h - 1, y));
          x = max(0, min(w - 1, x));
          
          sharedData[j][i] = img[y * w + x];
      }
  }

  __syncthreads();


  if (gx < w && gy < h) {
      float sum = 0.0f;
        
      for (int ky = 0; ky < ks; ky++) {
        for (int kx = 0; kx < ks; kx++) {
          float pixelValue = sharedData[ly + ky][lx + kx];
          float kernelValue = kernel[ky * ks + kx];
          sum += pixelValue * kernelValue;
        }
      }
        
      out[gy * w + gx] = sum;
  }
}


int main(){
  size_t img_size = img_w * img_h * sizeof(float);
  size_t k_size = kernel_size * kernel_size * sizeof(float);

  float *h_img = new float[img_w * img_h];
  float *out_naive = new float[img_w * img_h];
  float *out_shared = new float[img_w * img_h];
  float *kernel = new float[kernel_size * kernel_size];

  init_img(h_img, img_w, img_h);
  initializeGaussianKernel(kernel, kernel_size);


  float *d_img, *d_out_naive, *d_out_shared, *d_kernel;
  cudaCheckErrors(cudaMalloc((void**)&d_img, img_size));
  cudaCheckErrors(cudaMalloc((void**)&d_out_naive, img_size));
  cudaCheckErrors(cudaMalloc((void**)&d_out_shared, img_size));
  cudaCheckErrors(cudaMalloc((void**)&d_kernel, k_size));


  cudaCheckErrors(cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice));
  cudaCheckErrors(cudaMemcpy(d_kernel, kernel, k_size, cudaMemcpyHostToDevice));

  dim3 blockDim(block_size_x, block_size_y);
  dim3 gridSize((img_w+blockDim.x-1)/blockDim.x, (img_h+blockDim.y-1)/blockDim.y);
  
  clock_t start_global, end_global;

  start_global = clock();

  globalMemoryConv<<<gridSize, blockDim>>>(d_img, d_out_naive, d_kernel, img_w, img_h, kernel_size);
  cudaCheckErrors(cudaDeviceSynchronize());

  end_global = clock();
  double time_taken_global = (double) (end_global - start_global) / CLOCKS_PER_SEC;

  clock_t start_shared, end_shared;

  start_shared = clock();
  size_t sharedMemSize = (block_size_y + kernel_size - 1) * (block_size_x + kernel_size - 1) * sizeof(float);
  sharedMemoryConv<<<gridSize, blockDim, sharedMemSize>>>(d_img, d_out_shared, d_kernel, img_w, img_h, kernel_size);
  cudaCheckErrors(cudaDeviceSynchronize());

  end_shared = clock();
  double time_taken_shared = (double) (end_shared - start_shared) / CLOCKS_PER_SEC;

  cudaCheckErrors(cudaMemcpy(out_naive, d_out_naive, img_size, cudaMemcpyDeviceToHost));
  cudaCheckErrors(cudaMemcpy(out_shared, d_out_shared, img_size, cudaMemcpyDeviceToHost));

  float maxDif = 0.0f;
  for (int i=0; i<img_w * img_h; i++){
    float dif = fabs(out_shared[i] - out_naive[i]);
    maxDif = max(maxDif, dif);
  }

  printf("2D Convolution Performance Comparison (Image size: %dx%d, Kernel size: %dx%d)\n", 
         img_w, img_h, kernel_size, kernel_size);
  printf("Global Memory Implementation: %.4f ms\n", time_taken_global);
  printf("Shared Memory Implementation: %.4f ms\n", time_taken_shared);
  printf("Speedup: %.2fx\n", time_taken_global / time_taken_shared);
  printf("Maximum result difference: %f\n", maxDif);
  printf("Results match: %s\n", (maxDif < 1e-5) ? "Yes" : "No");

  delete[] h_img;
  delete[] out_naive;
  delete[] out_shared;
  delete[] kernel;

  cudaFree(d_img);
  cudaFree(d_out_shared);
  cudaFree(d_kernel);
  cudaFree(d_out_naive);

  return 0;
}
