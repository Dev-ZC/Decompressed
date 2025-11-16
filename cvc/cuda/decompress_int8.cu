#include "../cpu/cvc.h"
#include <cuda_runtime.h>

__global__ void decompress_int8_kernel(const uint8_t* src, float* dst, float minv, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx] * scale + minv;
    }
}

void cvc_decompress_int8_cuda(const uint8_t* src, float* dst, float minv, float scale, size_t n) {
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    decompress_int8_kernel<<<blocks, threads>>>(src, dst, minv, scale, n);
    cudaDeviceSynchronize();
}
