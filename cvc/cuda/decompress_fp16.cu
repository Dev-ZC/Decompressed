#include "../cpu/cvc.h"
#include <cuda_runtime.h>

__global__ void decompress_fp16_kernel(const uint16_t* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half_as_float(src[idx]);
    }
}

void cvc_decompress_fp16_cuda(const uint16_t* src, float* dst, size_t n) {
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    decompress_fp16_kernel<<<blocks, threads>>>(src, dst, n);
    cudaDeviceSynchronize();
}
