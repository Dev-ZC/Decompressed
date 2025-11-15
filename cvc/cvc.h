#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// CPU decompression (fallback)
void cvc_decompress_fp16(const uint16_t* src, float* dst, size_t n);
void cvc_decompress_int8(const uint8_t* src, float* dst, float minv, float scale, size_t n);

// GPU decompression helpers (CUDA)
void cvc_decompress_fp16_cuda(const uint16_t* src, float* dst, size_t n);
void cvc_decompress_int8_cuda(const uint8_t* src, float* dst, float minv, float scale, size_t n);

#ifdef __cplusplus
}
#endif
