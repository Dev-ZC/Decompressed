#include <vector>
#include <cstdint>
#include <cmath>

void cvc_decompress_fp16(const uint16_t* src, float* dst, size_t n) {
    for (size_t i=0; i<n; i++) {
        dst[i] = __half_as_float(src[i]); // convert fp16 -> float32
    }
}

void cvc_decompress_int8(const uint8_t* src, float* dst, float minv, float scale, size_t n) {
    for (size_t i=0; i<n; i++) {
        dst[i] = src[i] * scale + minv;
    }
}
