#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>
 #include "cvc.h"

// Convert FP16 to FP32 (IEEE 754 half -> single precision)
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF) << 13;
    
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // Zero
        } else {
            // Denormalized
            exp = 127 - 15 + 1;
            while (!(mant & 0x00800000)) {
                mant <<= 1;
                exp--;
            }
            mant &= ~0x00800000;
            f = sign | (exp << 23) | mant;
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | mant;  // Inf or NaN
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | mant;
    }
    
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

void cvc_decompress_fp16(const uint16_t* src, float* dst, size_t n) {
    for (size_t i=0; i<n; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

void cvc_decompress_int8(const uint8_t* src, float* dst, float minv, float scale, size_t n) {
    for (size_t i=0; i<n; i++) {
        dst[i] = src[i] * scale + minv;
    }
}
