# CUDA PTX Toolchain Error - Fix Documentation

## Problem

The comprehensive CUDA test was failing with:
```
CUDA error: the provided PTX was compiled with an unsupported toolchain
```

While the simple test appeared to work, but the error was actually **asynchronous** and only reported on subsequent CUDA operations.

## Root Cause

The issue was in `CMakeLists.txt` with two problematic settings:

### 1. **Native Architecture Only**
```cmake
set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
```
- Compiles only for the GPU present at **build time**
- If runtime GPU/CUDA version differs → PTX error
- Not portable across different GPU models

### 2. **Separable Compilation Disabled**
```cmake
CUDA_SEPARABLE_COMPILATION OFF
```
- No intermediate PTX generated
- Less portable across CUDA versions

### 3. **Asynchronous Error Reporting**
CUDA errors are reported on the *next* CUDA operation, not when they occur:
- Simple test: Runs once, exits before error check → appears to work
- Comprehensive test: Calls `torch.cuda.synchronize()` → forces error reporting

## Solution

### Changes Made

1. **Multi-architecture compilation** instead of "native":
   ```cmake
   # CUDA 12.0+: Support V100, T4, A100, RTX 30xx/40xx, H100
   set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
   
   # CUDA 11.1+: Support V100, T4, A100, RTX 30xx
   set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
   
   # CUDA < 11.1: Support V100, T4
   set(CMAKE_CUDA_ARCHITECTURES "70;75")
   ```

2. **Enable separable compilation**:
   ```cmake
   CUDA_SEPARABLE_COMPILATION ON
   CUDA_RESOLVE_DEVICE_SYMBOLS ON
   ```

### Rebuild Instructions

1. **Clean previous build**:
   ```bash
   pip uninstall -y decompressed
   rm -rf build/ _skbuild/ dist/ *.egg-info
   ```

2. **Rebuild with new settings**:
   ```bash
   pip install -e . -v
   ```

3. **Verify the fix**:
   ```python
   from decompressed import load_cvc, pack_cvc
   import numpy as np
   import torch
   
   # Create test data
   vectors = np.random.randn(1000, 768).astype(np.float32)
   pack_cvc(vectors, "test.cvc", compression="fp16")
   
   # Test CUDA (should work now)
   for i in range(3):
       result = load_cvc("test.cvc", device="cuda", backend="cuda")
       torch.cuda.synchronize()
       print(f"✅ Load {i+1} successful: {result.shape}")
   ```

## Technical Details

### CUDA Architecture Numbers
- **70**: Volta (V100)
- **75**: Turing (T4, RTX 20xx)
- **80**: Ampere (A100, A30)
- **86**: Ampere (RTX 30xx)
- **89**: Ada Lovelace (RTX 40xx, L4)
- **90**: Hopper (H100, H800)

### Why Multi-Architecture Works
- CMake generates PTX code for each specified architecture
- CUDA runtime selects appropriate PTX at load time
- JIT-compiles PTX to machine code for actual GPU
- Works across different GPU models and CUDA versions

### Performance Impact
- Slightly larger binary (contains multiple PTX versions)
- First load: ~10ms slower (JIT compilation)
- Subsequent loads: No performance impact (cached)
- **Worth it** for reliability and portability

## Related Files
- `CMakeLists.txt`: Build configuration (FIXED)
- `cvc/cuda/decompress_fp16.cu`: FP16 decompression kernel
- `cvc/cuda/decompress_int8.cu`: INT8 decompression kernel
- `python/bindings/cvc_bindings.cpp`: Python/C++ interface

## Prevention
Going forward:
- ✅ Always compile for multiple architectures
- ✅ Enable separable compilation for CUDA libraries
- ✅ Test with `torch.cuda.synchronize()` to catch errors
- ✅ Run warmup iterations in benchmarks to catch latent errors
