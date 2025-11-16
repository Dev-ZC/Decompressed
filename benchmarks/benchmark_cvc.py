#!/usr/bin/env python3
"""
Benchmark script for Decompressed CVC format.
Tests decompression performance and shows which backend is being used.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add the python directory to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "python"))

from decompressed import load_cvc, pack_cvc


def detect_backends():
    """Detect available backends and GPU info."""
    print("=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)
    
    # Check for C++ native extensions
    try:
        from decompressed._cvc_native import CUDA_AVAILABLE
        print(f"✅ C++ Native Extensions: Available")
        print(f"   CUDA Support: {'✅ Yes' if CUDA_AVAILABLE else '❌ No'}")
        backend_cpp = True
        backend_cuda_native = CUDA_AVAILABLE
    except ImportError:
        print(f"❌ C++ Native Extensions: Not built (using pure Python)")
        backend_cpp = False
        backend_cuda_native = False
    
    # Check for Triton
    try:
        import triton
        version = getattr(triton, '__version__', 'unknown')
        print(f"✅ Triton: Available (version {version})")
        backend_triton = True
    except ImportError:
        print(f"❌ Triton: Not available")
        backend_triton = False
    
    # Check for GPU frameworks
    gpu_framework = None
    gpu_name = None
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_framework = "torch"
            print(f"✅ PyTorch GPU: Available")
            print(f"   Device: {gpu_name}")
            
            # Detect GPU vendor
            if "NVIDIA" in gpu_name.upper() or "RTX" in gpu_name.upper() or "GTX" in gpu_name.upper():
                gpu_vendor = "NVIDIA"
            elif "AMD" in gpu_name.upper() or "Radeon" in gpu_name.upper():
                gpu_vendor = "AMD"
            elif "Intel" in gpu_name.upper() or "Arc" in gpu_name.upper():
                gpu_vendor = "Intel"
            else:
                gpu_vendor = "Unknown"
            print(f"   Vendor: {gpu_vendor}")
        else:
            print(f"❌ PyTorch GPU: Not available (CPU only)")
    except ImportError:
        try:
            import cupy as cp
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
            gpu_framework = "cupy"
            print(f"✅ CuPy GPU: Available")
            print(f"   Device: {gpu_name}")
        except:
            print(f"❌ GPU Frameworks: None available")
    
    print()
    
    # Determine expected backend
    if gpu_framework:
        if backend_cuda_native:
            expected = "CUDA Native (fastest)"
        elif backend_triton:
            expected = "Triton (GPU-compiled)"
        else:
            expected = "Pure Python (slow)"
    else:
        if backend_cpp:
            expected = "C++ CPU (fast)"
        else:
            expected = "Pure Python (slow)"
    
    print(f"Expected GPU Backend: {expected}")
    print(f"Expected CPU Backend: {'C++ Native' if backend_cpp else 'Pure Python'}")
    print()
    
    return {
        'cpp': backend_cpp,
        'cuda_native': backend_cuda_native,
        'triton': backend_triton,
        'gpu_framework': gpu_framework,
        'gpu_name': gpu_name
    }


def format_throughput(size_bytes, time_seconds):
    """Format throughput in GB/s."""
    if time_seconds == 0:
        return "N/A"
    gb_per_sec = (size_bytes / 1e9) / time_seconds
    return f"{gb_per_sec:.2f} GB/s"


def run_benchmark(backends):
    """Run decompression benchmarks."""
    print("=" * 70)
    print("BENCHMARK SETUP")
    print("=" * 70)
    
    N, D = 1_000_000, 768
    print(f"Vectors: {N:,} × {D} dimensions")
    
    vectors = np.random.rand(N, D).astype(np.float32)
    uncompressed_size = vectors.nbytes
    print(f"Uncompressed size: {uncompressed_size / 1e9:.2f} GB")
    print()
    
    # Pack files
    print("Creating compressed files...")
    pack_cvc(vectors, "test_fp16.cvc", compression="fp16")
    pack_cvc(vectors, "test_int8.cvc", compression="int8")
    
    fp16_size = Path("test_fp16.cvc").stat().st_size
    int8_size = Path("test_int8.cvc").stat().st_size
    
    print(f"  FP16 compressed: {fp16_size / 1e9:.2f} GB (ratio: {uncompressed_size/fp16_size:.2f}x)")
    print(f"  INT8 compressed: {int8_size / 1e9:.2f} GB (ratio: {uncompressed_size/int8_size:.2f}x)")
    print()
    
    # CPU Benchmarks
    print("=" * 70)
    print("CPU BENCHMARKS")
    print("=" * 70)
    print(f"Backend: {'C++ Native' if backends['cpp'] else 'Pure Python'}")
    print()
    
    start = time.time()
    arr_cpu_fp16 = load_cvc("test_fp16.cvc", device="cpu")
    time_fp16 = time.time() - start
    throughput_fp16 = format_throughput(uncompressed_size, time_fp16)
    print(f"FP16 Decompression: {time_fp16:.3f}s ({throughput_fp16})")
    
    start = time.time()
    arr_cpu_int8 = load_cvc("test_int8.cvc", device="cpu")
    time_int8 = time.time() - start
    throughput_int8 = format_throughput(uncompressed_size, time_int8)
    print(f"INT8 Decompression: {time_int8:.3f}s ({throughput_int8})")
    print()
    
    # GPU Benchmarks
    if backends['gpu_framework']:
        print("=" * 70)
        print("GPU BENCHMARKS")
        print("=" * 70)
        
        # Determine which GPU backend will be used
        if backends['cuda_native']:
            gpu_backend = "CUDA Native (fastest)"
        elif backends['triton']:
            gpu_backend = "Triton (GPU-compiled)"
        else:
            gpu_backend = "Pure Python (fallback)"
        
        print(f"Device: {backends['gpu_name']}")
        print(f"Backend: {gpu_backend}")
        print(f"Framework: {backends['gpu_framework']}")
        print()
        
        try:
            if backends['gpu_framework'] == "torch":
                import torch
                
                start = time.time()
                arr_gpu_fp16 = load_cvc("test_fp16.cvc", device="cuda", framework="torch")
                torch.cuda.synchronize()
                time_fp16_gpu = time.time() - start
                throughput_fp16_gpu = format_throughput(uncompressed_size, time_fp16_gpu)
                print(f"FP16 Decompression: {time_fp16_gpu:.3f}s ({throughput_fp16_gpu})")
                
                start = time.time()
                arr_gpu_int8 = load_cvc("test_int8.cvc", device="cuda", framework="torch")
                torch.cuda.synchronize()
                time_int8_gpu = time.time() - start
                throughput_int8_gpu = format_throughput(uncompressed_size, time_int8_gpu)
                print(f"INT8 Decompression: {time_int8_gpu:.3f}s ({throughput_int8_gpu})")
                
                # Accuracy check
                print()
                print("Accuracy Check:")
                fp16_diff = np.max(np.abs(arr_cpu_fp16 - arr_gpu_fp16.cpu().numpy()))
                int8_diff = np.max(np.abs(arr_cpu_int8 - arr_gpu_int8.cpu().numpy()))
                print(f"  FP16 max diff: {fp16_diff:.6f}")
                print(f"  INT8 max diff: {int8_diff:.6f}")
                
            elif backends['gpu_framework'] == "cupy":
                import cupy as cp
                
                start = time.time()
                arr_gpu_fp16 = load_cvc("test_fp16.cvc", device="cuda", framework="cupy")
                cp.cuda.Device(0).synchronize()
                time_fp16_gpu = time.time() - start
                throughput_fp16_gpu = format_throughput(uncompressed_size, time_fp16_gpu)
                print(f"FP16 Decompression: {time_fp16_gpu:.3f}s ({throughput_fp16_gpu})")
                
                start = time.time()
                arr_gpu_int8 = load_cvc("test_int8.cvc", device="cuda", framework="cupy")
                cp.cuda.Device(0).synchronize()
                time_int8_gpu = time.time() - start
                throughput_int8_gpu = format_throughput(uncompressed_size, time_int8_gpu)
                print(f"INT8 Decompression: {time_int8_gpu:.3f}s ({throughput_int8_gpu})")
                
                # Accuracy check
                print()
                print("Accuracy Check:")
                fp16_diff = np.max(np.abs(arr_cpu_fp16 - cp.asnumpy(arr_gpu_fp16)))
                int8_diff = np.max(np.abs(arr_cpu_int8 - cp.asnumpy(arr_gpu_int8)))
                print(f"  FP16 max diff: {fp16_diff:.6f}")
                print(f"  INT8 max diff: {int8_diff:.6f}")
                
        except Exception as e:
            print(f"❌ GPU benchmarks failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("=" * 70)
        print("GPU BENCHMARKS")
        print("=" * 70)
        print("⚠️  No GPU framework available (install PyTorch or CuPy)")
    
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    backends = detect_backends()
    run_benchmark(backends)
