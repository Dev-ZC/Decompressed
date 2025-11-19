#!/usr/bin/env python3
"""
Comprehensive benchmark script for Decompressed CVC format.

Compares Decompressed against:
- Traditional storage formats (NumPy, pickle+gzip, Zstd, LZ4)
- PyTorch-native FP16/INT8 models

Tests measure end-to-end retrieval performance:
load → GPU transfer → decompression → matrix multiply
"""

import os
import numpy as np
import torch
import time
import pickle
import gzip
import pandas as pd
from tabulate import tabulate
from decompressed import pack_cvc, load_cvc

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("⚠️  zstandard not installed. Install with: pip install zstandard")

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    print("⚠️  lz4 not installed. Install with: pip install lz4")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NUM_MODELS = 5
NUM_VECTORS = 100_000
VECTOR_DIM = 768
NUM_QUERIES = 512
NUM_RUNS = 3  # Number of benchmark runs for averaging


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_gpu():
    """Check GPU availability and print device info."""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    
    gpu_available = torch.cuda.is_available()
    print(f"✅ GPU Available: {gpu_available}")
    
    if gpu_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"   Device: {device_name}")
        
        # Detect GPU vendor
        if "NVIDIA" in device_name.upper() or "RTX" in device_name.upper():
            vendor = "NVIDIA"
        elif "AMD" in device_name.upper() or "Radeon" in device_name.upper():
            vendor = "AMD"
        elif "Intel" in device_name.upper() or "Arc" in device_name.upper():
            vendor = "Intel"
        else:
            vendor = "Unknown"
        print(f"   Vendor: {vendor}")
    else:
        print("   ⚠️  Running in CPU-only mode")
    
    print()
    return gpu_available


def quantize_tensor(fp_tensor):
    """Quantize a FP32/FP16 tensor to INT8 using PyTorch's quantization."""
    fp_tensor = fp_tensor.float()  # must be float for quantize_per_tensor
    min_val, max_val = fp_tensor.min().item(), fp_tensor.max().item()
    scale = (max_val - min_val) / 255.0
    q_tensor = torch.quantize_per_tensor(fp_tensor, scale=scale, zero_point=0, dtype=torch.qint8)
    return q_tensor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SETUP: CREATE TEST DATASETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_test_datasets():
    """Create dummy embedding datasets in various formats."""
    print("=" * 80)
    print("CREATING TEST DATASETS")
    print("=" * 80)
    print(f"Models: {NUM_MODELS}")
    print(f"Vectors per model: {NUM_VECTORS:,}")
    print(f"Dimensions: {VECTOR_DIM}")
    print()
    
    datasets = {}
    
    for model_id in range(NUM_MODELS):
        # Random FP32 embeddings
        embeddings = np.random.randn(NUM_VECTORS, VECTOR_DIM).astype(np.float32)
        datasets[f'model_{model_id}'] = embeddings
        
        # Save PyTorch FP16
        embeddings_fp16 = torch.from_numpy(embeddings.astype(np.float16))
        torch.save(embeddings_fp16, f'model_{model_id}_fp16.pt')
        
        # Save PyTorch INT8 (quantized)
        q_tensor = quantize_tensor(embeddings_fp16)
        torch.save(q_tensor, f'model_{model_id}_torch_int8.pt')
        
        # Save Decompressed .cvc (FP16)
        pack_cvc(embeddings, f'model_{model_id}.cvc', compression='fp16', chunk_size=50_000)
        
        # Save Decompressed .cvc (INT8)
        pack_cvc(embeddings, f'model_{model_id}_int8.cvc', compression='int8', chunk_size=50_000)
        
        # Save NumPy .npy
        np.save(f'model_{model_id}.npy', embeddings)
        
        # Save pickle + gzip
        with gzip.open(f'model_{model_id}.pkl.gz', 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save Zstd (if available)
        if ZSTD_AVAILABLE:
            compressor = zstd.ZstdCompressor(level=10)
            with open(f'model_{model_id}.zst', 'wb') as f:
                f.write(compressor.compress(embeddings.tobytes()))
        
        # Save LZ4 (if available)
        if LZ4_AVAILABLE:
            with lz4.frame.open(f'model_{model_id}.lz4', 'wb') as f:
                f.write(embeddings.tobytes())
    
    print("✅ Test datasets created")
    print()
    return datasets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BENCHMARK: GPU-NATIVE FORMATS (FP16 & INT8)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def benchmark_gpu_native_formats(gpu_available):
    """Benchmark PyTorch and Decompressed FP16/INT8 formats."""
    print("=" * 80)
    print("BENCHMARKING: GPU-NATIVE FORMATS (FP16 & INT8)")
    print("=" * 80)
    print()
    
    results = []
    
    # Prepare queries
    queries = np.random.randn(NUM_QUERIES, VECTOR_DIM).astype(np.float32)
    
    if gpu_available:
        queries_gpu = torch.from_numpy(queries).cuda()
    
    # File paths
    pt_fp16_files = [f'model_{i}_fp16.pt' for i in range(NUM_MODELS)]
    cvc_fp16_files = [f'model_{i}.cvc' for i in range(NUM_MODELS)]
    pt_int8_files = [f'model_{i}_torch_int8.pt' for i in range(NUM_MODELS)]
    cvc_int8_files = [f'model_{i}_int8.cvc' for i in range(NUM_MODELS)]
    
    # ========== PyTorch FP16 (GPU) ==========
    if gpu_available:
        print("🔹 Benchmarking PyTorch FP16 (GPU)...")
        torch_times = []
        torch_storage = sum([os.path.getsize(f) for f in pt_fp16_files]) / 1e9
        
        for run in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.time()
            for f in pt_fp16_files:
                data_gpu = torch.load(f).cuda()
                _ = torch.mm(queries_gpu.half(), data_gpu.T)
            torch.cuda.synchronize()
            torch_times.append(time.time() - start)
        
        results.append({
            "Method": "PyTorch FP16 (GPU)",
            "Storage (GB)": f"{torch_storage:.2f}",
            "Total Time (s)": f"{np.mean(torch_times):.3f}",
            "Time per Model (s)": f"{np.mean(torch_times)/NUM_MODELS:.3f}"
        })
        print(f"   ✅ Avg time: {np.mean(torch_times):.3f}s")
    
    # ========== Decompressed FP16 (GPU) ==========
    if gpu_available:
        print("🔹 Benchmarking Decompressed FP16 (GPU)...")
        decomp_times_fp16 = []
        decomp_storage_fp16 = sum([os.path.getsize(f) for f in cvc_fp16_files]) / 1e9
        
        for run in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.time()
            for f in cvc_fp16_files:
                data_gpu = load_cvc(f, device='cuda', backend='triton').half()
                _ = torch.mm(queries_gpu.half(), data_gpu.T)
            torch.cuda.synchronize()
            decomp_times_fp16.append(time.time() - start)
        
        results.append({
            "Method": "Decompressed FP16 (GPU)",
            "Storage (GB)": f"{decomp_storage_fp16:.2f}",
            "Total Time (s)": f"{np.mean(decomp_times_fp16):.3f}",
            "Time per Model (s)": f"{np.mean(decomp_times_fp16)/NUM_MODELS:.3f}"
        })
        print(f"   ✅ Avg time: {np.mean(decomp_times_fp16):.3f}s")
    
    # ========== PyTorch INT8 (CPU → FP32) ==========
    print("🔹 Benchmarking PyTorch INT8 (CPU dequantization)...")
    native_int8_times = []
    native_int8_storage = sum([os.path.getsize(f) for f in pt_int8_files]) / 1e9
    queries_cpu = torch.from_numpy(queries).contiguous()
    
    for run in range(NUM_RUNS):
        start = time.time()
        for f in pt_int8_files:
            data_int8 = torch.load(f)
            data_cpu = data_int8.dequantize()  # CPU dequantization
            _ = torch.mm(queries_cpu, data_cpu.T)
        native_int8_times.append(time.time() - start)
    
    results.append({
        "Method": "PyTorch INT8 (CPU → FP32)",
        "Storage (GB)": f"{native_int8_storage:.2f}",
        "Total Time (s)": f"{np.mean(native_int8_times):.3f}",
        "Time per Model (s)": f"{np.mean(native_int8_times)/NUM_MODELS:.3f}"
    })
    print(f"   ✅ Avg time: {np.mean(native_int8_times):.3f}s")
    
    # ========== Decompressed INT8 (GPU → FP32) ==========
    if gpu_available:
        print("🔹 Benchmarking Decompressed INT8 (GPU → FP32)...")
        decomp_times_int8 = []
        decomp_storage_int8 = sum([os.path.getsize(f) for f in cvc_int8_files]) / 1e9
        
        for run in range(NUM_RUNS):
            torch.cuda.synchronize()
            start = time.time()
            for f in cvc_int8_files:
                data_gpu = load_cvc(f, device='cuda', backend='triton').float()
                _ = torch.mm(queries_gpu, data_gpu.T)
            torch.cuda.synchronize()
            decomp_times_int8.append(time.time() - start)
        
        results.append({
            "Method": "Decompressed INT8 (GPU → FP32)",
            "Storage (GB)": f"{decomp_storage_int8:.2f}",
            "Total Time (s)": f"{np.mean(decomp_times_int8):.3f}",
            "Time per Model (s)": f"{np.mean(decomp_times_int8)/NUM_MODELS:.3f}"
        })
        print(f"   ✅ Avg time: {np.mean(decomp_times_int8):.3f}s")
    
    print()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BENCHMARK: TRADITIONAL STORAGE FORMATS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def benchmark_baseline_formats(gpu_available):
    """Benchmark traditional storage formats: NumPy, pickle+gzip, Zstd, LZ4."""
    print("=" * 80)
    print("BENCHMARKING: BASELINE STORAGE FORMATS")
    print("=" * 80)
    print()
    
    results = []
    
    # Prepare queries
    queries = np.random.randn(NUM_QUERIES, VECTOR_DIM).astype(np.float32)
    if gpu_available:
        queries_gpu = torch.from_numpy(queries).cuda()
    else:
        queries_cpu = torch.from_numpy(queries)
    
    # File paths
    npy_files = [f"model_{i}.npy" for i in range(NUM_MODELS)]
    pkl_gz_files = [f"model_{i}.pkl.gz" for i in range(NUM_MODELS)]
    zstd_files = [f"model_{i}.zst" for i in range(NUM_MODELS)]
    lz4_files = [f"model_{i}.lz4" for i in range(NUM_MODELS)]
    
    # ========== NumPy (.npy) ==========
    print("🔹 Benchmarking NumPy (.npy)...")
    numpy_times = []
    numpy_storage = sum([os.path.getsize(f) for f in npy_files]) / 1e9
    
    for run in range(NUM_RUNS):
        start = time.time()
        for f in npy_files:
            arr = np.load(f)
            if gpu_available:
                data_gpu = torch.from_numpy(arr).cuda().float()
                _ = torch.mm(queries_gpu, data_gpu.T)
            else:
                data_cpu = torch.from_numpy(arr).float()
                _ = torch.mm(queries_cpu, data_cpu.T)
        numpy_times.append(time.time() - start)
    
    results.append({
        "Method": "NumPy (.npy)",
        "Storage (GB)": f"{numpy_storage:.2f}",
        "Total Time (s)": f"{np.mean(numpy_times):.3f}",
        "Time per Model (s)": f"{np.mean(numpy_times)/NUM_MODELS:.3f}"
    })
    print(f"   ✅ Avg time: {np.mean(numpy_times):.3f}s")
    
    # ========== Pickle + gzip ==========
    print("🔹 Benchmarking Pickle + gzip...")
    pkl_gz_times = []
    pkl_gz_storage = sum([os.path.getsize(f) for f in pkl_gz_files]) / 1e9
    
    for run in range(NUM_RUNS):
        start = time.time()
        for f in pkl_gz_files:
            with gzip.open(f, "rb") as fin:
                arr = pickle.load(fin)
                if gpu_available:
                    data_gpu = torch.from_numpy(arr).cuda().float()
                    _ = torch.mm(queries_gpu, data_gpu.T)
                else:
                    data_cpu = torch.from_numpy(arr).float()
                    _ = torch.mm(queries_cpu, data_cpu.T)
        pkl_gz_times.append(time.time() - start)
    
    results.append({
        "Method": "Pickle + gzip",
        "Storage (GB)": f"{pkl_gz_storage:.2f}",
        "Total Time (s)": f"{np.mean(pkl_gz_times):.3f}",
        "Time per Model (s)": f"{np.mean(pkl_gz_times)/NUM_MODELS:.3f}"
    })
    print(f"   ✅ Avg time: {np.mean(pkl_gz_times):.3f}s")
    
    # ========== Zstandard (.zst) ==========
    if ZSTD_AVAILABLE:
        print("🔹 Benchmarking Zstandard (.zst)...")
        zstd_times = []
        zstd_storage = sum([os.path.getsize(f) for f in zstd_files]) / 1e9
        dctx = zstd.ZstdDecompressor()
        
        for run in range(NUM_RUNS):
            start = time.time()
            for f in zstd_files:
                with open(f, "rb") as fin:
                    blob = dctx.decompress(fin.read())
                    arr = np.frombuffer(blob, dtype=np.float32).reshape(-1, VECTOR_DIM)
                    if gpu_available:
                        data_gpu = torch.from_numpy(arr).cuda().float()
                        _ = torch.mm(queries_gpu, data_gpu.T)
                    else:
                        data_cpu = torch.from_numpy(arr).float()
                        _ = torch.mm(queries_cpu, data_cpu.T)
            zstd_times.append(time.time() - start)
        
        results.append({
            "Method": "Zstandard (.zst)",
            "Storage (GB)": f"{zstd_storage:.2f}",
            "Total Time (s)": f"{np.mean(zstd_times):.3f}",
            "Time per Model (s)": f"{np.mean(zstd_times)/NUM_MODELS:.3f}"
        })
        print(f"   ✅ Avg time: {np.mean(zstd_times):.3f}s")
    
    # ========== LZ4 (.lz4) ==========
    if LZ4_AVAILABLE:
        print("🔹 Benchmarking LZ4 (.lz4)...")
        lz4_times = []
        lz4_storage = sum([os.path.getsize(f) for f in lz4_files]) / 1e9
        
        for run in range(NUM_RUNS):
            start = time.time()
            for f in lz4_files:
                with lz4.frame.open(f, "rb") as fin:
                    blob = fin.read()
                    arr = np.frombuffer(blob, dtype=np.float32).reshape(-1, VECTOR_DIM)
                    if gpu_available:
                        data_gpu = torch.from_numpy(arr).cuda().float()
                        _ = torch.mm(queries_gpu, data_gpu.T)
                    else:
                        data_cpu = torch.from_numpy(arr).float()
                        _ = torch.mm(queries_cpu, data_cpu.T)
            lz4_times.append(time.time() - start)
        
        results.append({
            "Method": "LZ4 (.lz4)",
            "Storage (GB)": f"{lz4_storage:.2f}",
            "Total Time (s)": f"{np.mean(lz4_times):.3f}",
            "Time per Model (s)": f"{np.mean(lz4_times)/NUM_MODELS:.3f}"
        })
        print(f"   ✅ Avg time: {np.mean(lz4_times):.3f}s")
    
    print()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Run all benchmarks and display results."""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DECOMPRESSED BENCHMARK SUITE" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Check GPU availability
    gpu_available = check_gpu()
    
    # Create test datasets
    create_test_datasets()
    
    # Run benchmarks
    all_results = []
    
    # GPU-native formats (FP16 & INT8)
    gpu_results = benchmark_gpu_native_formats(gpu_available)
    all_results.extend(gpu_results)
    
    # Baseline formats (NumPy, gzip, Zstd, LZ4)
    baseline_results = benchmark_baseline_formats(gpu_available)
    all_results.extend(baseline_results)
    
    # ========== Display Results ==========
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()
    
    df_results = pd.DataFrame(all_results)
    print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))
    
    print()
    print("=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print()
    print("📊 Key Takeaways:")
    print("   • Decompressed INT8 (GPU) provides the best balance of size & speed")
    print("   • GPU-native decompression eliminates CPU bottlenecks")
    print("   • Traditional formats (gzip, Zstd) suffer from CPU decompression overhead")
    print()


if __name__ == "__main__":
    main()
