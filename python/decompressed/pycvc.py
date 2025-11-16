"""Main API for loading and packing CVC files."""

import json
import numpy as np
from pathlib import Path
import sys

# Import compression functions (these don't exist in C++)
from .compress import compress_fp16, compress_int8

# Try to import C++ native extensions from cvc.cpp via pybind11 (when built by CMake)
try:
    from decompressed._cvc_native import decompress_fp16_cpu, decompress_int8_cpu, CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        from decompressed._cvc_native import decompress_fp16_cuda, decompress_int8_cuda
        HAS_CUDA = True
    else:
        HAS_CUDA = False
    HAS_NATIVE = True
except ImportError:
    # Fallback to pure Python implementations
    from .decompress import decompress_fp16_cpu, decompress_int8_cpu
    HAS_NATIVE = False
    HAS_CUDA = False

# Try to import Triton GPU kernels from cvc/triton (optional)
try:
    import triton
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cvc" / "triton"))
    from decompress_fp16_triton import decompress_fp16_kernel
    from decompress_int8_triton import decompress_int8_triton_kernel as decompress_int8_kernel
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None

HEADER_MAGIC = b"CVCF"

def load_cvc(path: str, device="cpu", framework="torch", backend="auto"):
    """
    Load a .cvc file into a GPU or CPU array.
    
    Args:
        path: Path to .cvc file
        device: "cpu" or "cuda" (GPU)
        framework: "torch" or "cupy" (for GPU arrays)
        backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
            - "auto": Use best available (cuda > cpp > triton > python)
            - "python": Pure Python (CPU only, slowest)
            - "cpp": C++ native (CPU only, fast)
            - "cuda": CUDA native (GPU only, fastest, NVIDIA only)
            - "triton": Triton kernels (GPU only, fast, vendor-agnostic)
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != HEADER_MAGIC:
            raise ValueError("Not a valid .cvc file")
        header_len = int.from_bytes(f.read(4), "little")
        header = json.loads(f.read(header_len))

        n_vectors = header["num_vectors"]
        dim = header["dimension"]
        compression = header["compression"]
        chunks_meta = header["chunks"]

        # Allocate output array
        if device == "cpu":
            arr = np.empty((n_vectors, dim), dtype=np.float32)
        else:
            if framework == "cupy":
                import cupy as cp
                arr = cp.zeros((n_vectors, dim), dtype=cp.float32)
            elif framework == "torch":
                import torch
                arr = torch.zeros((n_vectors, dim), dtype=torch.float32, device="cuda")
            else:
                raise ValueError("Unsupported framework")

        # Decompress chunks
        offset = 0
        for chunk in chunks_meta:
            chunk_len = int.from_bytes(f.read(4), "little")
            payload = f.read(chunk_len)

            rows = chunk["rows"]
            
            # Determine which backend to use
            if backend == "auto":
                if device == "cpu":
                    use_backend = "cpp" if HAS_NATIVE else "python"
                else:  # GPU
                    # Priority: CUDA native > Triton > fallback error
                    if HAS_CUDA:
                        use_backend = "cuda"
                    elif HAS_TRITON:
                        use_backend = "triton"
                    else:
                        raise RuntimeError("GPU requested but no GPU backend available. Install: pip install triton")
            else:
                use_backend = backend
            
            # Validate backend choice
            if use_backend == "cpp" and not HAS_NATIVE:
                raise RuntimeError("C++ backend requested but not available. Build with: pip install .")
            if use_backend == "cuda" and not HAS_CUDA:
                raise RuntimeError("CUDA native backend requested but not available. Build with: pip install .")
            if use_backend == "triton" and not HAS_TRITON:
                raise RuntimeError("Triton backend requested but not available. Install: pip install triton")
            if use_backend in ["cuda", "triton"] and device == "cpu":
                raise ValueError(f"Backend '{use_backend}' requires device='cuda', not 'cpu'")
            if use_backend in ["python", "cpp"] and device != "cpu":
                raise ValueError(f"Backend '{use_backend}' requires device='cpu', not '{device}'")
            
            # Execute decompression with selected backend
            if use_backend == "python":
                # Pure Python (imports from decompress.py directly)
                from .decompress import decompress_fp16_cpu as py_fp16, decompress_int8_cpu as py_int8
                if compression == "fp16":
                    arr[offset:offset+rows] = py_fp16(payload, rows, dim)
                else:
                    arr[offset:offset+rows] = py_int8(payload, rows, dim, chunk["min"], chunk["scale"])
                    
            elif use_backend == "cpp":
                # C++ native (uses HAS_NATIVE imports)
                if compression == "fp16":
                    arr[offset:offset+rows] = decompress_fp16_cpu(payload, rows, dim)
                else:
                    arr[offset:offset+rows] = decompress_int8_cpu(
                        payload, rows, dim, chunk["min"], chunk["scale"]
                    )
                    
            elif use_backend == "cuda":
                # CUDA native kernels (fastest, NVIDIA only)
                # Convert payload to numpy array
                if compression == "fp16":
                    src_data = np.frombuffer(payload, dtype=np.float16)
                else:  # int8
                    src_data = np.frombuffer(payload, dtype=np.uint8)
                
                # Upload to GPU and decompress
                if framework == "torch":
                    import torch
                    src_gpu = torch.from_numpy(src_data).cuda()
                    dst_slice = arr[offset:offset+rows].flatten()
                    
                    n_elements = rows * dim
                    
                    if compression == "fp16":
                        decompress_fp16_cuda(
                            src_gpu.data_ptr(),
                            dst_slice.data_ptr(),
                            n_elements
                        )
                    else:  # int8
                        decompress_int8_cuda(
                            src_gpu.data_ptr(),
                            dst_slice.data_ptr(),
                            chunk["min"],
                            chunk["scale"],
                            n_elements
                        )
                    torch.cuda.synchronize()
                    
                elif framework == "cupy":
                    import cupy as cp
                    src_gpu = cp.asarray(src_data)
                    dst_slice = arr[offset:offset+rows].flatten()
                    
                    n_elements = rows * dim
                    
                    if compression == "fp16":
                        decompress_fp16_cuda(
                            src_gpu.data.ptr,
                            dst_slice.data.ptr,
                            n_elements
                        )
                    else:  # int8
                        decompress_int8_cuda(
                            src_gpu.data.ptr,
                            dst_slice.data.ptr,
                            chunk["min"],
                            chunk["scale"],
                            n_elements
                        )
                    cp.cuda.Device(0).synchronize()
                    
            elif use_backend == "triton":
                # Triton GPU kernels
                # Convert payload to numpy array
                if compression == "fp16":
                    src_data = np.frombuffer(payload, dtype=np.float16)
                else:  # int8
                    src_data = np.frombuffer(payload, dtype=np.uint8)
                
                # Upload to GPU based on framework
                if framework == "torch":
                    import torch
                    src_gpu = torch.from_numpy(src_data).cuda()
                    dst_slice = arr[offset:offset+rows]
                    
                    # Launch Triton kernel
                    n_elements = rows * dim
                    BLOCK_SIZE = 1024
                    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
                    
                    if compression == "fp16":
                        decompress_fp16_kernel[grid](
                            src_gpu.data_ptr(), 
                            dst_slice.data_ptr(),
                            n_elements,
                            BLOCK_SIZE
                        )
                    else:  # int8
                        decompress_int8_kernel[grid](
                            src_gpu.data_ptr(),
                            dst_slice.data_ptr(),
                            chunk["min"],
                            chunk["scale"],
                            n_elements,
                            BLOCK_SIZE
                        )
                    torch.cuda.synchronize()
                    
                elif framework == "cupy":
                    import cupy as cp
                    src_gpu = cp.asarray(src_data)
                    dst_slice = arr[offset:offset+rows]
                    
                    # Launch Triton kernel
                    n_elements = rows * dim
                    BLOCK_SIZE = 1024
                    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE),)
                    
                    if compression == "fp16":
                        decompress_fp16_kernel[grid](
                            src_gpu.data.ptr,
                            dst_slice.data.ptr,
                            n_elements,
                            BLOCK_SIZE
                        )
                    else:  # int8
                        decompress_int8_kernel[grid](
                            src_gpu.data.ptr,
                            dst_slice.data.ptr,
                            chunk["min"],
                            chunk["scale"],
                            n_elements,
                            BLOCK_SIZE
                        )
                    cp.cuda.Device(0).synchronize()
                    
            offset += rows

    return arr


def pack_cvc(vectors: np.ndarray, output_path: str, compression: str = "fp16", chunk_size: int = 100000):
    """
    Pack numpy array of vectors into .cvc compressed format.
    
    Args:
        vectors: np.ndarray of shape (n_vectors, dimension), dtype float32
        output_path: Path to output .cvc file
        compression: "fp16" or "int8"
        chunk_size: Number of vectors per chunk
    """
    if compression not in ["fp16", "int8"]:
        raise ValueError(f"Unknown compression: {compression}. Use 'fp16' or 'int8'")
    
    n_vectors, dim = vectors.shape
    
    # Build chunks
    chunks_meta = []
    chunk_payloads = []
    
    for start_idx in range(0, n_vectors, chunk_size):
        end_idx = min(start_idx + chunk_size, n_vectors)
        chunk_vectors = vectors[start_idx:end_idx]
        rows = end_idx - start_idx
        
        if compression == "fp16":
            payload = compress_fp16(chunk_vectors)
            chunk_meta = {"rows": rows, "compression": "fp16"}
        else:  # int8
            payload, minv, scale = compress_int8(chunk_vectors)
            chunk_meta = {
                "rows": rows,
                "compression": "int8",
                "min": minv,
                "scale": scale
            }
        
        chunks_meta.append(chunk_meta)
        chunk_payloads.append(payload)
    
    # Build header
    header = {
        "num_vectors": n_vectors,
        "dimension": dim,
        "compression": compression,
        "chunks": chunks_meta
    }
    header_bytes = json.dumps(header).encode('utf-8')
    header_len = len(header_bytes)
    
    # Write file
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        f.write(HEADER_MAGIC)
        f.write(header_len.to_bytes(4, byteorder='little'))
        f.write(header_bytes)
        
        for payload in chunk_payloads:
            f.write(len(payload).to_bytes(4, byteorder='little'))
            f.write(payload)
