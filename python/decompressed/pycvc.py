"""Main API for loading and packing CVC files."""

import json
import numpy as np
from pathlib import Path
import sys

# Import compression functions (these don't exist in C++)
from .compress import compress_fp16, compress_int8

# Try to import C++ native extensions from cvc.cpp via pybind11 (when built by CMake)
try:
    from decompressed._cvc_native import decompress_fp16_cpu, decompress_int8_cpu
    HAS_NATIVE = True
except ImportError:
    # Fallback to pure Python implementations
    from .decompress import decompress_fp16_cpu, decompress_int8_cpu
    HAS_NATIVE = False

# Try to import Triton GPU kernels from cvc/triton (optional)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "cvc" / "triton"))
    from decompress_fp16_triton import decompress_fp16_kernel
    from decompress_int8_triton import decompress_int8_triton_kernel
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

HEADER_MAGIC = b"CVCF"

def load_cvc(path: str, device="cpu", framework="cupy"):
    """
    Load a .cvc file into a GPU or CPU array.
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
            if device == "cpu":
                # decompress_fp16_cpu and decompress_int8_cpu are from C++ (cvc.cpp) if built,
                # otherwise from pure Python (decompress.py)
                if compression == "fp16":
                    arr[offset:offset+rows] = decompress_fp16_cpu(payload, rows, dim)
                else:  # int8
                    arr[offset:offset+rows] = decompress_int8_cpu(
                        payload, rows, dim, chunk["min"], chunk["scale"]
                    )
            else:
                # GPU decompression - not yet implemented
                raise NotImplementedError("GPU decompression via Triton coming soon. Use device='cpu' for now.")
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
