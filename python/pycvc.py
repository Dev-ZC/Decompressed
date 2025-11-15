import json
import numpy as np
from pathlib import Path

from triton_kernels import decompress_fp16_kernel, decompress_int8_kernel
from cvc.cpu import cvc_decompress_cpu  # CPU fallback
from cvc.utils import get_cuda_ptr     # framework agnostic pointer helper

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
                arr[offset:offset+rows] = cvc_decompress_cpu(payload, compression, rows, dim,
                                                            minv=chunk.get("min"), scale=chunk.get("scale"))
            else:
                ptr, _, _ = get_cuda_ptr(arr[offset:offset+rows])
                if compression == "fp16":
                    decompress_fp16_kernel(payload, ptr, rows*dim)
                else:
                    decompress_int8_kernel(payload, ptr, chunk["min"], chunk["scale"], rows*dim)
            offset += rows

    return arr
