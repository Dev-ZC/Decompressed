"""CVC file format packer."""

import json
import numpy as np
from pathlib import Path

from .compress import compress_fp16, compress_int8

HEADER_MAGIC = b"CVCF"


def pack_cvc(vectors, output_path, compression="fp16", chunk_size=100000):
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
