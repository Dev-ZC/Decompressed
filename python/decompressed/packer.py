"""CVC file format packer."""

import json
import numpy as np
from pathlib import Path
import math

from .compress import compress_fp16, compress_int8

HEADER_MAGIC = b"CVCF"


def pack_cvc(vectors, output_path, compression="fp16", chunk_size=100000, chunk_metadata=None):
    """
    Pack numpy array of vectors into .cvc compressed format.
    
    Args:
        vectors: np.ndarray of shape (n_vectors, dimension), dtype float32
        output_path: Path to output .cvc file
        compression: "fp16" or "int8"
        chunk_size: Number of vectors per chunk
        chunk_metadata: Optional list of dicts with metadata per chunk
    """
    if compression not in ["fp16", "int8"]:
        raise ValueError(f"Unknown compression: {compression}. Use 'fp16' or 'int8'")
    
    n_vectors, dim = vectors.shape
    
    # Build chunks
    chunks_meta = []
    chunk_payloads = []
    
    num_chunks = -(-n_vectors // chunk_size)  # Ceiling division
    
    # Validate metadata
    if chunk_metadata and len(chunk_metadata) != num_chunks:
        raise ValueError(f"chunk_metadata must have {num_chunks} items")
    
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
        
        if chunk_metadata:
            chunk_meta["metadata"] = chunk_metadata[len(chunks_meta)]
        
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


def pack_cvc_sections(sections, output_path, compression="fp16", chunk_size=100000):
    """
    Pack multiple arrays with section-level metadata into a single .cvc file.
    
    This function allows you to combine multiple data sources (with different sizes)
    into one file while maintaining section-level metadata for filtering.
    
    Args:
        sections: List of tuples (array, metadata_dict) where:
                 - array: np.ndarray of shape (n_vectors, dimension), dtype float32
                 - metadata_dict: dict with metadata for this section
        output_path: Path to output .cvc file
        compression: "fp16" or "int8"
        chunk_size: Number of vectors per chunk (applies to all sections)
    
    Example:
        >>> wikipedia = np.random.randn(10_000, 768).astype(np.float32)
        >>> arxiv = np.random.randn(110_000, 768).astype(np.float32)
        >>> 
        >>> sections = [
        >>>     (wikipedia, {"source": "wikipedia", "date": "2024-01"}),
        >>>     (arxiv, {"source": "arxiv", "date": "2024-02", "quality": "high"}),
        >>> ]
        >>> 
        >>> pack_cvc_sections(sections, "combined.cvc", chunk_size=10_000)
    """
    if compression not in ["fp16", "int8"]:
        raise ValueError(f"Unknown compression: {compression}. Use 'fp16' or 'int8'")
    
    if not sections:
        raise ValueError("sections cannot be empty")
    
    # Validate all arrays have same dimension
    dimensions = [arr.shape[1] for arr, _ in sections]
    if len(set(dimensions)) > 1:
        raise ValueError(f"All arrays must have same dimension. Found: {dimensions}")
    dim = dimensions[0]
    
    # Concatenate all arrays and track section boundaries
    section_info = []
    current_offset = 0
    
    for arr, metadata in sections:
        n_vectors = arr.shape[0]
        section_info.append({
            "start_index": current_offset,
            "end_index": current_offset + n_vectors,
            "num_vectors": n_vectors,
            "metadata": metadata
        })
        current_offset += n_vectors
    
    # Concatenate all arrays
    all_vectors = np.vstack([arr for arr, _ in sections])
    total_vectors = all_vectors.shape[0]
    
    # Build chunks with section tracking
    chunks_meta = []
    chunk_payloads = []
    
    for chunk_start in range(0, total_vectors, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_vectors)
        chunk_vectors = all_vectors[chunk_start:chunk_end]
        rows = chunk_end - chunk_start
        
        # Determine which sections this chunk intersects
        chunk_sections = []
        for section in section_info:
            # Check if chunk overlaps with section
            if chunk_start < section["end_index"] and chunk_end > section["start_index"]:
                overlap_start = max(chunk_start, section["start_index"])
                overlap_end = min(chunk_end, section["end_index"])
                chunk_sections.append({
                    "metadata": section["metadata"],
                    "start_in_chunk": overlap_start - chunk_start,
                    "end_in_chunk": overlap_end - chunk_start,
                    "section_start": section["start_index"],
                    "section_end": section["end_index"]
                })
        
        if compression == "fp16":
            payload = compress_fp16(chunk_vectors)
            chunk_meta = {
                "rows": rows,
                "compression": "fp16",
                "sections": chunk_sections
            }
        else:  # int8
            payload, minv, scale = compress_int8(chunk_vectors)
            chunk_meta = {
                "rows": rows,
                "compression": "int8",
                "min": minv,
                "scale": scale,
                "sections": chunk_sections
            }
        
        chunks_meta.append(chunk_meta)
        chunk_payloads.append(payload)
    
    # Build header with section information
    header = {
        "num_vectors": total_vectors,
        "dimension": dim,
        "compression": compression,
        "sections": section_info,
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
