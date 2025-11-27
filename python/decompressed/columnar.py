"""Multi-column CVC packer and loader for heterogeneous tensor storage."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import zlib

from .compress import compress_fp16, compress_int8
from .schema import ColumnSchema, ColumnarFileSchema, infer_column_schema, validate_column_data

HEADER_MAGIC = b"CVCF"
FORMAT_VERSION_MAJOR = 2
FORMAT_VERSION_MINOR = 0


def pack_cvc_columns(
    data: Dict[str, np.ndarray],
    output_path: str,
    compressions: Optional[Dict[str, str]] = None,
    chunk_size: int = 100_000
):
    """
    Pack multiple arrays (columns) into a single columnar CVC file.
    
    Args:
        data: Dict mapping column names to arrays.
              Each array must have shape (n_vectors, dimension) or (n_vectors,) for scalars.
        output_path: Path to output .cvc file
        compressions: Dict mapping column names to compression type ("fp16", "int8", "none").
                     Defaults to "fp16" for float columns, "none" for integer columns.
        chunk_size: Number of vectors per chunk
    
    Example:
        >>> data = {
        >>>     "text": np.random.randn(1_000_000, 768).astype(np.float32),
        >>>     "image": np.random.randn(1_000_000, 512).astype(np.float32),
        >>>     "doc_id": np.arange(1_000_000, dtype=np.int32)
        >>> }
        >>> 
        >>> compressions = {
        >>>     "text": "fp16",
        >>>     "image": "int8", 
        >>>     "doc_id": "none"
        >>> }
        >>> 
        >>> pack_cvc_columns(data, "multi_modal.cvc", compressions)
    """
    if not data:
        raise ValueError("data cannot be empty")
    
    # Get num_vectors from first column
    first_col = next(iter(data.values()))
    num_vectors = first_col.shape[0]
    
    # Validate all columns have same num_vectors
    for name, arr in data.items():
        if arr.shape[0] != num_vectors:
            raise ValueError(
                f"All columns must have same number of vectors. "
                f"Expected {num_vectors}, got {arr.shape[0]} for '{name}'"
            )
    
    # Build column schemas
    compressions = compressions or {}
    columns = [
        infer_column_schema(name, arr, compressions.get(name))
        for name, arr in data.items()
    ]
    
    # Create and validate schema
    schema = ColumnarFileSchema(
        format_type="columnar",
        num_vectors=num_vectors,
        columns=columns
    )
    schema.validate()
    validate_column_data(columns, data)
    
    # Pack chunks
    chunks_meta, chunk_payloads = _pack_columnar_chunks(
        data, columns, num_vectors, chunk_size
    )
    
    # Write file
    _write_columnar_file(output_path, schema, chunks_meta, chunk_payloads)
    
    print(f"✅ Packed {num_vectors:,} vectors with {len(columns)} columns")
    print(f"   Columns: {[col.name for col in columns]}")
    print(f"   File: {output_path} ({Path(output_path).stat().st_size / 1e9:.2f} GB)")


def pack_cvc_sections_columnar(
    sections: List[Tuple[Dict[str, np.ndarray], dict]],
    output_path: str,
    compressions: Optional[Dict[str, str]] = None,
    chunk_size: int = 100_000
):
    """
    Pack multi-column data from multiple sections.
    
    Combines sections (vertical stacking) with columns (horizontal expansion).
    
    Args:
        sections: List of (column_dict, metadata) tuples where:
                 - column_dict: Dict mapping column names to arrays
                 - metadata: Dict with section metadata (e.g., {"source": "wikipedia"})
        output_path: Path to output .cvc file
        compressions: Dict mapping column names to compression type
        chunk_size: Number of vectors per chunk
    
    Example:
        >>> sections = [
        >>>     (
        >>>         {"text": wiki_text, "image": wiki_img},
        >>>         {"source": "wikipedia", "date": "2024-01"}
        >>>     ),
        >>>     (
        >>>         {"text": arxiv_text, "image": arxiv_img},
        >>>         {"source": "arxiv", "date": "2024-02"}
        >>>     )
        >>> ]
        >>> pack_cvc_sections_columnar(sections, "combined.cvc")
    """
    if not sections:
        raise ValueError("sections cannot be empty")
    
    # Validate all sections have same columns
    first_section_cols = set(sections[0][0].keys())
    for i, (col_dict, _) in enumerate(sections[1:], 1):
        if set(col_dict.keys()) != first_section_cols:
            raise ValueError(
                f"Section {i} has different columns than section 0. "
                f"All sections must have identical column names."
            )
    
    # Validate all columns have same dimensions across sections
    for col_name in first_section_cols:
        dims = [
            (1 if col_dict[col_name].ndim == 1 else col_dict[col_name].shape[1])
            for col_dict, _ in sections
        ]
        if len(set(dims)) > 1:
            raise ValueError(
                f"Column '{col_name}' has different dimensions across sections: {dims}"
            )
    
    # Concatenate sections and track boundaries
    section_info = []
    current_offset = 0
    all_data = {col_name: [] for col_name in first_section_cols}
    
    for col_dict, metadata in sections:
        n_vectors = col_dict[next(iter(col_dict))].shape[0]
        
        section_info.append({
            "start_index": current_offset,
            "end_index": current_offset + n_vectors,
            "num_vectors": n_vectors,
            "metadata": metadata
        })
        
        # Collect column data
        for col_name in first_section_cols:
            all_data[col_name].append(col_dict[col_name])
        
        current_offset += n_vectors
    
    # Concatenate all column data
    concatenated_data = {
        col_name: np.vstack(arrays) if arrays[0].ndim > 1 else np.concatenate(arrays)
        for col_name, arrays in all_data.items()
    }
    
    num_vectors = current_offset
    
    # Build column schemas
    compressions = compressions or {}
    columns = [
        infer_column_schema(name, arr, compressions.get(name))
        for name, arr in concatenated_data.items()
    ]
    
    # Create schema with sections
    schema = ColumnarFileSchema(
        format_type="columnar",
        num_vectors=num_vectors,
        columns=columns,
        sections=section_info
    )
    schema.validate()
    
    # Pack chunks with section tracking
    chunks_meta, chunk_payloads = _pack_columnar_chunks_with_sections(
        concatenated_data, columns, num_vectors, chunk_size, section_info
    )
    
    # Write file
    _write_columnar_file(output_path, schema, chunks_meta, chunk_payloads)
    
    print(f"✅ Packed {num_vectors:,} vectors from {len(sections)} sections")
    print(f"   Columns: {[col.name for col in columns]}")
    print(f"   Sections: {[s['metadata'] for s in section_info]}")
    print(f"   File: {output_path} ({Path(output_path).stat().st_size / 1e9:.2f} GB)")


def _pack_columnar_chunks(
    data: Dict[str, np.ndarray],
    columns: List[ColumnSchema],
    num_vectors: int,
    chunk_size: int
) -> Tuple[List[dict], List[bytes]]:
    """Pack data into columnar chunks."""
    chunks_meta = []
    chunk_payloads = []
    
    num_chunks = -(-num_vectors // chunk_size)  # Ceiling division
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_vectors)
        chunk_rows = chunk_end - chunk_start
        
        # Compress each column for this chunk
        column_chunks = {}
        chunk_payload_parts = []
        current_offset = 0
        
        for col_schema in columns:
            col_name = col_schema.name
            col_data = data[col_name][chunk_start:chunk_end]
            
            # Ensure 2D for compression
            if col_data.ndim == 1:
                col_data = col_data.reshape(-1, 1)
            
            # Compress based on column schema
            payload, col_meta = _compress_column(col_data, col_schema)
            col_meta["offset"] = current_offset
            col_meta["size"] = len(payload)
            
            column_chunks[col_name] = col_meta
            chunk_payload_parts.append(payload)
            current_offset += len(payload)
        
        # Concatenate all column payloads
        chunk_payload = b"".join(chunk_payload_parts)
        
        chunks_meta.append({
            "rows": chunk_rows,
            "columns": column_chunks
        })
        chunk_payloads.append(chunk_payload)
    
    return chunks_meta, chunk_payloads


def _pack_columnar_chunks_with_sections(
    data: Dict[str, np.ndarray],
    columns: List[ColumnSchema],
    num_vectors: int,
    chunk_size: int,
    section_info: List[dict]
) -> Tuple[List[dict], List[bytes]]:
    """Pack data into columnar chunks with section tracking."""
    chunks_meta, chunk_payloads = _pack_columnar_chunks(
        data, columns, num_vectors, chunk_size
    )
    
    # Add section tracking to chunks
    for chunk_idx, chunk_meta in enumerate(chunks_meta):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, num_vectors)
        
        # Find sections overlapping this chunk
        chunk_sections = []
        for section in section_info:
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
        
        if chunk_sections:
            chunk_meta["sections"] = chunk_sections
    
    return chunks_meta, chunk_payloads


def _compress_column(
    col_data: np.ndarray,
    col_schema: ColumnSchema
) -> Tuple[bytes, dict]:
    """Compress column data based on schema."""
    compression = col_schema.compression
    
    if compression == "fp16":
        payload = compress_fp16(col_data)
        return payload, {"compression": "fp16"}
    
    elif compression == "int8":
        payload, minv, scale = compress_int8(col_data)
        return payload, {
            "compression": "int8",
            "min": float(minv),
            "scale": float(scale)
        }
    
    elif compression == "none":
        payload = col_data.tobytes()
        return payload, {"compression": "none"}
    
    else:
        raise ValueError(f"Unknown compression: {compression}")


def _write_columnar_file(
    output_path: str,
    schema: ColumnarFileSchema,
    chunks_meta: List[dict],
    chunk_payloads: List[bytes]
):
    """Write columnar CVC file to disk."""
    # Build header
    header_dict = schema.to_dict()
    header_dict["chunks"] = chunks_meta
    
    header_bytes = json.dumps(
        header_dict,
        sort_keys=True,
        separators=(',', ':')
    ).encode('utf-8')
    header_len = len(header_bytes)
    
    # Write file
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        f.write(HEADER_MAGIC)
        f.write(FORMAT_VERSION_MAJOR.to_bytes(2, byteorder='little'))
        f.write(FORMAT_VERSION_MINOR.to_bytes(2, byteorder='little'))
        f.write(header_len.to_bytes(4, byteorder='little'))
        f.write(header_bytes)
        
        for payload in chunk_payloads:
            checksum = zlib.crc32(payload) & 0xFFFFFFFF
            f.write(len(payload).to_bytes(4, byteorder='little'))
            f.write(checksum.to_bytes(4, byteorder='little'))
            f.write(payload)
