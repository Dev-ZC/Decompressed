"""CVC file validation utilities."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import zlib

HEADER_MAGIC = b"CVCF"


def validate_cvc(path: str) -> Dict[str, Any]:
    """
    Validate CVC file format and return metadata without loading vectors.
    
    This performs basic validation checks:
    - Magic number verification
    - JSON header parsing
    - Chunk count validation
    - File size consistency
    
    Args:
        path: Path to .cvc file
        
    Returns:
        dict: Validation report containing:
            - valid: bool - whether file passed all checks
            - version: str - format version (currently "0.1.0")
            - num_vectors: int - total number of vectors
            - dimension: int - vector dimensionality
            - compression: str - compression scheme
            - num_chunks: int - number of chunks
            - size_bytes: int - total file size
            - size_mb: float - file size in megabytes
            - errors: list[str] - any validation errors found
            - warnings: list[str] - any validation warnings
            
    Raises:
        FileNotFoundError: If file does not exist
        
    Example:
        >>> report = validate_cvc("embeddings.cvc")
        >>> if report['valid']:
        ...     print(f"Valid CVC file with {report['num_vectors']} vectors")
        ... else:
        ...     print(f"Validation errors: {report['errors']}")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    errors = []
    warnings = []
    metadata = {}
    
    try:
        with open(path, "rb") as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            f.seek(0)  # Seek back to start
            
            # Check magic number
            magic = f.read(4)
            if magic != HEADER_MAGIC:
                errors.append(f"Invalid magic number: expected {HEADER_MAGIC!r}, got {magic!r}")
                return {
                    'valid': False,
                    'version': 'unknown',
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'errors': errors,
                    'warnings': warnings,
                }
            
            # Detect version
            try:
                version_or_len = f.read(4)
                
                # Try to parse as v1.x first
                major = int.from_bytes(version_or_len[:2], "little")
                minor = int.from_bytes(version_or_len[2:4], "little")
                
                if 1 <= major <= 10 and 0 <= minor <= 100:
                    # v1.x format
                    version = f"v{major}.{minor}"
                    header_len = int.from_bytes(f.read(4), "little")
                else:
                    # v0.x format (version_or_len is header_len)
                    version = "v0.1"
                    header_len = int.from_bytes(version_or_len, "little")
                    warnings.append("Legacy v0.x format detected. Consider upgrading to v1.0")
                
                if header_len <= 0 or header_len > 100_000_000:  # 100MB sanity check
                    errors.append(f"Unreasonable header length: {header_len} bytes")
                    return {
                        'valid': False,
                        'version': 'unknown',
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'errors': errors,
                        'warnings': warnings,
                    }
                
                header_bytes = f.read(header_len)
                if len(header_bytes) != header_len:
                    errors.append(f"Truncated header: expected {header_len} bytes, got {len(header_bytes)}")
                    return {
                        'valid': False,
                        'version': 'unknown',
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'errors': errors,
                        'warnings': warnings,
                    }
                
                header = json.loads(header_bytes)
                
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON header: {e}")
                return {
                    'valid': False,
                    'version': 'unknown',
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'errors': errors,
                    'warnings': warnings,
                }
            
            # Validate required header fields
            required_fields = ['num_vectors', 'dimension', 'compression', 'chunks']
            for field in required_fields:
                if field not in header:
                    errors.append(f"Missing required header field: {field}")
            
            if errors:
                return {
                    'valid': False,
                    'version': '0.1.0',
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'errors': errors,
                    'warnings': warnings,
                }
            
            # Extract metadata
            num_vectors = header['num_vectors']
            dimension = header['dimension']
            compression = header['compression']
            chunks_meta = header['chunks']
            num_chunks = len(chunks_meta)
            
            # Validate compression type
            if compression not in ['fp16', 'int8', 'lossless']:
                warnings.append(f"Unknown compression type: {compression}")
            
            # Validate chunk metadata
            total_chunk_vectors = 0
            for i, chunk in enumerate(chunks_meta):
                if 'rows' not in chunk:
                    errors.append(f"Chunk {i} missing 'rows' field")
                else:
                    total_chunk_vectors += chunk['rows']
                
                chunk_compression = chunk.get('compression', compression)
                if chunk_compression == 'int8':
                    if 'min' not in chunk or 'scale' not in chunk:
                        errors.append(f"Chunk {i} with INT8 compression missing 'min' or 'scale'")
            
            # Validate total vectors match chunk sum
            if total_chunk_vectors != num_vectors:
                errors.append(
                    f"Vector count mismatch: header says {num_vectors}, "
                    f"chunks sum to {total_chunk_vectors}"
                )
            
            # Validate file size (check if all chunks are present)
            # For v1.x: magic(4) + version(4) + header_len(4) + header
            # For v0.x: magic(4) + header_len(4) + header
            is_v1_format = version.startswith("v1")
            if is_v1_format:
                header_size = 4 + 4 + 4 + header_len  # magic + version + header_len + header
                bytes_per_chunk = 8  # chunk_len(4) + checksum(4)
            else:
                header_size = 4 + 4 + header_len  # magic + header_len + header
                bytes_per_chunk = 4  # chunk_len(4) only
            
            expected_min_size = header_size + (num_chunks * bytes_per_chunk)
            
            if file_size < expected_min_size:
                errors.append(
                    f"File too small: {file_size} bytes, "
                    f"expected at least {expected_min_size} bytes"
                )
            
            # Try to validate chunk structure (without reading payloads)
            chunk_offset = header_size
            for i, chunk in enumerate(chunks_meta):
                if f.tell() != chunk_offset:
                    f.seek(chunk_offset)
                
                if f.tell() + 4 > file_size:
                    errors.append(f"Chunk {i} length header extends beyond file")
                    break
                
                chunk_len_bytes = f.read(4)
                if len(chunk_len_bytes) != 4:
                    errors.append(f"Truncated chunk {i} length header")
                    break
                
                chunk_len = int.from_bytes(chunk_len_bytes, "little")
                
                # v1.x has checksums, skip them
                if is_v1_format:
                    f.read(4)  # skip checksum
                
                chunk_offset = f.tell() + chunk_len
                
                if chunk_offset > file_size:
                    errors.append(
                        f"Chunk {i} payload extends beyond file: "
                        f"ends at {chunk_offset}, file is {file_size} bytes"
                    )
                    break
                
                # Skip to next chunk
                f.seek(chunk_offset)
            
            # Build result
            result = {
                'valid': len(errors) == 0,
                'version': version,
                'num_vectors': num_vectors,
                'dimension': dimension,
                'compression': compression,
                'num_chunks': num_chunks,
                'size_bytes': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'errors': errors,
                'warnings': warnings,
            }
            
            return result
            
    except Exception as e:
        errors.append(f"Unexpected error during validation: {e}")
        return {
            'valid': False,
            'version': 'unknown',
            'errors': errors,
            'warnings': warnings,
        }


def validate_cvc_integrity(path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate all chunk checksums in a CVC file without loading data.
    
    This performs deep integrity validation by checking CRC32 checksums
    for all chunks. Use this to verify file integrity after:
    - Network transfer
    - Storage migration
    - Object store caching
    
    Args:
        path: Path to .cvc file
        verbose: If True, print progress for each chunk
        
    Returns:
        dict: Integrity report containing:
            - valid: bool - whether all chunks passed checksum validation
            - total_chunks: int - number of chunks in file
            - corrupted_chunks: list[int] - indices of corrupted chunks
            - corrupted_ranges: list[tuple] - (start_idx, end_idx) for corrupted vectors
            - checksum_details: list[dict] - per-chunk checksum info
            
    Example:
        >>> report = validate_cvc_integrity("embeddings.cvc")
        >>> if report['valid']:
        ...     print("All chunks intact")
        ... else:
        ...     print(f"Corrupted chunks: {report['corrupted_chunks']}")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    corrupted_chunks = []
    corrupted_ranges = []
    checksum_details = []
    
    try:
        with open(path, "rb") as f:
            # Read header
            magic = f.read(4)
            if magic != HEADER_MAGIC:
                return {
                    'valid': False,
                    'error': f"Invalid magic number: {magic!r}",
                    'total_chunks': 0,
                    'corrupted_chunks': [],
                    'corrupted_ranges': [],
                }
            
            # Detect version
            version_or_len = f.read(4)
            major = int.from_bytes(version_or_len[:2], "little")
            minor = int.from_bytes(version_or_len[2:4], "little")
            
            if 1 <= major <= 10 and 0 <= minor <= 100:
                # v1.x format
                header_len = int.from_bytes(f.read(4), "little")
                is_v1 = True
            else:
                # v0.x format (no checksums!)
                header_len = int.from_bytes(version_or_len, "little")
                is_v1 = False
            
            header = json.loads(f.read(header_len))
            
            chunks_meta = header['chunks']
            total_chunks = len(chunks_meta)
            
            if verbose:
                print(f"Validating {total_chunks} chunks...")
            
            # Validate each chunk checksum
            vector_offset = 0
            
            if not is_v1:
                # v0.x files have no checksums - skip validation
                return {
                    'valid': False,
                    'error': "Cannot validate integrity of v0.x file (no checksums). Please upgrade to v1.0 first.",
                    'total_chunks': total_chunks,
                    'corrupted_chunks': [],
                    'corrupted_ranges': [],
                }
            
            for i, chunk in enumerate(chunks_meta):
                chunk_len = int.from_bytes(f.read(4), "little")
                expected_checksum = int.from_bytes(f.read(4), "little")
                payload = f.read(chunk_len)
                
                rows = chunk['rows']
                actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
                
                is_valid = actual_checksum == expected_checksum
                
                checksum_details.append({
                    'chunk_index': i,
                    'valid': is_valid,
                    'expected_crc32': f"{expected_checksum:08x}",
                    'actual_crc32': f"{actual_checksum:08x}",
                    'vector_range': (vector_offset, vector_offset + rows),
                })
                
                if not is_valid:
                    corrupted_chunks.append(i)
                    corrupted_ranges.append((vector_offset, vector_offset + rows))
                    
                    if verbose:
                        print(f"  ✗ Chunk {i}: CORRUPTED (vectors {vector_offset}-{vector_offset+rows})")
                elif verbose:
                    print(f"  ✓ Chunk {i}: OK")
                
                vector_offset += rows
            
            result = {
                'valid': len(corrupted_chunks) == 0,
                'total_chunks': total_chunks,
                'corrupted_chunks': corrupted_chunks,
                'corrupted_ranges': corrupted_ranges,
                'checksum_details': checksum_details if verbose else None,
            }
            
            return result
            
    except Exception as e:
        return {
            'valid': False,
            'error': f"Error during integrity check: {e}",
            'total_chunks': 0,
            'corrupted_chunks': [],
            'corrupted_ranges': [],
        }
