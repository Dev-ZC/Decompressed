"""CVC file format migration utilities."""

import json
import zlib
from pathlib import Path
from typing import Dict, Any, Optional

HEADER_MAGIC = b"CVCF"
FORMAT_VERSION_MAJOR = 1
FORMAT_VERSION_MINOR = 0


def detect_version(path: str) -> tuple[int, int]:
    """
    Detect the format version of a CVC file.
    
    Args:
        path: Path to .cvc file
        
    Returns:
        tuple: (major, minor) version numbers
        
    Example:
        >>> version = detect_version("embeddings.cvc")
        >>> print(f"Format version: v{version[0]}.{version[1]}")
    """
    path = Path(path)
    
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != HEADER_MAGIC:
            raise ValueError(f"Not a valid .cvc file: {path}")
        
        version_bytes = f.read(4)
        
        # Try to parse as v1.x
        major = int.from_bytes(version_bytes[:2], "little")
        minor = int.from_bytes(version_bytes[2:4], "little")
        
        # Check if this looks like version bytes or header_len
        # v1.x: major=1-10, minor=0-100
        # v0.x: header_len typically 100-10000
        if 1 <= major <= 10 and 0 <= minor <= 100:
            # Likely v1.x
            return (major, minor)
        else:
            # Likely v0.x (version_bytes is actually header_len)
            return (0, 1)


def upgrade_cvc(input_path: str, output_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Upgrade a v0.x CVC file to v1.0 format with versioning and checksums.
    
    This is a one-time migration that:
    1. Adds format version bytes (v1.0)
    2. Adds CRC32 checksums to all chunks (if missing)
    3. Preserves all data and metadata
    
    Args:
        input_path: Path to input CVC file (v0.x or v1.x)
        output_path: Path to output CVC file (will be v1.0)
        verbose: If True, print progress
        
    Returns:
        dict: Migration report with statistics
        
    Raises:
        ValueError: If input file is invalid
        FileExistsError: If output file already exists
        
    Example:
        >>> report = upgrade_cvc("old_embeddings.cvc", "new_embeddings.cvc")
        >>> print(f"Upgraded {report['num_chunks']} chunks")
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            f"Please remove it first or choose a different path."
        )
    
    # Detect input version
    input_version = detect_version(str(input_path))
    
    if verbose:
        print(f"Input file version: v{input_version[0]}.{input_version[1]}")
        print(f"Upgrading to: v{FORMAT_VERSION_MAJOR}.{FORMAT_VERSION_MINOR}")
    
    # Read input file
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != HEADER_MAGIC:
            raise ValueError("Not a valid .cvc file")
        
        # Read based on version
        version_or_len = f.read(4)
        
        if input_version[0] >= 1:
            # v1.x input - has version bytes
            header_len = int.from_bytes(f.read(4), "little")
        else:
            # v0.x input - version_or_len is header_len
            header_len = int.from_bytes(version_or_len, "little")
        
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes)
        
        num_chunks = len(header['chunks'])
        
        if verbose:
            print(f"Reading {num_chunks} chunks...")
        
        # Read all chunks and compute checksums
        chunks_data = []
        for i, chunk_meta in enumerate(header['chunks']):
            if input_version[0] >= 1:
                # v1.x has checksums
                chunk_len = int.from_bytes(f.read(4), "little")
                checksum = int.from_bytes(f.read(4), "little")
                payload = f.read(chunk_len)
                
                # Verify checksum
                actual_checksum = zlib.crc32(payload) & 0xFFFFFFFF
                if actual_checksum != checksum:
                    raise ValueError(
                        f"Chunk {i} corrupted in input file. "
                        f"Expected CRC32: {checksum:08x}, Got: {actual_checksum:08x}"
                    )
            else:
                # v0.x has no checksums
                chunk_len = int.from_bytes(f.read(4), "little")
                payload = f.read(chunk_len)
                
                # Compute new checksum
                checksum = zlib.crc32(payload) & 0xFFFFFFFF
            
            chunks_data.append((payload, checksum))
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_chunks} chunks")
    
    # Write output file in v1.0 format
    # Re-serialize header for determinism
    header_bytes_new = json.dumps(header, sort_keys=True, separators=(',', ':')).encode('utf-8')
    header_len_new = len(header_bytes_new)
    
    with open(output_path, "wb") as f:
        # Write v1.0 header
        f.write(HEADER_MAGIC)
        f.write(FORMAT_VERSION_MAJOR.to_bytes(2, byteorder='little'))
        f.write(FORMAT_VERSION_MINOR.to_bytes(2, byteorder='little'))
        f.write(header_len_new.to_bytes(4, byteorder='little'))
        f.write(header_bytes_new)
        
        # Write chunks with checksums
        for payload, checksum in chunks_data:
            f.write(len(payload).to_bytes(4, byteorder='little'))
            f.write(checksum.to_bytes(4, byteorder='little'))
            f.write(payload)
    
    if verbose:
        print(f"✓ Migration complete: {output_path}")
    
    return {
        'input_version': f"v{input_version[0]}.{input_version[1]}",
        'output_version': f"v{FORMAT_VERSION_MAJOR}.{FORMAT_VERSION_MINOR}",
        'num_chunks': num_chunks,
        'num_vectors': header['num_vectors'],
        'dimension': header['dimension'],
        'input_size_mb': round(input_path.stat().st_size / (1024 * 1024), 2),
        'output_size_mb': round(output_path.stat().st_size / (1024 * 1024), 2),
    }


def validate_cvc_version(path: str) -> Dict[str, Any]:
    """
    Validate CVC file and return version information.
    
    Args:
        path: Path to .cvc file
        
    Returns:
        dict: Version information and validation status
        
    Example:
        >>> info = validate_cvc_version("embeddings.cvc")
        >>> if info['needs_upgrade']:
        ...     print("File should be upgraded to v1.0")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        version = detect_version(str(path))
        
        return {
            'valid': True,
            'version': f"v{version[0]}.{version[1]}",
            'version_tuple': version,
            'needs_upgrade': version[0] == 0,
            'current_latest': f"v{FORMAT_VERSION_MAJOR}.{FORMAT_VERSION_MINOR}",
            'path': str(path),
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'path': str(path),
        }
