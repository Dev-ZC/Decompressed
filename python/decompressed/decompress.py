"""
Pure Python CPU decompression utilities for CVC format.

These are FALLBACK implementations used when C++ native extensions
from cvc/cvc.cpp are not built. The C++ versions (wrapped via pybind11)
are much faster and should be preferred when available.
"""

import numpy as np


def decompress_fp16_cpu(data: bytes, rows: int, dim: int) -> np.ndarray:
    """Pure Python FP16 decompression."""
    uint16_data = np.frombuffer(data, dtype=np.uint16)
    fp16_data = uint16_data.view(np.float16)
    return fp16_data.reshape(rows, dim).astype(np.float32)


def decompress_int8_cpu(data: bytes, rows: int, dim: int, minv: float, scale: float) -> np.ndarray:
    """Pure Python INT8 decompression."""
    uint8_data = np.frombuffer(data, dtype=np.uint8)
    float_data = uint8_data.astype(np.float32) * scale + minv
    return float_data.reshape(rows, dim)


def _byte_unshuffle(shuffled_data: bytes, n_values: int) -> bytes:
    """
    Reverse byte-shuffling to reconstruct float32 values (CPU version).
    
    This is the inverse of _byte_shuffle. Takes 4 contiguous byte planes
    and interleaves them back into the original float32 representation.
    
    This operation is embarrassingly parallel and will have a GPU
    equivalent in Triton for high-throughput decompression.
    
    Args:
        shuffled_data: Byte-shuffled data (4 planes)
        n_values: Number of float32 values
    
    Returns:
        Original byte layout
    """
    if len(shuffled_data) != n_values * 4:
        raise ValueError(
            f"Shuffled data size mismatch: expected {n_values * 4}, got {len(shuffled_data)}"
        )
    
    # Convert to numpy array
    byte_array = np.frombuffer(shuffled_data, dtype=np.uint8)
    
    # Reshape to (4, n_values) - currently all byte0s, then byte1s, etc.
    byte_planes = byte_array.reshape(4, n_values)
    
    # Transpose back to (n_values, 4)
    byte_matrix = byte_planes.T
    
    # Flatten to get original byte order
    return byte_matrix.flatten().tobytes()


def decompress_lossless_cpu(data: bytes, rows: int, dim: int) -> np.ndarray:
    """
    Pure Python lossless decompression (byte-unshuffle only).
    
    This is a simple memory swizzle operation that reverses the
    byte-shuffling compression. On CPU it's sequential, but on GPU
    it becomes massively parallel.
    
    Args:
        data: Byte-shuffled data (4 contiguous byte planes)
        rows: Number of vectors
        dim: Dimension of each vector
    
    Returns:
        Decompressed float32 array of shape (rows, dim)
    """
    n_values = rows * dim
    
    # Byte-unshuffle to restore original layout
    byte_data = _byte_unshuffle(data, n_values)
    
    # Convert back to float32
    float_data = np.frombuffer(byte_data, dtype=np.float32)
    
    return float_data.reshape(rows, dim)
