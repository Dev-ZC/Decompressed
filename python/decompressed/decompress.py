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


def _bit_unpack_plane(packed_data: bytes, n_values: int, bits_per_value: int) -> np.ndarray:
    """
    Unpack bit-packed plane back to full bytes.
    
    Args:
        packed_data: Bit-packed bytes
        n_values: Number of values to unpack
        bits_per_value: Number of bits per value (1-8)
    
    Returns:
        Unpacked uint8 array
    """
    if bits_per_value == 8:
        # No unpacking needed
        return np.frombuffer(packed_data, dtype=np.uint8)
    
    packed = np.frombuffer(packed_data, dtype=np.uint8)
    unpacked = np.zeros(n_values, dtype=np.uint8)
    
    bit_offset = 0
    for i in range(n_values):
        value = 0
        remaining_bits = bits_per_value
        
        while remaining_bits > 0:
            byte_idx = bit_offset // 8
            bit_in_byte = bit_offset % 8
            bits_to_read = min(8 - bit_in_byte, remaining_bits)
            
            # Extract bits
            shift = 8 - bit_in_byte - bits_to_read
            mask = ((1 << bits_to_read) - 1) << shift
            bits = (packed[byte_idx] & mask) >> shift
            
            # Add to value
            value = (value << bits_to_read) | bits
            
            remaining_bits -= bits_to_read
            bit_offset += bits_to_read
        
        unpacked[i] = value
    
    return unpacked


def decompress_lossless_cpu(data: bytes, rows: int, dim: int) -> np.ndarray:
    """
    Pure Python lossless decompression (bit-unpack + byte-unshuffle).
    
    This reverses the compression: bit-unpacking each plane, then
    byte-unshuffling to restore the original float32 layout.
    
    Args:
        data: Bit-packed byte-shuffled data with header
        rows: Number of vectors
        dim: Dimension of each vector
    
    Returns:
        Decompressed float32 array of shape (rows, dim)
    """
    n_values = rows * dim
    
    # Parse header
    offset = 0
    
    # Read n_values
    stored_n_values = int.from_bytes(data[offset:offset+4], 'little')
    offset += 4
    
    if stored_n_values != n_values:
        raise ValueError(f"Value count mismatch: expected {n_values}, got {stored_n_values}")
    
    # Read bits per plane
    bits_per_plane = list(data[offset:offset+4])
    offset += 4
    
    # Unpack each plane
    byte_planes = []
    for plane_idx in range(4):
        # Read dictionary
        dict_size = int.from_bytes(data[offset:offset+2], 'little')
        offset += 2
        
        dictionary = np.frombuffer(data[offset:offset+dict_size], dtype=np.uint8)
        offset += dict_size
        
        # Calculate packed data size
        bits_needed = bits_per_plane[plane_idx]
        packed_size = (n_values * bits_needed + 7) // 8
        
        packed_data = data[offset:offset+packed_size]
        offset += packed_size
        
        # Bit-unpack
        codes = _bit_unpack_plane(packed_data, n_values, bits_needed)
        
        # Decode using dictionary
        plane = dictionary[codes]
        byte_planes.append(plane)
    
    # Stack planes and byte-unshuffle
    byte_planes_array = np.stack(byte_planes, axis=0)  # Shape: (4, n_values)
    byte_matrix = byte_planes_array.T  # Shape: (n_values, 4)
    byte_data = byte_matrix.flatten().tobytes()
    
    # Convert back to float32
    float_data = np.frombuffer(byte_data, dtype=np.float32)
    
    return float_data.reshape(rows, dim)
