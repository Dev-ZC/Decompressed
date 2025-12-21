"""Triton kernel for GPU-native lossless (bit-pack + byte-shuffle) decompression."""

import triton
import triton.language as tl
import numpy as np


@triton.jit
def bit_unpack_kernel(
    packed_ptr,      # Pointer to bit-packed data
    dict_ptr,        # Pointer to dictionary (unique values)
    output_ptr,      # Pointer to output byte plane
    n_values,        # Number of values to unpack
    bits_per_value: tl.constexpr,  # Bits per value (1-8)
    BLOCK_SIZE: tl.constexpr
):
    """
    GPU-native bit-unpacking kernel.
    
    Each thread unpacks one value by:
    1. Calculating bit offset
    2. Reading packed bits (may span bytes)
    3. Looking up in dictionary
    4. Writing decoded byte
    
    Args:
        packed_ptr: Bit-packed data
        dict_ptr: Dictionary for decoding
        output_ptr: Output plane (unpacked bytes)
        n_values: Number of values
        bits_per_value: Bits per value
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_values
    
    # Calculate bit offset for each value
    bit_offset = idx * bits_per_value
    byte_idx = bit_offset // 8
    bit_in_byte = bit_offset % 8
    
    # Read value (may span up to 2 bytes)
    byte0 = tl.load(packed_ptr + byte_idx, mask=mask, other=0).to(tl.uint32)
    byte1 = tl.load(packed_ptr + byte_idx + 1, mask=mask & ((bit_in_byte + bits_per_value) > 8), other=0).to(tl.uint32)
    
    # Extract bits from byte stream
    # Combine two bytes and extract the middle bits_per_value bits
    combined = (byte0 << 8) | byte1
    shift = 16 - bit_in_byte - bits_per_value
    bitmask = (1 << bits_per_value) - 1
    code = (combined >> shift) & bitmask
    
    # Dictionary lookup
    decoded = tl.load(dict_ptr + code, mask=mask, other=0).to(tl.uint8)
    
    # Store result
    tl.store(output_ptr + idx, decoded, mask=mask)


@triton.jit
def byte_unshuffle_kernel(
    plane0_ptr,      # Pointer to plane 0 (unpacked bytes)
    plane1_ptr,      # Pointer to plane 1
    plane2_ptr,      # Pointer to plane 2
    plane3_ptr,      # Pointer to plane 3
    output_ptr,      # Pointer to output float32 array
    n_values,        # Total number of float32 values
    BLOCK_SIZE: tl.constexpr
):
    """
    GPU-native byte-unshuffle kernel.
    
    Takes 4 separate byte planes and reconstructs float32 values
    in parallel. Each thread handles one float32 value independently.
    
    Args:
        plane0_ptr through plane3_ptr: Input byte planes
        output_ptr: Output float32 array
        n_values: Number of float32 values
        BLOCK_SIZE: Block size
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_values
    
    # Load bytes from each plane
    byte0 = tl.load(plane0_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    byte1 = tl.load(plane1_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    byte2 = tl.load(plane2_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    byte3 = tl.load(plane3_ptr + offsets, mask=mask, other=0).to(tl.uint8)
    
    # Reconstruct 32-bit integer from 4 bytes (little-endian)
    reconstructed = (
        byte0.to(tl.uint32) |
        (byte1.to(tl.uint32) << 8) |
        (byte2.to(tl.uint32) << 16) |
        (byte3.to(tl.uint32) << 24)
    )
    
    # Reinterpret as float32
    float_value = reconstructed.to(tl.float32, bitcast=True)
    
    # Store result
    tl.store(output_ptr + offsets, float_value, mask=mask)


def decompress_lossless_triton(compressed_data, rows, dim, framework="torch"):
    """
    High-level wrapper for Triton bit-unpack + byte-unshuffle decompression.
    
    Args:
        compressed_data: Bit-packed byte-shuffled data with header (bytes)
        rows: Number of vectors
        dim: Dimension of each vector
        framework: "torch" or "cupy"
    
    Returns:
        Decompressed float32 tensor on GPU
    """
    import torch
    
    n_values = rows * dim
    
    # Parse header (on CPU)
    offset = 0
    stored_n_values = int.from_bytes(compressed_data[offset:offset+4], 'little')
    offset += 4
    
    if stored_n_values != n_values:
        raise ValueError(f"Value count mismatch: expected {n_values}, got {stored_n_values}")
    
    bits_per_plane = list(compressed_data[offset:offset+4])
    offset += 4
    
    # Allocate GPU buffers for unpacked planes
    plane_buffers = []
    dictionaries = []
    packed_data_list = []
    
    for plane_idx in range(4):
        # Read dictionary
        dict_size = int.from_bytes(compressed_data[offset:offset+2], 'little')
        offset += 2
        
        dictionary = np.frombuffer(compressed_data[offset:offset+dict_size], dtype=np.uint8)
        offset += dict_size
        
        # Calculate packed data size
        bits_needed = bits_per_plane[plane_idx]
        packed_size = (n_values * bits_needed + 7) // 8
        
        packed_data = compressed_data[offset:offset+packed_size]
        offset += packed_size
        
        # Upload to GPU
        dict_gpu = torch.from_numpy(dictionary).cuda()
        packed_gpu = torch.from_numpy(np.frombuffer(packed_data, dtype=np.uint8)).cuda()
        plane_gpu = torch.empty(n_values, dtype=torch.uint8, device='cuda')
        
        dictionaries.append(dict_gpu)
        packed_data_list.append(packed_gpu)
        plane_buffers.append(plane_gpu)
    
    # Step 1: Bit-unpack each plane on GPU
    BLOCK_SIZE = 1024
    for plane_idx in range(4):
        bits_needed = bits_per_plane[plane_idx]
        grid = lambda meta: (triton.cdiv(n_values, BLOCK_SIZE),)
        
        bit_unpack_kernel[grid](
            packed_data_list[plane_idx],
            dictionaries[plane_idx],
            plane_buffers[plane_idx],
            n_values,
            bits_needed,
            BLOCK_SIZE
        )
    
    # Step 2: Byte-unshuffle on GPU
    output = torch.empty(n_values, dtype=torch.float32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_values, BLOCK_SIZE),)
    
    byte_unshuffle_kernel[grid](
        plane_buffers[0],
        plane_buffers[1],
        plane_buffers[2],
        plane_buffers[3],
        output,
        n_values,
        BLOCK_SIZE
    )
    
    torch.cuda.synchronize()
    
    if framework == "torch":
        return output.reshape(rows, dim)
    elif framework == "cupy":
        import cupy as cp
        return cp.asarray(output).reshape(rows, dim)
    else:
        raise ValueError(f"Unsupported framework: {framework}")
