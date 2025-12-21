"""Triton kernel for GPU-native lossless (byte-shuffle) decompression."""

import triton
import triton.language as tl


@triton.jit
def byte_unshuffle_kernel(
    shuffled_ptr,    # Pointer to byte-shuffled data (4 planes)
    output_ptr,      # Pointer to output float32 array
    n_values,        # Total number of float32 values
    BLOCK_SIZE: tl.constexpr
):
    """
    GPU-native byte-unshuffle kernel for lossless decompression.
    
    Takes 4 contiguous byte planes and reconstructs float32 values
    in parallel. This is embarrassingly parallel - each thread handles
    one float32 value independently.
    
    Memory layout:
    - Input:  [all byte0s | all byte1s | all byte2s | all byte3s]
    - Output: [b0 b1 b2 b3 | b0 b1 b2 b3 | ...] (interleaved)
    
    Args:
        shuffled_ptr: Input byte-shuffled data (n_values * 4 bytes)
        output_ptr: Output float32 array (n_values floats)
        n_values: Number of float32 values to reconstruct
        BLOCK_SIZE: Number of values per thread block
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_values
    
    # Calculate byte plane offsets
    # Each plane contains n_values bytes
    plane0_offset = offsets              # Byte 0 plane (LSB)
    plane1_offset = offsets + n_values   # Byte 1 plane
    plane2_offset = offsets + 2 * n_values  # Byte 2 plane
    plane3_offset = offsets + 3 * n_values  # Byte 3 plane (MSB)
    
    # Load bytes from each plane
    byte0 = tl.load(shuffled_ptr + plane0_offset, mask=mask, other=0).to(tl.uint8)
    byte1 = tl.load(shuffled_ptr + plane1_offset, mask=mask, other=0).to(tl.uint8)
    byte2 = tl.load(shuffled_ptr + plane2_offset, mask=mask, other=0).to(tl.uint8)
    byte3 = tl.load(shuffled_ptr + plane3_offset, mask=mask, other=0).to(tl.uint8)
    
    # Reconstruct 32-bit integer from 4 bytes (little-endian)
    # byte0 is LSB, byte3 is MSB
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


def decompress_lossless_triton(shuffled_data, rows, dim, framework="torch"):
    """
    High-level wrapper for Triton byte-unshuffle decompression.
    
    Args:
        shuffled_data: Input byte-shuffled data (bytes or GPU tensor)
        rows: Number of vectors
        dim: Dimension of each vector
        framework: "torch" or "cupy"
    
    Returns:
        Decompressed float32 tensor on GPU
    """
    import numpy as np
    
    n_values = rows * dim
    
    if framework == "torch":
        import torch
        
        # Convert to GPU tensor if needed
        if isinstance(shuffled_data, bytes):
            shuffled_np = np.frombuffer(shuffled_data, dtype=np.uint8)
            shuffled_gpu = torch.from_numpy(shuffled_np).cuda()
        else:
            shuffled_gpu = shuffled_data
        
        # Allocate output
        output = torch.empty(n_values, dtype=torch.float32, device='cuda')
        
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_values, BLOCK_SIZE),)
        
        byte_unshuffle_kernel[grid](
            shuffled_gpu,
            output,
            n_values,
            BLOCK_SIZE
        )
        
        torch.cuda.synchronize()
        return output.reshape(rows, dim)
    
    elif framework == "cupy":
        import cupy as cp
        import torch
        
        # Convert to GPU tensor
        if isinstance(shuffled_data, bytes):
            shuffled_np = np.frombuffer(shuffled_data, dtype=np.uint8)
            shuffled_gpu = cp.asarray(shuffled_np)
        else:
            shuffled_gpu = shuffled_data
        
        # Triton requires PyTorch tensors, so convert CuPy -> PyTorch
        shuffled_torch = torch.as_tensor(shuffled_gpu, device='cuda')
        output_torch = torch.empty(n_values, dtype=torch.float32, device='cuda')
        
        # Launch kernel
        BLOCK_SIZE = 1024
        grid = lambda meta: (triton.cdiv(n_values, BLOCK_SIZE),)
        
        byte_unshuffle_kernel[grid](
            shuffled_torch,
            output_torch,
            n_values,
            BLOCK_SIZE
        )
        
        torch.cuda.synchronize()
        
        # Convert back to CuPy
        output_cupy = cp.asarray(output_torch)
        return output_cupy.reshape(rows, dim)
    
    else:
        raise ValueError(f"Unsupported framework: {framework}")
