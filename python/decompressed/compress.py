"""Compression utilities for CVC format."""

import numpy as np


def compress_fp16(vectors: np.ndarray) -> bytes:
    """Compress vectors to FP16 format."""
    fp16_data = vectors.astype(np.float16)
    return fp16_data.view(np.uint16).tobytes()


def compress_int8(vectors: np.ndarray) -> tuple[bytes, float, float]:
    """
    Compress vectors to INT8 format with quantization.
    
    This function is deterministic: given the same input, it will always
    produce the same output bytes, which is critical for:
    - CI/CD pipelines
    - Data versioning
    - Caching and deduplication
    """
    # Use deterministic reduction and consistent precision
    minv = float(np.min(vectors, axis=None, keepdims=False))
    maxv = float(np.max(vectors, axis=None, keepdims=False))
    
    # Round to float32 for consistency across platforms/runs
    minv = np.float32(minv)
    maxv = np.float32(maxv)
    
    # Compute scale with consistent precision
    scale = np.float32((maxv - minv) / 255.0) if maxv != minv else np.float32(1.0)
    
    # Deterministic quantization using float32 precision
    quantized = np.round((vectors.astype(np.float32) - minv) / scale).astype(np.uint8)
    
    # Convert to native Python float for JSON consistency
    return quantized.tobytes(), float(minv), float(scale)


def _byte_shuffle(vectors: np.ndarray) -> bytes:
    """
    Byte-shuffle float32 vectors for GPU-native lossless compression.
    
    Rearranges 4-byte float32 values so that all byte0s are together,
    all byte1s together, etc. This transformation is:
    - Embarrassingly parallel (perfect for GPU/Triton)
    - Lossless (100% reversible)
    - Exposes redundancy in FP32 representation
    
    In embeddings, bytes 2-3 (sign/exponent) are highly redundant,
    while bytes 0-1 (mantissa) are noisy. Grouping them separately
    enables better compression ratios.
    
    Args:
        vectors: np.ndarray of float32 values
    
    Returns:
        Shuffled bytes (4 contiguous planes)
    """
    # Convert to bytes
    byte_data = vectors.astype(np.float32).tobytes()
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    
    # Reshape to (n_values, 4) where 4 is bytes per float32
    n_bytes = len(byte_array)
    if n_bytes % 4 != 0:
        raise ValueError("Data must be aligned to 4-byte float32 values")
    
    byte_matrix = byte_array.reshape(-1, 4)
    
    # Transpose: all byte0s, then all byte1s, then all byte2s, then all byte3s
    # This creates 4 contiguous "planes" that can be processed in parallel
    shuffled = byte_matrix.T.flatten()
    
    return shuffled.tobytes()


def compress_lossless(vectors: np.ndarray) -> bytes:
    """
    Lossless compression using byte-shuffling only (GPU-native).
    
    This compression:
    - Preserves 100% of the original bits (truly lossless)
    - GPU-native: Triton can decompress in parallel at high throughput
    - Vendor-agnostic: Works on NVIDIA, AMD, Intel via Triton
    - Compression ratio: Typically 1:1 raw, but enables downstream compression
    
    Algorithm:
    - Byte-shuffle: Transpose float32 bytes into 4 separate planes
    - Each plane can be further compressed (e.g., with zlib at file level)
    - Decompression is a simple parallel memory swizzle in GPU SRAM
    
    For embeddings:
    - Byte 3 (sign/exponent): Very predictable
    - Byte 2 (exponent/mantissa): Highly redundant  
    - Bytes 0-1 (low mantissa): Mostly noise
    
    Args:
        vectors: np.ndarray of shape (n_vectors, dimension), dtype float32
    
    Returns:
        Byte-shuffled data (ready for GPU-native decompression)
    """
    # Byte-shuffle only - no RLE needed for GPU parallelism
    return _byte_shuffle(vectors)
