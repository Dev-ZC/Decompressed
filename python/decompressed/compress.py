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
