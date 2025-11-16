"""Compression utilities for CVC format."""

import numpy as np


def compress_fp16(vectors: np.ndarray) -> bytes:
    """Compress vectors to FP16 format."""
    fp16_data = vectors.astype(np.float16)
    return fp16_data.view(np.uint16).tobytes()


def compress_int8(vectors: np.ndarray) -> tuple[bytes, float, float]:
    """Compress vectors to INT8 format with quantization."""
    minv = float(np.min(vectors))
    maxv = float(np.max(vectors))
    
    scale = (maxv - minv) / 255.0 if maxv != minv else 1.0
    
    quantized = np.round((vectors - minv) / scale).astype(np.uint8)
    
    return quantized.tobytes(), minv, scale
