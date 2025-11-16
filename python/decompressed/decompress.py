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
