"""CPU-based decompression backends."""

import numpy as np
from .base import BackendInterface


class PythonBackend(BackendInterface):
    """Pure Python CPU decompression backend."""
    
    def __init__(self):
        from ..decompress import decompress_fp16_cpu, decompress_int8_cpu
        self.decompress_fp16 = decompress_fp16_cpu
        self.decompress_int8 = decompress_int8_cpu
    
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset):
        """Decompress using pure Python implementations."""
        if compression == "fp16":
            arr[offset:offset+rows] = self.decompress_fp16(payload, rows, dim)
        else:  # int8
            arr[offset:offset+rows] = self.decompress_int8(
                payload, rows, dim, chunk_meta["min"], chunk_meta["scale"]
            )
    
    def is_available(self):
        """Always available (pure Python)."""
        return True


class CPPBackend(BackendInterface):
    """C++ native CPU decompression backend."""
    
    def __init__(self):
        try:
            from decompressed._cvc_native import decompress_fp16_cpu, decompress_int8_cpu
            self.decompress_fp16 = decompress_fp16_cpu
            self.decompress_int8 = decompress_int8_cpu
            self._available = True
        except ImportError:
            self._available = False
    
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset):
        """Decompress using C++ native implementations."""
        if compression == "fp16":
            src_np = np.frombuffer(payload, dtype=np.uint16)
            result = self.decompress_fp16(src_np)
            arr[offset:offset+rows] = result.reshape(rows, dim)
        else:  # int8
            src_np = np.frombuffer(payload, dtype=np.uint8)
            result = self.decompress_int8(src_np, chunk_meta["min"], chunk_meta["scale"])
            arr[offset:offset+rows] = result.reshape(rows, dim)
    
    def is_available(self):
        """Check if C++ native extensions are built."""
        return self._available
