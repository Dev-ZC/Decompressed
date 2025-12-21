"""CPU-based decompression backends."""

import numpy as np
from .base import BackendInterface


class PythonBackend(BackendInterface):
    """Pure Python CPU decompression backend."""
    
    def __init__(self):
        from ..decompress import decompress_fp16_cpu, decompress_int8_cpu, decompress_lossless_cpu
        self.decompress_fp16 = decompress_fp16_cpu
        self.decompress_int8 = decompress_int8_cpu
        self.decompress_lossless = decompress_lossless_cpu
    
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset):
        """Decompress using pure Python implementations."""
        if compression == "fp16":
            arr[offset:offset+rows] = self.decompress_fp16(payload, rows, dim)
        elif compression == "int8":
            arr[offset:offset+rows] = self.decompress_int8(
                payload, rows, dim, chunk_meta["min"], chunk_meta["scale"]
            )
        elif compression == "lossless":
            arr[offset:offset+rows] = self.decompress_lossless(payload, rows, dim)
        else:
            raise ValueError(f"Unknown compression type: {compression}")
    
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
        
        # Lossless always uses Python fallback (no C++ implementation yet)
        from ..decompress import decompress_lossless_cpu
        self.decompress_lossless = decompress_lossless_cpu
    
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset):
        """Decompress using C++ native implementations."""
        if compression == "fp16":
            src_np = np.frombuffer(payload, dtype=np.uint16)
            result = self.decompress_fp16(src_np)
            arr[offset:offset+rows] = result.reshape(rows, dim)
        elif compression == "int8":
            src_np = np.frombuffer(payload, dtype=np.uint8)
            result = self.decompress_int8(src_np, chunk_meta["min"], chunk_meta["scale"])
            arr[offset:offset+rows] = result.reshape(rows, dim)
        elif compression == "lossless":
            arr[offset:offset+rows] = self.decompress_lossless(payload, rows, dim)
        else:
            raise ValueError(f"Unknown compression type: {compression}")
    
    def is_available(self):
        """Check if C++ native extensions are built."""
        return self._available
