"""Base interface for decompression backends."""

from abc import ABC, abstractmethod
import numpy as np


class BackendInterface(ABC):
    """Abstract base class for decompression backends."""
    
    @abstractmethod
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset):
        """
        Decompress a chunk of data.
        
        Args:
            payload: Raw bytes to decompress
            rows: Number of rows in chunk
            dim: Dimension of vectors
            compression: Compression type ("fp16" or "int8")
            chunk_meta: Chunk metadata dict (contains "min", "scale" for int8)
            arr: Output array to write into
            offset: Row offset in output array
        """
        pass
    
    @abstractmethod
    def is_available(self):
        """Check if this backend is available."""
        pass
