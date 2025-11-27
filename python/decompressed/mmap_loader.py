"""Memory-mapped CVC loader for zero-copy access."""

import mmap
import numpy as np
from pathlib import Path
from typing import Optional
import json
import zlib

from .decompress import decompress_fp16_cpu, decompress_int8_cpu

HEADER_MAGIC = b"CVCF"


class MMapCVCLoader:
    """
    Zero-copy memory-mapped CVC loader.
    
    Provides direct access to compressed data via OS page cache,
    eliminating Python buffering and copies.
    """
    
    def __init__(self, path: str, mode: str = 'r'):
        """
        Initialize memory-mapped loader.
        
        Args:
            path: Path to CVC file
            mode: File mode ('r' for read-only)
        """
        self.path = Path(path)
        self.file = None
        self.mmap = None
        self.header = None
        self.is_mmap_optimized = False
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.close()
    
    def open(self):
        """Open file for memory-mapped access."""
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        
        # Open file
        self.file = open(self.path, 'rb')
        
        # Create memory map (read-only)
        self.mmap = mmap.mmap(
            self.file.fileno(),
            0,
            access=mmap.ACCESS_READ
        )
        
        # Read header
        self.header = self._read_header()
        
        # Check if file is mmap-optimized (has file offsets)
        if self.header.get("mmap_optimized") or (
            self.header.get("chunks") and 
            "file_offset" in self.header["chunks"][0]
        ):
            self.is_mmap_optimized = True
    
    def close(self):
        """Close mmap and file."""
        if self.mmap:
            self.mmap.close()
            self.mmap = None
        if self.file:
            self.file.close()
            self.file = None
    
    def get_chunk_view(self, chunk_idx: int) -> memoryview:
        """
        Get zero-copy view of chunk payload.
        
        Args:
            chunk_idx: Chunk index (0-indexed)
        
        Returns:
            memoryview: Zero-copy view into mmap'd region
        """
        if self.mmap is None:
            raise RuntimeError("File not opened. Call open() first.")
        
        chunk_meta = self.header["chunks"][chunk_idx]
        
        if self.is_mmap_optimized:
            # Fast path: use pre-computed offset
            offset = chunk_meta["file_offset"]
        else:
            # Slow path: compute offset by scanning
            offset = self._compute_chunk_offset(chunk_idx)
        
        # Read chunk length (4 bytes)
        chunk_len = int.from_bytes(self.mmap[offset:offset+4], "little")
        
        # Skip length (4 bytes) and checksum (4 bytes) = 8 bytes total
        payload_offset = offset + 8
        
        # Return zero-copy view of payload
        return memoryview(self.mmap[payload_offset:payload_offset + chunk_len])
    
    def load_chunk(self, chunk_idx: int, device: str = "cpu", framework: str = "torch") -> np.ndarray:
        """
        Load and decompress a single chunk.
        
        Args:
            chunk_idx: Chunk index
            device: "cpu" or "cuda"
            framework: "torch" or "cupy" (for GPU)
        
        Returns:
            Decompressed chunk data
        """
        # Get zero-copy view
        payload_view = self.get_chunk_view(chunk_idx)
        
        # Get chunk metadata
        chunk_meta = self.header["chunks"][chunk_idx]
        rows = chunk_meta["rows"]
        dim = self.header["dimension"]
        compression = chunk_meta.get("compression", self.header.get("compression"))
        
        # Decompress (currently CPU-only for mmap)
        # TODO: Add GPU support with pinned memory
        if compression == "fp16":
            # Convert memoryview to bytes for decompression
            payload_bytes = bytes(payload_view)
            chunk_data = decompress_fp16_cpu(payload_bytes, rows, dim)
        elif compression == "int8":
            payload_bytes = bytes(payload_view)
            minv = chunk_meta["min"]
            scale = chunk_meta["scale"]
            chunk_data = decompress_int8_cpu(payload_bytes, rows, dim, minv, scale)
        else:
            raise ValueError(f"Unknown compression: {compression}")
        
        # Convert to requested device/framework
        if device == "cuda":
            if framework == "torch":
                import torch
                return torch.from_numpy(chunk_data).cuda()
            elif framework == "cupy":
                import cupy as cp
                return cp.asarray(chunk_data)
        
        return chunk_data
    
    def load(self, device: str = "cpu", framework: str = "torch") -> np.ndarray:
        """
        Load entire file using memory-mapped access.
        
        This streams chunks lazily from the mmap'd region,
        reducing memory footprint compared to loading entire file at once.
        
        Args:
            device: "cpu" or "cuda"
            framework: "torch" or "cupy"
        
        Returns:
            Full decompressed array
        """
        num_vectors = self.header["num_vectors"]
        dim = self.header["dimension"]
        
        # Allocate output
        if device == "cpu":
            output = np.empty((num_vectors, dim), dtype=np.float32)
        elif device == "cuda":
            if framework == "torch":
                import torch
                output = torch.empty((num_vectors, dim), dtype=torch.float32, device="cuda")
            elif framework == "cupy":
                import cupy as cp
                output = cp.empty((num_vectors, dim), dtype=cp.float32)
        
        # Load chunks
        offset = 0
        for chunk_idx in range(len(self.header["chunks"])):
            chunk_data = self.load_chunk(chunk_idx, device="cpu", framework="torch")  # Always CPU first
            rows = chunk_data.shape[0]
            
            # Copy to output
            if device == "cpu":
                output[offset:offset+rows] = chunk_data
            else:
                # Transfer to GPU
                if framework == "torch":
                    import torch
                    output[offset:offset+rows] = torch.from_numpy(chunk_data).cuda()
                elif framework == "cupy":
                    import cupy as cp
                    output[offset:offset+rows] = cp.asarray(chunk_data)
            
            offset += rows
        
        return output
    
    def _read_header(self) -> dict:
        """Read and parse CVC file header from mmap."""
        # Read magic
        magic = bytes(self.mmap[0:4])
        if magic != HEADER_MAGIC:
            raise ValueError("Not a valid .cvc file")
        
        # Read version
        major = int.from_bytes(self.mmap[4:6], "little")
        minor = int.from_bytes(self.mmap[6:8], "little")
        
        # Read header length
        header_len = int.from_bytes(self.mmap[8:12], "little")
        
        # Read header JSON
        header_bytes = bytes(self.mmap[12:12+header_len])
        header = json.loads(header_bytes)
        
        header['_format_version'] = (major, minor)
        header['_header_end_offset'] = 12 + header_len
        
        return header
    
    def _compute_chunk_offset(self, chunk_idx: int) -> int:
        """
        Compute chunk offset by scanning file (fallback for non-optimized files).
        
        Args:
            chunk_idx: Target chunk index
        
        Returns:
            File offset where chunk starts
        """
        # Start after header
        offset = self.header['_header_end_offset']
        
        # Scan through chunks
        for i in range(chunk_idx):
            # Read chunk length
            chunk_len = int.from_bytes(self.mmap[offset:offset+4], "little")
            # Skip: length (4) + checksum (4) + payload (chunk_len)
            offset += 8 + chunk_len
        
        return offset
    
    def get_info(self) -> dict:
        """Get file metadata."""
        if self.header is None:
            raise RuntimeError("File not opened")
        
        return {
            "num_vectors": self.header["num_vectors"],
            "dimension": self.header["dimension"],
            "compression": self.header.get("compression"),
            "num_chunks": len(self.header["chunks"]),
            "mmap_optimized": self.is_mmap_optimized,
            "file_size": self.path.stat().st_size
        }
