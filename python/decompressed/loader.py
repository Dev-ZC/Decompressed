"""CVC file format loader with backend management."""

import json
import numpy as np
from pathlib import Path

from .backends import PythonBackend, CPPBackend, CUDABackend, TritonBackend
from .utils import validate_backend_availability, select_backend

HEADER_MAGIC = b"CVCF"


class CVCLoader:
    """Manages CVC file loading with automatic backend selection."""
    
    def __init__(self):
        # Initialize all backends
        self.python_backend = PythonBackend()
        self.cpp_backend = CPPBackend()
        self.cuda_backend = CUDABackend()
        self.triton_backend = TritonBackend()
    
    def get_backend_availability(self):
        """Get dict of available backends."""
        return {
            'python': self.python_backend.is_available(),
            'cpp': self.cpp_backend.is_available(),
            'cuda': self.cuda_backend.is_available(),
            'triton': self.triton_backend.is_available(),
        }
    
    def load(self, path, device="cpu", framework="torch", backend="auto"):
        """
        Load a .cvc file into a GPU or CPU array.
        
        Args:
            path: Path to .cvc file
            device: "cpu" or "cuda" (GPU)
            framework: "torch" or "cupy" (for GPU arrays)
            backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
                - "auto": Use best available (cuda > cpp > triton > python)
                - "python": Pure Python (CPU only, slowest)
                - "cpp": C++ native (CPU only, fast)
                - "cuda": CUDA native (GPU only, fastest, NVIDIA only)
                - "triton": Triton kernels (GPU only, fast, vendor-agnostic)
        
        Returns:
            Array of vectors (numpy, torch, or cupy depending on device/framework)
        """
        path = Path(path)
        
        # Read header
        with open(path, "rb") as f:
            header = self._read_header(f)
            n_vectors = header["num_vectors"]
            dim = header["dimension"]
            compression = header["compression"]
            chunks_meta = header["chunks"]
            
            # Allocate output array
            arr = self._allocate_output_array(n_vectors, dim, device, framework)
            
            # Select and validate backend
            availability = self.get_backend_availability()
            use_backend = select_backend(
                backend, device,
                availability['cpp'],
                availability['cuda'],
                availability['triton']
            )
            
            validate_backend_availability(
                use_backend, device,
                availability['cpp'],
                availability['cuda'],
                availability['triton']
            )
            
            # Get backend instance
            backend_instance = self._get_backend_instance(use_backend)
            
            # Decompress chunks
            offset = 0
            for chunk in chunks_meta:
                chunk_len = int.from_bytes(f.read(4), "little")
                payload = f.read(chunk_len)
                rows = chunk["rows"]
                
                # Decompress chunk using selected backend
                if use_backend in ["cuda", "triton"]:
                    # GPU backends need framework parameter
                    if use_backend == "triton" and backend == "auto":
                        # Pass CUDA backend as fallback for auto mode
                        backend_instance.decompress_chunk(
                            payload, rows, dim, compression, chunk,
                            arr, offset, framework=framework,
                            cuda_fallback=self.cuda_backend if self.cuda_backend.is_available() else None
                        )
                    else:
                        backend_instance.decompress_chunk(
                            payload, rows, dim, compression, chunk,
                            arr, offset, framework=framework
                        )
                else:
                    # CPU backends
                    backend_instance.decompress_chunk(
                        payload, rows, dim, compression, chunk,
                        arr, offset
                    )
                
                offset += rows
        
        return arr
    
    def _read_header(self, f):
        """Read and parse CVC file header."""
        magic = f.read(4)
        if magic != HEADER_MAGIC:
            raise ValueError("Not a valid .cvc file")
        
        header_len = int.from_bytes(f.read(4), "little")
        header = json.loads(f.read(header_len))
        return header
    
    def _allocate_output_array(self, n_vectors, dim, device, framework):
        """Allocate output array based on device and framework."""
        if device == "cpu":
            return np.empty((n_vectors, dim), dtype=np.float32)
        else:
            if framework == "cupy":
                import cupy as cp
                return cp.zeros((n_vectors, dim), dtype=cp.float32)
            elif framework == "torch":
                import torch
                return torch.zeros((n_vectors, dim), dtype=torch.float32, device="cuda")
            else:
                raise ValueError(f"Unsupported framework: {framework}")
    
    def _get_backend_instance(self, backend_name):
        """Get backend instance by name."""
        backend_map = {
            'python': self.python_backend,
            'cpp': self.cpp_backend,
            'cuda': self.cuda_backend,
            'triton': self.triton_backend,
        }
        return backend_map[backend_name]
