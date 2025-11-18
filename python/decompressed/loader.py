"""CVC file format loader with backend management."""

import json
import numpy as np
from pathlib import Path
import warnings

from .backends import PythonBackend, CPPBackend, CUDABackend, TritonBackend
from .utils import validate_backend_availability, select_backend, check_cuda_pytorch_compatibility

HEADER_MAGIC = b"CVCF"


class CVCLoader:
    """Manages CVC file loading with automatic backend selection."""
    
    def __init__(self):
        # Initialize all backends
        self.python_backend = PythonBackend()
        self.cpp_backend = CPPBackend()
        self.cuda_backend = CUDABackend()
        self.triton_backend = TritonBackend()
        
        # Check for CUDA/PyTorch compatibility and warn user
        self._check_gpu_compatibility()
    
    def _check_gpu_compatibility(self):
        """Check and warn about CUDA/PyTorch compatibility issues."""
        is_compat, sys_cuda, torch_cuda, fix_cmd = check_cuda_pytorch_compatibility()
        
        if not is_compat and sys_cuda and torch_cuda:
            # Only warn if we have both versions and they don't match
            if (self.cuda_backend.is_available() or self.triton_backend.is_available()):
                warnings.warn(
                    f"\n{'='*70}\n"
                    f"⚠️  CUDA/PYTORCH VERSION MISMATCH DETECTED\n"
                    f"{'='*70}\n"
                    f"System CUDA: {sys_cuda}\n"
                    f"PyTorch CUDA: {torch_cuda}\n\n"
                    f"GPU backends may fail with PTX errors. To fix:\n"
                    f"  {fix_cmd}\n"
                    f"Then restart your Python runtime.\n"
                    f"{'='*70}\n",
                    RuntimeWarning,
                    stacklevel=2
                )
    
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
    
    def get_info(self, path):
        """
        Read CVC file metadata without loading vectors.
        
        Args:
            path: Path to .cvc file
            
        Returns:
            dict: File metadata containing:
                - num_vectors: Total number of vectors
                - dimension: Vector dimensionality
                - compression: Default compression scheme
                - chunks: List of chunk metadata
                - num_chunks: Number of chunks
        """
        path = Path(path)
        with open(path, "rb") as f:
            header = self._read_header(f)
            header['num_chunks'] = len(header['chunks'])
            return {
                "num_vectors": header["num_vectors"],
                "dimension": header["dimension"],
                "compression": header["compression"],
                "chunks": [
                    {"index": i, "rows": chunk["rows"], "metadata": chunk.get("metadata")} 
                    for i, chunk in enumerate(header["chunks"])
                ],
                "num_chunks": len(header['chunks'])
            }
    
    def load_chunks(self, path, chunk_indices=None, device="cpu", framework="torch", backend="auto"):
        """
        Load specific chunks from a .cvc file.
        
        Args:
            path: Path to .cvc file
            chunk_indices: List of chunk indices to load (0-indexed), or None for all chunks
            device: "cpu" or "cuda"
            framework: "torch" or "cupy" (for GPU arrays)
            backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
            
        Yields:
            tuple: (chunk_index, chunk_array) for each requested chunk
        """
        path = Path(path)
        
        with open(path, "rb") as f:
            header = self._read_header(f)
            dim = header["dimension"]
            compression = header["compression"]
            chunks_meta = header["chunks"]
            
            # Determine which chunks to load
            if chunk_indices is None:
                chunk_indices = range(len(chunks_meta))
            else:
                # Validate chunk indices
                max_chunk = len(chunks_meta)
                for idx in chunk_indices:
                    if idx < 0 or idx >= max_chunk:
                        raise ValueError(f"Invalid chunk index {idx}. File has {max_chunk} chunks (indices 0-{max_chunk-1})")
            
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
            
            backend_instance = self._get_backend_instance(use_backend)
            
            # Read and decompress requested chunks
            for chunk_idx in range(len(chunks_meta)):
                chunk = chunks_meta[chunk_idx]
                chunk_len = int.from_bytes(f.read(4), "little")
                
                if chunk_idx in chunk_indices:
                    # Read and decompress this chunk
                    payload = f.read(chunk_len)
                    rows = chunk["rows"]
                    
                    # Allocate output array for this chunk
                    chunk_arr = self._allocate_output_array(rows, dim, device, framework)
                    
                    # Decompress chunk
                    if use_backend in ["cuda", "triton"]:
                        if use_backend == "triton" and backend == "auto":
                            backend_instance.decompress_chunk(
                                payload, rows, dim, compression, chunk,
                                chunk_arr, 0, framework=framework,
                                cuda_fallback=self.cuda_backend if self.cuda_backend.is_available() else None
                            )
                        else:
                            backend_instance.decompress_chunk(
                                payload, rows, dim, compression, chunk,
                                chunk_arr, 0, framework=framework
                            )
                    else:
                        backend_instance.decompress_chunk(
                            payload, rows, dim, compression, chunk,
                            chunk_arr, 0
                        )
                    
                    yield chunk_idx, chunk_arr
                else:
                    # Skip this chunk
                    f.seek(chunk_len, 1)  # Seek forward relative to current position
    
    def load_range(self, path, chunk_indices=None, device="cpu", framework="torch", backend="auto", metadata_key=None, metadata_value=None, section_key=None, section_value=None):
        """
        Load specific chunks from a .cvc file and concatenate them.
        
        Args:
            path: Path to .cvc file
            chunk_indices: List of chunk indices to load (0-indexed). If None and metadata filtering
                          is not used, loads all chunks.
            device: "cpu" or "cuda"
            framework: "torch" or "cupy" (for GPU arrays)
            backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
            metadata_key: Optional key to filter chunks by chunk metadata
            metadata_value: Value to match for metadata_key
            section_key: Optional key to filter by section metadata (for files packed with pack_cvc_sections)
            section_value: Value to match for section_key
            
        Returns:
            Array containing the requested chunks concatenated together
        """
        # Check for conflicting parameters
        if chunk_indices is not None and (metadata_key is not None or section_key is not None):
            raise ValueError("Cannot specify chunk_indices together with metadata or section filtering")
        
        if metadata_key is not None and section_key is not None:
            raise ValueError("Cannot specify both metadata_key and section_key filtering")
        
        # Section-based filtering (new approach)
        section_extraction_needed = False
        section_ranges = []  # List of (chunk_idx, start_in_chunk, end_in_chunk)
        
        if section_key is not None and section_value is not None:
            section_extraction_needed = True
            path_obj = Path(path)
            with open(path_obj, "rb") as f:
                header = self._read_header(f)
                
                # Check if file has sections
                if "sections" not in header:
                    raise ValueError("File does not have section metadata. Use pack_cvc_sections() to create files with sections.")
                
                # Find matching sections
                matching_sections = [
                    sec for sec in header["sections"]
                    if sec.get("metadata", {}).get(section_key) == section_value
                ]
                
                if not matching_sections:
                    raise ValueError(f"No sections found with {section_key}={section_value}")
                
                # Find all chunks that intersect with matching sections and extract ranges
                chunk_set = set()
                for chunk_idx, chunk in enumerate(header["chunks"]):
                    if "sections" in chunk:
                        for chunk_section in chunk["sections"]:
                            # Check if this chunk section matches our filter
                            if chunk_section.get("metadata", {}).get(section_key) == section_value:
                                chunk_set.add(chunk_idx)
                                section_ranges.append((
                                    chunk_idx,
                                    chunk_section["start_in_chunk"],
                                    chunk_section["end_in_chunk"]
                                ))
                                break
                
                chunk_indices = sorted(chunk_set)
                
                if not chunk_indices:
                    raise ValueError(f"No chunks found for sections with {section_key}={section_value}")
        
        # Chunk metadata filtering (original approach)
        elif metadata_key is not None and metadata_value is not None:
            # Get file info and filter by metadata
            info = self.get_info(path)
            chunk_indices = [
                chunk['index'] 
                for chunk in info['chunks']
                if chunk.get('metadata') and chunk['metadata'].get(metadata_key) == metadata_value
            ]
            
            if not chunk_indices:
                raise ValueError(f"No chunks found with {metadata_key}={metadata_value}")
        
        chunks = []
        
        # Load chunks
        for chunk_idx, chunk_arr in self.load_chunks(path, chunk_indices, device, framework, backend):
            # If section extraction is needed, extract only the relevant portion
            if section_extraction_needed:
                # Find the section range for this chunk
                for range_chunk_idx, start_in_chunk, end_in_chunk in section_ranges:
                    if range_chunk_idx == chunk_idx:
                        # Extract the section portion from this chunk
                        chunk_arr = chunk_arr[start_in_chunk:end_in_chunk]
                        break
            
            chunks.append(chunk_arr)
        
        if not chunks:
            raise ValueError("No chunks loaded")
        
        # Concatenate chunks
        if device == "cpu":
            return np.concatenate(chunks, axis=0)
        elif framework == "torch":
            import torch
            return torch.cat(chunks, dim=0)
        elif framework == "cupy":
            import cupy as cp
            return cp.concatenate(chunks, axis=0)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
