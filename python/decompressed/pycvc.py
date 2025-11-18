"""Main API for loading and packing CVC files.

This module provides the high-level interface for working with CVC compressed files.
"""

from .loader import CVCLoader
from .packer import pack_cvc as _pack_cvc, pack_cvc_sections as _pack_cvc_sections

# Singleton loader instance
_loader = CVCLoader()


def load_cvc(path, device="cpu", framework="torch", backend="auto"):
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
    
    Examples:
        >>> # CPU loading with auto backend selection
        >>> vectors = load_cvc("embeddings.cvc", device="cpu")
        
        >>> # GPU loading with CUDA native
        >>> vectors = load_cvc("embeddings.cvc", device="cuda", backend="cuda")
        
        >>> # GPU loading with Triton (vendor-agnostic)
        >>> vectors = load_cvc("embeddings.cvc", device="cuda", backend="triton")
    """
    return _loader.load(path, device=device, framework=framework, backend=backend)


def pack_cvc(vectors, output_path, compression="fp16", chunk_size=100000, chunk_metadata=None):
    """
    Pack numpy array of vectors into .cvc compressed format.
    
    Args:
        vectors: np.ndarray of shape (n_vectors, dimension), dtype float32
        output_path: Path to output .cvc file
        compression: "fp16" or "int8"
        chunk_size: Number of vectors per chunk
        chunk_metadata: Optional list of dicts with metadata per chunk.
                       Must have length equal to the number of chunks.
    
    Examples:
        >>> import numpy as np
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> pack_cvc(embeddings, "embeddings.cvc", compression="fp16")
        
        >>> # With custom chunk metadata
        >>> metadata = [{"source": "batch1"}, {"source": "batch2"}]
        >>> pack_cvc(embeddings, "embeddings.cvc", chunk_size=5000, chunk_metadata=metadata)
    """
    return _pack_cvc(vectors, output_path, compression=compression, chunk_size=chunk_size, chunk_metadata=chunk_metadata)


def pack_cvc_sections(sections, output_path, compression="fp16", chunk_size=100000):
    """
    Pack multiple arrays with section-level metadata into a single .cvc file.
    
    This allows you to combine data from different sources (with arbitrary sizes)
    into one file while maintaining section-level metadata for filtering.
    
    Args:
        sections: List of tuples (array, metadata_dict) where:
                 - array: np.ndarray of shape (n_vectors, dimension), dtype float32
                 - metadata_dict: dict with metadata for this section
        output_path: Path to output .cvc file
        compression: "fp16" or "int8"
        chunk_size: Number of vectors per chunk (applies uniformly)
    
    Examples:
        >>> import numpy as np
        >>> 
        >>> # Different sized arrays from different sources
        >>> wikipedia = np.random.randn(10_000, 768).astype(np.float32)
        >>> arxiv = np.random.randn(110_000, 768).astype(np.float32)
        >>> github = np.random.randn(50_000, 768).astype(np.float32)
        >>> 
        >>> sections = [
        >>>     (wikipedia, {"source": "wikipedia", "date": "2024-01"}),
        >>>     (arxiv, {"source": "arxiv", "date": "2024-02", "quality": "high"}),
        >>>     (github, {"source": "github", "date": "2024-03"}),
        >>> ]
        >>> 
        >>> pack_cvc_sections(sections, "combined.cvc", chunk_size=10_000)
        >>> 
        >>> # Later, load only arxiv vectors
        >>> from decompressed import load_cvc_range
        >>> arxiv_vectors = load_cvc_range("combined.cvc", 
        >>>                                 section_key="source", 
        >>>                                 section_value="arxiv")
    """
    return _pack_cvc_sections(sections, output_path, compression=compression, chunk_size=chunk_size)


def get_available_backends():
    """
    Get information about available backends.
    
    Returns:
        dict: Dictionary mapping backend names to availability status
    
    Examples:
        >>> backends = get_available_backends()
        >>> print(f"CUDA available: {backends['cuda']}")
    """
    return _loader.get_backend_availability()


def get_backend_errors():
    """
    Get error messages for backends that failed to load.
    
    Returns:
        dict: Dictionary mapping backend names to error messages (None if no error)
    
    Examples:
        >>> errors = get_backend_errors()
        >>> if errors['triton']:
        >>>     print(f"Triton error: {errors['triton']}")
    """
    return {
        'python': None,  # Always available
        'cpp': None if _loader.cpp_backend.is_available() else "C++ extensions not built",
        'cuda': None if _loader.cuda_backend.is_available() else "CUDA extensions not built",
        'triton': _loader.triton_backend.get_error() if hasattr(_loader.triton_backend, 'get_error') else None,
    }


def get_cvc_info(path):
    """
    Read CVC file metadata without loading vectors.
    
    Useful for inspecting file contents before loading, checking chunk structure,
    or implementing custom loading strategies.
    
    Args:
        path: Path to .cvc file
        
    Returns:
        dict: File metadata containing:
            - num_vectors: Total number of vectors
            - dimension: Vector dimensionality  
            - compression: Default compression scheme
            - chunks: List of chunk metadata (each with rows, compression, etc.)
            - num_chunks: Number of chunks
    
    Examples:
        >>> info = get_cvc_info("embeddings.cvc")
        >>> print(f"File contains {info['num_vectors']} vectors in {info['num_chunks']} chunks")
        >>> print(f"Dimension: {info['dimension']}, Compression: {info['compression']}")
    """
    return _loader.get_info(path)


def load_cvc_chunked(path, chunk_indices=None, device="cpu", framework="torch", backend="auto"):
    """
    Load and decompress specific chunks from a .cvc file as an iterator.
    
    This is useful for:
    - Processing large files that don't fit in memory
    - Streaming/iterative processing of embeddings
    - Loading only a subset of vectors from a large collection
    
    Args:
        path: Path to .cvc file
        chunk_indices: List of chunk indices to load (0-indexed), or None to load all chunks.
                      Use get_cvc_info() to determine how many chunks exist.
        device: "cpu" or "cuda"
        framework: "torch" or "cupy" (for GPU arrays)
        backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
        
    Yields:
        tuple: (chunk_index, chunk_array) for each chunk
            - chunk_index: 0-indexed chunk number
            - chunk_array: Decompressed vectors for that chunk
    
    Examples:
        >>> # Iterate through all chunks
        >>> for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", device="cpu"):
        >>>     print(f"Processing chunk {chunk_idx}: {vectors.shape}")
        >>>     # Process this chunk...
        
        >>> # Load only specific chunks (e.g., chunks 0, 2, and 5)
        >>> for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", 
        >>>                                             chunk_indices=[0, 2, 5],
        >>>                                             device="cuda"):
        >>>     print(f"Loaded chunk {chunk_idx}: {vectors.shape}")
    """
    return _loader.load_chunks(path, chunk_indices, device, framework, backend)


def load_cvc_range(path, chunk_indices=None, device="cpu", framework="torch", backend="auto", 
                   metadata_key=None, metadata_value=None, section_key=None, section_value=None):
    """
    Load specific chunks from a .cvc file and concatenate them into a single array.
    
    This is useful for loading a specific subset of vectors from a large file
    without loading the entire dataset.
    
    Args:
        path: Path to .cvc file
        chunk_indices: List of chunk indices to load (0-indexed).
                      Use get_cvc_info() to determine how many chunks exist.
                      Cannot be used together with filtering parameters.
        device: "cpu" or "cuda"
        framework: "torch" or "cupy" (for GPU arrays)
        backend: Backend to use - "auto", "python", "cpp", "cuda", or "triton"
        metadata_key: Optional metadata key to filter chunks by (for files with chunk_metadata)
        metadata_value: Value to match for metadata_key
        section_key: Optional section metadata key to filter by (for files packed with pack_cvc_sections)
        section_value: Value to match for section_key
        
    Returns:
        Array containing the requested chunks concatenated together
    
    Examples:
        >>> # Load first 3 chunks only
        >>> vectors = load_cvc_range("embeddings.cvc", chunk_indices=[0, 1, 2], device="cpu")
        
        >>> # Load specific non-contiguous chunks
        >>> vectors = load_cvc_range("embeddings.cvc", 
        >>>                          chunk_indices=[0, 5, 10],
        >>>                          device="cuda",
        >>>                          backend="triton")
        
        >>> # Load chunks by chunk metadata (for files with chunk_metadata)
        >>> vectors = load_cvc_range("embeddings.cvc", 
        >>>                          metadata_key="source", 
        >>>                          metadata_value="arxiv")
        
        >>> # Load by section metadata (for files packed with pack_cvc_sections)
        >>> vectors = load_cvc_range("combined.cvc",
        >>>                          section_key="source",
        >>>                          section_value="arxiv",
        >>>                          device="cuda")
    """
    return _loader.load_range(path, chunk_indices, device, framework, backend, 
                             metadata_key=metadata_key, metadata_value=metadata_value,
                             section_key=section_key, section_value=section_value)


# Legacy module-level constants for compatibility
HEADER_MAGIC = b"CVCF"

# Check what backends are available (for backward compatibility)
_availability = _loader.get_backend_availability()
HAS_NATIVE = _availability['cpp']
HAS_CUDA = _availability['cuda']
HAS_TRITON = _availability['triton']
