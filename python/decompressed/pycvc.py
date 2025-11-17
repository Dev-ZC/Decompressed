"""Main API for loading and packing CVC files.

This module provides the high-level interface for working with CVC compressed files.
"""

from .loader import CVCLoader
from .packer import pack_cvc as _pack_cvc

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


def pack_cvc(vectors, output_path, compression="fp16", chunk_size=100000):
    """
    Pack numpy array of vectors into .cvc compressed format.
    
    Args:
        vectors: np.ndarray of shape (n_vectors, dimension), dtype float32
        output_path: Path to output .cvc file
        compression: "fp16" or "int8"
        chunk_size: Number of vectors per chunk
    
    Examples:
        >>> import numpy as np
        >>> embeddings = np.random.randn(10000, 768).astype(np.float32)
        >>> pack_cvc(embeddings, "embeddings.cvc", compression="fp16")
    """
    return _pack_cvc(vectors, output_path, compression=compression, chunk_size=chunk_size)


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


# Legacy module-level constants for compatibility
HEADER_MAGIC = b"CVCF"

# Check what backends are available (for backward compatibility)
_availability = _loader.get_backend_availability()
HAS_NATIVE = _availability['cpp']
HAS_CUDA = _availability['cuda']
HAS_TRITON = _availability['triton']
