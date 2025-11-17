# Decompression Backends

This directory contains modular backend implementations for CVC decompression.

## Structure

- **`base.py`**: Abstract interface that all backends must implement
- **`cpu.py`**: CPU-based backends (Python, C++)
- **`gpu.py`**: GPU-based backends (CUDA native, Triton)

## Backend Implementations

### CPU Backends

- **PythonBackend**: Pure Python implementation (always available, slowest)
- **CPPBackend**: C++ native implementation (requires compilation, ~2x faster)

### GPU Backends

- **CUDABackend**: CUDA native kernels (NVIDIA only, fastest for large batches)
- **TritonBackend**: Triton-compiled kernels (vendor-agnostic: NVIDIA/AMD/Intel)

## Adding New Backends

To add a new backend:

1. Create a class that inherits from `BackendInterface`
2. Implement `decompress_chunk()` method
3. Implement `is_available()` method
4. Add to `__init__.py` exports
5. Update `loader.py` to register the backend

Example:

```python
from .base import BackendInterface

class MyBackend(BackendInterface):
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset):
        # Your implementation here
        pass
    
    def is_available(self):
        # Check if backend dependencies are available
        return True
```
