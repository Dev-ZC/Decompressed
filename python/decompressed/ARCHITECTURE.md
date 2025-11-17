# Decompressed Package Architecture

Clean, modular architecture following best practices from projects like HuggingFace Transformers and scikit-learn.

## Package Structure

```
decompressed/
├── __init__.py              # Public API exports
├── pycvc.py                 # Main user-facing API (thin wrapper)
├── loader.py                # CVC file loader with backend management
├── packer.py                # CVC file packer/writer
├── compress.py              # Compression implementations
├── decompress.py            # Pure Python decompression (fallback)
├── utils.py                 # Utility functions (error handling, validation)
└── backends/                # Modular backend implementations
    ├── __init__.py          # Backend exports
    ├── README.md            # Backend documentation
    ├── base.py              # Abstract backend interface
    ├── cpu.py               # CPU backends (Python, C++)
    └── gpu.py               # GPU backends (CUDA, Triton)
```

## Module Responsibilities

### `pycvc.py` (80 lines)
- **Purpose**: High-level user-facing API
- **Exports**: `load_cvc()`, `pack_cvc()`, `get_available_backends()`
- **Design**: Thin wrapper around `loader` and `packer` modules

### `loader.py` (~150 lines)
- **Purpose**: Manages CVC file loading and backend orchestration
- **Key Class**: `CVCLoader`
- **Responsibilities**:
  - Header parsing
  - Output array allocation
  - Backend selection and validation
  - Chunk iteration and decompression dispatch

### `packer.py` (~70 lines)
- **Purpose**: Handles CVC file writing
- **Key Function**: `pack_cvc()`
- **Responsibilities**:
  - Chunking vectors
  - Compression
  - File format writing

### `utils.py` (~100 lines)
- **Purpose**: Shared utilities
- **Functions**:
  - Error message generation
  - Backend validation
  - Backend selection logic

### `backends/` (modular backends)

#### `base.py`
- Abstract interface: `BackendInterface`
- All backends implement: `decompress_chunk()`, `is_available()`

#### `cpu.py`
- `PythonBackend`: Pure Python (always available)
- `CPPBackend`: C++ native extensions

#### `gpu.py`
- `CUDABackend`: CUDA native kernels (NVIDIA)
- `TritonBackend`: Triton kernels (vendor-agnostic)
  - Includes PTX error handling with auto-fallback

## Design Principles

### 1. **Separation of Concerns**
- Each module has a single, clear responsibility
- Business logic separated from I/O, backend management, and user API

### 2. **Extensibility**
- New backends can be added without modifying existing code
- Abstract interface ensures consistency

### 3. **Error Handling**
- Centralized error messages in `utils.py`
- Graceful fallbacks with helpful user guidance

### 4. **Backward Compatibility**
- Old API still works: `from decompressed import load_cvc, pack_cvc`
- Legacy constants maintained: `HAS_NATIVE`, `HAS_CUDA`, `HAS_TRITON`

### 5. **Professional Organization**
- Similar structure to popular ML libraries
- Clear documentation in each module
- Type hints and docstrings throughout

## Usage Examples

### Basic Loading
```python
from decompressed import load_cvc
vectors = load_cvc("embeddings.cvc")  # Auto-selects best backend
```

### Backend Introspection
```python
from decompressed import get_available_backends
backends = get_available_backends()
print(f"Available: {backends}")
# {'python': True, 'cpp': True, 'cuda': True, 'triton': True}
```

### Advanced Backend Selection
```python
# Explicitly choose backend
vectors = load_cvc("embeddings.cvc", device="cuda", backend="triton")
```

## Testing

Each module can be tested independently:
- Backends can be mocked/tested in isolation
- Loader logic tested without actual backends
- Error handling tested without triggering real errors

## Future Extensions

Easy to add:
- **New backends**: Implement `BackendInterface`
- **New compression schemes**: Add to `compress.py` and backends
- **Streaming support**: Extend `CVCLoader` with iterator interface
- **Async loading**: Add async methods to loader
