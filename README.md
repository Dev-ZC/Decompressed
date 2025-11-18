# Decompressed

[![PyPI version](https://img.shields.io/pypi/v/decompressed)](https://pypi.org/project/decompressed/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Decompressed is a GPU-native decompression library for vector embeddings and similarity search workloads.  
It provides a compact on-disk format (`.cvc`) and high-throughput decompression paths for CPU and GPU.

## Installation

```bash
# Basic installation (CPU only)
pip install decompressed

# With GPU support (Triton backend - vendor agnostic)
pip install decompressed[gpu]

# Development installation
git clone https://github.com/Dev-ZC/Decompressed.git
cd Decompressed
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from decompressed import pack_cvc_sections, load_cvc_range

# Pack embeddings from different sources
wikipedia = np.random.randn(10_000, 768).astype(np.float32)
arxiv = np.random.randn(110_000, 768).astype(np.float32)

pack_cvc_sections([
    (wikipedia, {"source": "wikipedia"}),
    (arxiv, {"source": "arxiv"}),
], "embeddings.cvc")

# Load only arXiv embeddings
arxiv_vectors = load_cvc_range("embeddings.cvc", 
                              section_key="source", 
                              section_value="arxiv")
print(f"Loaded {arxiv_vectors.shape[0]:,} vectors")
```

---

## Why Decompressed?

- **Efficient storage**: 2-4× compression with FP16/INT8
- **Fast GPU decompression**: Direct to GPU memory
- **Flexible metadata**: Pack multiple sources with section metadata
- **Selective loading**: Load only the data you need
- **Simple API**: One-line filtering and loading

---

## Key Features

- **GPU-native decompression**

  - Direct decompression into GPU memory.
  - Triton-based kernels for vendor-agnostic GPU support (NVIDIA, AMD, Intel).
  - CUDA kernels planned as the highest-performance path on NVIDIA.

- **Multiple compression schemes**

  - **FP16**: 2× compression vs FP32 with minimal accuracy loss.
  - **INT8**: 4× compression vs FP32 via linear quantization.

- **Chunked, streaming format**

  - `.cvc` format is chunked for efficient storage and streaming.
  - **Chunked decompression API**: Load and decompress specific chunks on-demand.
  - Load datasets that do not fit into host RAM via `load_cvc_chunked()`.
  - Per-chunk compression parameters.

- **Framework-agnostic integration**

  - Python API supports NumPy, PyTorch, and CuPy.
  - CPU decompression via Python or C++ backend.
  - GPU decompression via Triton backend, with CUDA backend under development.

---

## Installation

Decompressed can be used in three main configurations:

1. **CPU-only**
2. **GPU with Triton backend (recommended today)**
3. **GPU with CUDA native backend (under active development)**

### CPU-only

This installs the Python + C++ (if available) backends without GPU dependencies.

```bash
pip install decompressed
```

- **Device support**: `device="cpu"` only.
- **Backends**:
  - `backend="python"`: pure Python, always available.
  - `backend="cpp"`: C++ extension (if built), typically faster.

---

### GPU (Triton backend, vendor-agnostic)

This path targets any GPU supported by PyTorch + Triton (NVIDIA, AMD, Intel).

```bash
pip install decompressed[gpu]
```

Requirements (typical):

- `torch` with CUDA / ROCm / other GPU build.
- `triton` compatible with your PyTorch / CUDA stack.

**Device support:**

- `device="cuda"` with `backend="triton"` or `backend="auto"`.

Triton is the default GPU backend when CUDA native is not available.  
On a compatible GPU stack, you should expect high throughput and portability.

---

### GPU (CUDA native backend, NVIDIA) — under development

A CUDA native backend (`backend="cuda"`) is being developed as the highest-performance path for NVIDIA GPUs.  
At the moment:

- The CUDA backend is **experimental / under active development**.
- Depending on your build and environment, it may not be available or may fall back to Triton.

If you want to experiment with CUDA native once it is available:

```bash
# Build against your local CUDA toolkit
pip install --no-binary=decompressed decompressed[cuda]
```

- **Important**: use `--no-binary=decompressed` so that the extension is compiled against the CUDA toolkit present on your system.
- If the CUDA backend cannot be built or loaded, `load_cvc(..., backend="auto")` will fall back to Triton (if installed) or CPU.

---

### “All backends” install

For development and benchmarking, you can install everything:

```bash
# CPU + Triton GPU (+ CUDA when available)
pip install --no-binary=decompressed decompressed[all]
```

This attempts to provide:

- CPU backends (`python`, `cpp`).
- Triton GPU backend (`triton`).
- CUDA backend when buildable against your local CUDA.

---

### CUDA / PyTorch compatibility and PTX errors

On GPU, you may run into errors like:

> `PTX was compiled with an unsupported toolchain`

This typically indicates a **mismatch between the CUDA version used by PyTorch and the CUDA toolkit / driver on your system**.  
This can affect both Triton and CUDA backends.

Internally, Decompressed checks CUDA/PyTorch compatibility and may emit a warning at import or first use. To avoid PTX errors:

- Ensure that **PyTorch’s CUDA version matches your system CUDA**.
- For example:

```bash
# Example: system CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Example: system CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

If Triton encounters a PTX toolchain error at runtime:

- Decompressed will print a detailed help message.
- If a CUDA backend is available, Triton will attempt to **fall back** to CUDA for decompression.
- If no fallback is available, the original error is re-raised with additional diagnostic information.

---

## Python API

The primary user-facing API is exposed from `decompressed.pycvc` and re-exported at the package level.

### `load_cvc`

```python
from decompressed import load_cvc

vectors = load_cvc(
    path,
    device="cpu",
    framework="torch",
    backend="auto",
)
```

**Signature**

```python
load_cvc(path, device="cpu", framework="torch", backend="auto")
```

**Arguments**

- `path`: `str` or `pathlib.Path`  
  Path to a `.cvc` file on disk.

- `device`: `str`  
  - `"cpu"`: allocate and decompress into a NumPy array (or CPU tensor if desired).
  - `"cuda"`: allocate and decompress directly into GPU memory.

- `framework`: `str`  
  Used **only** when `device="cuda"`:
  - `"torch"`: returns a `torch.Tensor` on CUDA.
  - `"cupy"`: returns a `cupy.ndarray` on the current CUDA device.

- `backend`: `str`  
  Backend implementation to use:
  - `"auto"` (recommended): select the best available backend for the given `device`.
  - `"python"`: pure Python CPU implementation.
  - `"cpp"`: C++ CPU backend.
  - `"triton"`: Triton GPU backend (vendor-agnostic).
  - `"cuda"`: CUDA native GPU backend (NVIDIA, under development).

**Returns**

- For `device="cpu"`: `numpy.ndarray` of shape `(num_vectors, dim)`, `dtype=float32`.
- For `device="cuda"`, `framework="torch"`: `torch.Tensor` on CUDA.
- For `device="cuda"`, `framework="cupy"`: `cupy.ndarray` on CUDA.

**Examples**

```python
# CPU, automatic backend selection (prefers C++ if available)
vectors_cpu = load_cvc("embeddings.cvc", device="cpu")

# GPU with Triton backend (vendor-agnostic)
vectors_torch = load_cvc(
    "embeddings.cvc",
    device="cuda",
    framework="torch",
    backend="triton",
)

# GPU with automatic backend selection
# (prefers CUDA native when available, otherwise Triton)
vectors_gpu = load_cvc("embeddings.cvc", device="cuda", backend="auto")
```

---

### `pack_cvc`

```python
from decompressed import pack_cvc
import numpy as np

embeddings = np.random.randn(1_000_000, 768).astype(np.float32)
pack_cvc(
    embeddings,
    output_path="embeddings.cvc",
    compression="fp16",
    chunk_size=100_000,
)

# With custom chunk metadata (10 chunks of 100k vectors each)
metadata = [
    {"source": "batch_0", "date": "2024-01"},
    {"source": "batch_1", "date": "2024-02"},
    # ... 8 more chunks
]
pack_cvc(
    embeddings,
    output_path="embeddings_with_metadata.cvc",
    compression="fp16",
    chunk_size=100_000,
    chunk_metadata=metadata,
)
```

**Signature**

```python
pack_cvc(vectors, output_path, compression="fp16", chunk_size=100000, chunk_metadata=None)
```

**Arguments**

- `vectors`: `numpy.ndarray`  
  Shape `(num_vectors, dimension)`, `dtype=float32`.

- `output_path`: `str` or `pathlib.Path`  
  Path at which to write the `.cvc` file.

- `compression`: `str`  
  Compression scheme:
  - `"fp16"`: half-precision floats.
  - `"int8"`: 8‑bit linear quantization with per-chunk `min` and `scale`.

- `chunk_size`: `int`  
  Number of vectors per chunk.

- `chunk_metadata`: `list[dict]` or `None` (optional)  
  Optional list of metadata dictionaries, one per chunk. Must have length equal to the number of chunks (`ceil(num_vectors / chunk_size)`). Metadata can be retrieved later using `get_cvc_info()`.

**Returns**

- `None`. Writes the `.cvc` file to `output_path`.

---

### `pack_cvc_sections`

**Combine multiple arrays of different sizes with section-level metadata**

```python
from decompressed import pack_cvc_sections
import numpy as np

# Create arrays from different sources (arbitrary sizes!)
wikipedia = np.random.randn(10_000, 768).astype(np.float32)
arxiv = np.random.randn(110_000, 768).astype(np.float32)
github = np.random.randn(25_000, 768).astype(np.float32)

# Pack them into one file with section metadata
sections = [
    (wikipedia, {"source": "wikipedia", "quality": "high"}),
    (arxiv, {"source": "arxiv", "quality": "high", "date": "2024-02"}),
    (github, {"source": "github", "quality": "medium"}),
]

pack_cvc_sections(
    sections,
    output_path="combined.cvc",
    compression="fp16",
    chunk_size=10_000
)

# Later, load only the data you need
from decompressed import load_cvc_range

arxiv_only = load_cvc_range("combined.cvc", 
                            section_key="source", 
                            section_value="arxiv")
# Returns 110k vectors (only arXiv section)
```

**Signature**

```python
pack_cvc_sections(sections, output_path, compression="fp16", chunk_size=100000)
```

**Arguments**

- `sections`: `list[tuple[ndarray, dict]]`  
  List of `(array, metadata_dict)` tuples where:
  - `array`: `numpy.ndarray` of shape `(n_vectors, dimension)`, dtype `float32`
  - `metadata_dict`: Dictionary with metadata for this section (any keys/values)

- `output_path`: `str` or `pathlib.Path`  
  Path at which to write the `.cvc` file.

- `compression`: `str`  
  Compression scheme (`"fp16"` or `"int8"`).

- `chunk_size`: `int`  
  Number of vectors per chunk (applied uniformly to all sections).

**Returns**

- `None`. Writes the `.cvc` file to `output_path`.

**When to use `pack_cvc_sections` vs `pack_cvc`:**

- **Use `pack_cvc_sections`** when:
  - You have multiple data sources with different sizes
  - Sizes don't align with chunk boundaries (e.g., 10k + 110k + 25k)
  - You want to filter by data source or other section properties
  - You need flexible metadata per data source

- **Use `pack_cvc`** when:
  - You have a single array
  - You need chunk-level metadata for batch processing

---

### `get_cvc_info`

```python
from decompressed import get_cvc_info

info = get_cvc_info("embeddings.cvc")
print(f"File contains {info['num_vectors']} vectors in {info['num_chunks']} chunks")
```

**Signature**

```python
get_cvc_info(path)
```

**Arguments**

- `path`: `str` or `pathlib.Path`  
  Path to a `.cvc` file on disk.

**Returns**

- `dict`: File metadata containing:
  - `num_vectors`: Total number of vectors in the file.
  - `dimension`: Vector dimensionality.
  - `compression`: Default compression scheme.
  - `chunks`: List of chunk information. Each chunk dict contains:
    - `index`: Chunk index (0-based).
    - `rows`: Number of vectors in this chunk.
    - `metadata`: Custom metadata for this chunk (if provided during packing), or `None`.
  - `num_chunks`: Number of chunks.

**Use Cases**

- Inspect file contents before loading.
- Determine chunk structure for implementing custom loading strategies.
- Get file statistics without loading vectors into memory.

---

### `load_cvc_chunked`

```python
from decompressed import load_cvc_chunked

# Iterate through all chunks
for chunk_idx, vectors in load_cvc_chunked("embeddings.cvc", device="cpu"):
    print(f"Processing chunk {chunk_idx}: {vectors.shape}")
    # Process vectors chunk by chunk...

# Load only specific chunks
for chunk_idx, vectors in load_cvc_chunked(
    "embeddings.cvc",
    chunk_indices=[0, 2, 5],
    device="cuda",
):
    print(f"Loaded chunk {chunk_idx}")
```

**Signature**

```python
load_cvc_chunked(path, chunk_indices=None, device="cpu", framework="torch", backend="auto")
```

**Arguments**

- `path`: `str` or `pathlib.Path`  
  Path to a `.cvc` file on disk.

- `chunk_indices`: `list[int]` or `None`  
  List of chunk indices to load (0-indexed), or `None` to load all chunks.  
  Use `get_cvc_info()` to determine how many chunks exist.

- `device`: `str`  
  - `"cpu"`: decompress to CPU memory.
  - `"cuda"`: decompress to GPU memory.

- `framework`: `str`  
  Used when `device="cuda"`:
  - `"torch"`: returns PyTorch tensors.
  - `"cupy"`: returns CuPy arrays.

- `backend`: `str`  
  Backend implementation to use (same options as `load_cvc`).

**Yields**

- `tuple[int, array]`: For each chunk:
  - `chunk_index`: 0-indexed chunk number.
  - `chunk_array`: Decompressed vectors for that chunk.

**Use Cases**

- **Streaming processing**: Process large files that don't fit in memory.
- **Memory-efficient workflows**: Load and process one chunk at a time.
- **Selective loading**: Load only the chunks you need.
- **Incremental computation**: Compute embeddings or similarity scores incrementally.

---

### `load_cvc_range`

```python
from decompressed import load_cvc_range

# Load first 3 chunks only
vectors = load_cvc_range("embeddings.cvc", chunk_indices=[0, 1, 2], device="cpu")

# Load specific non-contiguous chunks
vectors = load_cvc_range(
    "embeddings.cvc",
    chunk_indices=[0, 5, 10],
    device="cuda",
    backend="triton",
)
```

**Signature**

```python
load_cvc_range(path, chunk_indices=None, device="cpu", framework="torch", backend="auto",
               metadata_key=None, metadata_value=None)
```

**Arguments**

- `path`: `str` or `pathlib.Path`  
  Path to a `.cvc` file on disk.

- `chunk_indices`: `list[int]` or `None`  
  List of chunk indices to load (0-indexed).  
  Use `get_cvc_info()` to determine how many chunks exist.  
  Cannot be used together with `metadata_key`/`metadata_value`.

- `device`: `str`  
  - `"cpu"`: decompress to CPU memory.
  - `"cuda"`: decompress to GPU memory.

- `framework`: `str`  
  Used when `device="cuda"`:
  - `"torch"`: returns PyTorch tensors.
  - `"cupy"`: returns CuPy arrays.

- `backend`: `str`  
  Backend implementation to use (same options as `load_cvc`).

- `metadata_key`: `str` or `None`  
  Optional metadata key to filter chunks by.

- `metadata_value`: `any` or `None`  
  Value to match for `metadata_key`. Only chunks with matching metadata will be loaded.

**Returns**

- Array containing the requested chunks concatenated together.

**Use Cases**

- **Partial loading**: Load only a subset of vectors from a large collection.
- **Range queries**: Load vectors in a specific index range without loading the full file.
- **Sharded processing**: Process different chunks on different GPUs or machines.

---

## Advanced Pattern: Metadata-Based Filtering

### Section-Based Filtering (Recommended)

When you have multiple data sources with different sizes, use `pack_cvc_sections` and filter by section metadata:

```python
from decompressed import pack_cvc_sections, load_cvc_range
import numpy as np

# Pack multiple sources together
sections = [
    (wikipedia_vectors, {"source": "wikipedia", "quality": "high"}),
    (arxiv_vectors, {"source": "arxiv", "quality": "high"}),
    (github_vectors, {"source": "github", "quality": "medium"}),
]

pack_cvc_sections(sections, "combined.cvc", chunk_size=10_000)

# Load only arXiv section (regardless of size or chunk alignment)
arxiv_only = load_cvc_range("combined.cvc",
                            section_key="source",
                            section_value="arxiv")

# Load all high-quality sections
high_quality = load_cvc_range("combined.cvc",
                              section_key="quality",
                              section_value="high")
```

**Benefits:**
- ✅ Works with any section sizes (no alignment needed)
- ✅ Automatically handles chunk boundaries
- ✅ Filters and extracts only the requested section data
- ✅ Single simple API call

### Chunk-Level Filtering

For files packed with `chunk_metadata`, filter by chunk properties:

```python
# Load chunks by metadata (for files with chunk_metadata)
vectors = load_cvc_range("embeddings.cvc", 
                        metadata_key="batch_id", 
                        metadata_value="batch_42")
```

### Manual Complex Filtering

For complex multi-criteria queries:

```python
from decompressed import get_cvc_info, load_cvc_range

info = get_cvc_info("embeddings.cvc")

# Filter chunks by multiple criteria
filtered_chunks = [
    chunk['index'] 
    for chunk in info['chunks'] 
    if chunk.get('metadata') 
    and chunk['metadata'].get('quality') == 'high'
    and chunk['metadata'].get('date') >= '2024-03'
]

vectors = load_cvc_range("embeddings.cvc", chunk_indices=filtered_chunks)
```

---

### `get_available_backends`

```python
from decompressed import get_available_backends

backends = get_available_backends()
print(backends)
# Example: {'python': True, 'cpp': True, 'cuda': False, 'triton': True}
```

**Signature**

```python
get_available_backends()
```

**Returns**

- `dict[str, bool]` mapping:

  - `"python"`: pure Python CPU backend.
  - `"cpp"`: C++ CPU backend.
  - `"cuda"`: CUDA native GPU backend (True only if built and importable).
  - `"triton"`: Triton GPU backend (True if Triton and its kernels are importable).

---

### `get_backend_errors`

```python
from decompressed import get_backend_errors

errors = get_backend_errors()
if errors["triton"]:
    print("Triton backend issue:", errors["triton"])
```

**Signature**

```python
get_backend_errors()
```

**Returns**

- `dict[str, Optional[str]]` mapping backend names to an error string (or `None`):

  - `"python"`: always `None`.
  - `"cpp"`: `None` if C++ extensions are built, otherwise a message.
  - `"cuda"`: `None` if CUDA extensions are built and importable, otherwise a message.
  - `"triton"`: error message from Triton initialization if it failed, otherwise `None`.

---

## Backend selection and device behavior

The loader uses a `CVCLoader` with a simple selection mechanism:

- For `device="cpu"`:
  - If `backend="auto"`: prefer `"cpp"` if available, otherwise `"python"`.
  - If `backend="python"` or `"cpp"` is explicitly requested, the loader validates that a CPU device is used.
- For `device="cuda"`:
  - If `backend="auto"`:
    - Prefer `"cuda"` if the CUDA backend is available.
    - Otherwise, fall back to `"triton"` if available.
    - If neither CUDA nor Triton are available, a runtime error is raised.
  - If `backend="cuda"` or `"triton"` is explicitly requested, the loader validates that `device="cuda"`.

This logic is implemented in `select_backend` and `validate_backend_availability` and is used within `CVCLoader.load`.

---

## How it works on each device

### CPU

- Input `.cvc` file is parsed on CPU.
- Decompression happens either in pure Python or via a C++ extension.
- Output: `numpy.ndarray` on host memory.

### GPU with Triton

- Header is parsed on CPU.
- Chunk payloads are transferred as needed.
- Triton kernels run on the GPU to perform:
  - FP16 → FP32 conversion (for `compression="fp16"`).
  - INT8 dequantization (for `compression="int8"`) using stored `min`/`scale`.
- Output: `torch.Tensor` or `cupy.ndarray` on the GPU.

### GPU with CUDA native (under development)

- Design goal: provide a custom CUDA kernel path optimized for NVIDIA GPUs.
- Intended behavior:
  - Use CUDA kernels for FP16 and INT8 decompression.
  - Match or exceed Triton throughput on NVIDIA hardware.
- Current state:
  - The backend is under active development and may not be available in all builds.
  - When not available, `get_available_backends()['cuda']` is `False`, and `backend="auto"` will fall back to Triton.

---

## CVC file format

The `.cvc` format is documented in detail in [`format.md`](format.md).  
In brief:

- A fixed magic header (`"CVCF"`).
- A JSON metadata header describing:
  - `num_vectors`, `dimension`, default `compression`.
  - Per-chunk metadata including `rows`, optional `compression`, and quantization parameters.
- A sequence of chunk length + compressed payload pairs.

For implementation details, see:

- [`python/decompressed/packer.py`](python/decompressed/packer.py) for packing.
- [`python/decompressed/loader.py`](python/decompressed/loader.py) for loading/decompression.

---

## Building from source

Refer to:

- [`BUILD.md`](BUILD.md) for manual build instructions.
- [`INSTALL_CUDA.md`](INSTALL_CUDA.md) for CUDA-specific build and troubleshooting notes.

A typical CMake-based C++ build looks like:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

---

## Additional documentation

- [`format.md`](format.md): CVC file format specification.
- [`python/decompressed/ARCHITECTURE.md`](python/decompressed/ARCHITECTURE.md): high-level architecture and backend selection.
- `benchmarks/`: benchmark scripts and example throughput numbers.

---

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Triton ≥ 2.0.0 (for GPU)
- CUDA Toolkit ≥ 11.0 (optional, for CUDA kernels)

---

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) for details.

---

## Contributing

Contributions are welcome. Please open an issue or pull request.

---

## Citation

If you use Decompressed in your research, please consider citing:

```bibtex
@software{decompressed2025,
  title  = {Decompressed: GPU-Native Decompression for Vector Embeddings},
  author = {Zac Icole and contributors},
  year   = {2025},
  url    = {https://github.com/Dev-ZC/Decompressed}
}
```
