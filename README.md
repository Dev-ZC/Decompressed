# Decompressed
 
Decompressed is a GPU-native decompression library for vector embeddings and similarity search workloads.  
It provides a compact on-disk format (`.cvc`) and high-throughput decompression paths for CPU and GPU.

The focus is:

- Efficient storage for large embedding collections (FP16 and INT8).
- Fast, streaming decompression directly into the target device.
- Simple, minimal Python API suitable for production and research.

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

  - `.cvc` format is chunked.
  - Load datasets that do not fit into host RAM.
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
```

**Signature**

```python
pack_cvc(vectors, output_path, compression="fp16", chunk_size=100000)
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

**Returns**

- `None`. Writes the `.cvc` file to `output_path`.

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
