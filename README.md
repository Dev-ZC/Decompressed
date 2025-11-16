# Decompressed

An open-source GPU-native decompression framework for vector embeddings, optimized for machine learning workloads.

## Features

- **🚀 GPU-Native Decompression**: Direct-to-GPU decompression with CUDA and Triton kernels
- **🎮 Multi-GPU Support**: Works with NVIDIA (CUDA), AMD (ROCm via Triton), and CPU
- **📦 Multiple Compression Schemes**: FP16 (2× compression) and INT8 (4× compression)
- **🔄 Framework Agnostic**: Works with PyTorch, CuPy, NumPy
- **⚡ High Performance**: Optimized kernels for modern GPUs (Tensor Cores, fast math)
- **💾 Streaming Support**: Chunked format for efficient memory usage
- **🛠️ Easy to Use**: Simple Python API with minimal dependencies

## Quick Start

### Installation

**Basic (Works everywhere, auto-detects CUDA if available):**
```bash
pip install decompressed
# or: pip install git+https://github.com/Dev-ZC/Decompressed.git

# Automatically builds:
# - C++ CPU extensions (if CMake available)
# - CUDA GPU extensions (if CUDA Toolkit detected)
# - Gracefully falls back to pure Python if neither available
```

**With GPU Support for AMD/Intel (Triton):**
```bash
pip install decompressed[gpu]
# Installs: numpy + torch + triton for GPU-agnostic support
```

**Alternative GPU (NVIDIA-only with CuPy):**
```bash
pip install decompressed[gpu-cupy]
# Installs: numpy + cupy + triton
```

**Development:**
```bash
git clone https://github.com/Dev-ZC/Decompressed.git
cd Decompressed
pip install -e ".[dev,gpu]"
```

### Installation Options

| Command | NumPy | PyTorch | Triton | Use Case |
|---------|-------|---------|--------|----------|
| `pip install decompressed` | Yes | No | No | CPU-only (works everywhere) |
| `pip install decompressed[gpu]` | Yes | Yes | Yes | GPU acceleration (any vendor) |
| `pip install decompressed[gpu-cupy]` | Yes | No | Yes | GPU with CuPy (NVIDIA only) |
| `pip install decompressed[all]` | Yes | Yes | Yes | All features |

### What Gets Built Automatically

| Environment | What Happens | Performance |
|-------------|--------------|-------------|
| **Colab (T4/A100)** | ✅ C++ CPU + ✅ CUDA native | 🚀 20+ GB/s |
| **Colab + `[gpu]`** | ✅ C++ CPU + ✅ CUDA native + ✅ Triton | 🚀 20+ GB/s |
| **Linux + NVIDIA GPU** | ✅ C++ CPU + ✅ CUDA native (if nvcc found) | 🚀 20+ GB/s |
| **Linux + AMD GPU + `[gpu]`** | ✅ C++ CPU + ✅ Triton (via ROCm) | ⚡ 3-5 GB/s |
| **macOS (Apple Silicon)** | ✅ C++ CPU only | ⚡ 1-2 GB/s |
| **Windows + NVIDIA** | ✅ C++ CPU + ✅ CUDA native (if CUDA SDK found) | 🚀 20+ GB/s |
| **No compilers** | ✅ Pure Python fallback | 🐌 0.5 GB/s |

**100% Turnkey** - One `pip install` command works everywhere! See [INSTALL.md](INSTALL.md) for details.

### Usage

```python
from decompressed import load_cvc, pack_cvc
import numpy as np

# Create embeddings
embeddings = np.random.randn(1000000, 768).astype(np.float32)

# Compress and save
pack_cvc(embeddings, "embeddings.cvc", compression="fp16")  # 2x compression
# pack_cvc(embeddings, "embeddings.cvc", compression="int8")  # 4x compression

# Load with auto backend selection (recommended)
vectors = load_cvc("embeddings.cvc", device="cpu")  # Uses best CPU backend
vectors_gpu = load_cvc("embeddings.cvc", device="cuda", framework="torch")  # Uses best GPU backend

# Or explicitly choose backend
vectors = load_cvc("embeddings.cvc", device="cpu", backend="python")   # Pure Python
vectors = load_cvc("embeddings.cvc", device="cpu", backend="cpp")      # C++ native (fast)
vectors = load_cvc("embeddings.cvc", device="cuda", backend="triton")  # Triton (GPU-agnostic)
# vectors = load_cvc("embeddings.cvc", device="cuda", backend="cuda")  # CUDA native (coming soon)

**Backend Options:**
- `backend="auto"` (default) - Automatically selects best available
- `backend="python"` - Pure Python (CPU, slowest, always works)
- `backend="cpp"` - C++ native (CPU, fast, requires build)
- `backend="triton"` - Triton kernels (GPU, fast, vendor-agnostic)
- `backend="cuda"` - CUDA native (GPU, fastest, coming soon)
```

## Compression Schemes

| Scheme | Compression Ratio | Accuracy | Speed |
|--------|------------------|----------|-------|
| FP16   | 2:1              | ~3 decimal digits | ~500 GB/s |
| INT8   | 4:1              | Calibrated | ~800 GB/s |

## Documentation

- **[INSTALL.md](INSTALL.md)**: Complete installation guide
- **[GPU_SUPPORT.md](GPU_SUPPORT.md)**: GPU support (NVIDIA, AMD, Intel)
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: How the turnkey installation works
- **[format.md](format.md)**: Detailed CVC file format specification
- **[BUILD.md](BUILD.md)**: Manual build instructions (advanced)
- **[benchmarks/](benchmarks/)**: Performance benchmarks

## Building from Source

See [BUILD.md](BUILD.md) for detailed build instructions.

Quick build:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Triton ≥ 2.0.0 (for GPU)
- CUDA Toolkit ≥ 11.0 (optional, for CUDA kernels)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or pull request.

## Citation

If you use Decompressed in your research, please cite:

```bibtex
@software{decompressed2025,
  title = {Decompressed: GPU-Native Decompression for Vector Embeddings},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/Dev-ZC/Decompressed}
}
```
