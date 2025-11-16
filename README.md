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

**🎯 Smart Install (Detects your system automatically):**
```bash
curl -sSL https://raw.githubusercontent.com/Dev-ZC/Decompressed/main/install.py | python3
# Detects GPU (NVIDIA/AMD/Intel) and installs optimal version
```

**Quick Install (Recommended):**
```bash
# From PyPI (when published)
pip install decompressed[gpu]

# From GitHub (for now)
pip install 'decompressed[gpu] @ git+https://github.com/Dev-ZC/Decompressed.git'

# Installs everything: numpy + torch + triton
# Works on: NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Intel GPUs, and CPU
```

**Manual Detection:**
```bash
# Check what you have:
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# If GPU available → install with GPU support:
pip install decompressed[gpu]

# If no GPU or just need CPU:
pip install decompressed
```

**Advanced Options:**
```bash
# Minimal (CPU-only, auto-builds CUDA if toolkit found)
pip install 'decompressed @ git+https://github.com/Dev-ZC/Decompressed.git'

# With PyTorch + Triton (GPU-agnostic)
pip install 'decompressed[gpu] @ git+https://github.com/Dev-ZC/Decompressed.git'

# With CuPy + Triton (NVIDIA-only alternative)
pip install 'decompressed[gpu-cupy] @ git+https://github.com/Dev-ZC/Decompressed.git'

# Development (local clone)
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

| Install Command | Environment | What's Built | Performance |
|----------------|-------------|--------------|-------------|
| `[gpu]` | **Colab (T4/A100)** | C++ + CUDA + Triton | 🚀 20+ GB/s |
| `[gpu]` | **Linux + NVIDIA** | C++ + CUDA + Triton | 🚀 20+ GB/s |
| `[gpu]` | **Linux + AMD** | C++ + Triton (ROCm) | ⚡ 3-5 GB/s |
| `[gpu]` | **Windows + NVIDIA** | C++ + CUDA + Triton | 🚀 20+ GB/s |
| base | **Linux + NVIDIA** | C++ + CUDA (if nvcc) | 🚀 20+ GB/s |
| base | **macOS (Apple)** | C++ CPU only | ⚡ 1-2 GB/s |
| base | **No compilers** | Pure Python | 🐌 0.5 GB/s |

**✨ Smart Detection:**
- `pip install decompressed` → Builds CUDA if toolkit found, CPU otherwise
- `pip install decompressed[gpu]` → Adds Triton for AMD/Intel GPU support
- Both work everywhere with graceful fallbacks

See [INSTALL.md](INSTALL.md) for details.

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
