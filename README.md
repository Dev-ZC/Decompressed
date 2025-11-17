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

Choose the installation that matches your hardware:

**🖥️ CPU Only (Default):**
```bash
pip install decompressed
# C++ native CPU decompression (~1 GB/s)
# No GPU dependencies, minimal install
```

**🎮 GPU - Triton (Vendor Agnostic):**
```bash
pip install decompressed[gpu]
# Works on: NVIDIA, AMD, Intel GPUs
# Uses Triton kernels (requires PyTorch + Triton)
# ~3-10 GB/s depending on GPU
```

**🚀 GPU - CUDA Native (NVIDIA Only, Fastest):**
```bash
# Requires: CUDA Toolkit installed on system
# IMPORTANT: Use --no-binary to build against YOUR CUDA version
pip install --no-binary=decompressed decompressed[cuda]

# CUDA native kernels (~1-5 GB/s, faster with larger batches)
# Best performance on NVIDIA GPUs
```

### ⚠️ CUDA Installation - Important!

**Always build CUDA from source** to match your CUDA toolkit:

```bash
# ✅ Correct way to install CUDA support
pip install --no-binary=decompressed decompressed[cuda]

# ❌ Don't use this (may install pre-built binary with wrong CUDA version)
pip install decompressed[cuda]
```

The `--no-binary=decompressed` flag ensures the package builds against **YOUR installed CUDA toolkit**, avoiding "PTX unsupported toolchain" errors.

**Why?** CUDA binaries must match your exact CUDA version (11.8, 12.1, 12.4, etc.). Pre-built wheels may not match your system.

**Alternative:** Use Triton instead (no build needed, works on all GPUs):
```bash
pip install decompressed[gpu]  # Uses Triton, ~80-90% of CUDA performance
```

See [INSTALL_CUDA.md](INSTALL_CUDA.md) for troubleshooting.

**🌟 Everything (Both GPU Backends):**
```bash
# Install both Triton + CUDA native
pip install --no-binary=decompressed decompressed[all]

# Automatic backend selection picks fastest available
```

### Installation Summary

| Command | Dependencies | Backends Available | Best For |
|---------|-------------|-------------------|----------|
| `decompressed` | numpy | Python, C++ (CPU) | CPU-only servers, minimal deps |
| `decompressed[gpu]` | numpy, torch, triton | Python, C++, Triton (GPU) | AMD/Intel GPUs, any PyTorch user |
| `decompressed[cuda]` | numpy, torch | Python, C++, CUDA (GPU)* | NVIDIA GPUs, maximum performance |
| `decompressed[all]` | numpy, torch, triton | All backends* | Development, benchmarking |

*CUDA native backend auto-builds if CUDA Toolkit is installed on system.

### Fixing Triton "PTX Toolchain" Errors

If you get `PTX was compiled with an unsupported toolchain`, your PyTorch CUDA version doesn't match your system CUDA:

```bash
# Check your CUDA version
nvcc --version  # e.g., CUDA 12.1

# Install matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121  # for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu118  # for CUDA 11.8
```

### Quick Install (GitHub)

```bash
# CPU-only
pip install 'decompressed @ git+https://github.com/Dev-ZC/Decompressed.git'

# With Triton GPU (no build needed, works on all GPUs)
pip install 'decompressed[gpu] @ git+https://github.com/Dev-ZC/Decompressed.git'

# With CUDA native (builds against YOUR CUDA version)
pip install --no-binary=decompressed \
  'decompressed[cuda] @ git+https://github.com/Dev-ZC/Decompressed.git'

# Everything (Triton + CUDA)
pip install --no-binary=decompressed \
  'decompressed[all] @ git+https://github.com/Dev-ZC/Decompressed.git'
```

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
