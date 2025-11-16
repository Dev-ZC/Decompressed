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

**Basic (CPU-only, works everywhere):**
```bash
pip install decompressed
# or: pip install git+https://github.com/Dev-ZC/Decompressed.git
```

**With GPU Support (NVIDIA, AMD, Intel):**
```bash
pip install decompressed[gpu]
# Installs: numpy + torch + triton
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

**That's it!** The package auto-detects your system and builds optimized extensions automatically. See [INSTALL.md](INSTALL.md) for details.

### Usage

```python
from decompressed import load_cvc, pack_cvc
import numpy as np

# Create embeddings
embeddings = np.random.randn(1000000, 768).astype(np.float32)

# Compress and save
pack_cvc(embeddings, "embeddings.cvc", compression="fp16")

# Load to GPU
vectors = load_cvc("embeddings.cvc", device="cuda", framework="torch")
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
