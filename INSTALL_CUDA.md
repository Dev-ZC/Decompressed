# Installing Decompressed with CUDA Native Support

## The Problem

CUDA native extensions must be compiled against your **exact CUDA toolkit version**. Pre-built wheels may not match your system.

## Solution: Build from Source

Always build from source when installing CUDA support to ensure compatibility with your CUDA toolkit:

### Quick Install (Recommended)

```bash
# For any CUDA version - automatically uses your system CUDA
pip install --no-binary=decompressed 'decompressed[cuda] @ git+https://github.com/Dev-ZC/Decompressed.git'
```

The `--no-binary=decompressed` flag forces pip to build from source using **your installed CUDA toolkit**.

### Alternative: PyPI (when published)

```bash
pip install --no-binary=decompressed decompressed[cuda]
```

### Manual Build

```bash
git clone https://github.com/Dev-ZC/Decompressed.git
cd Decompressed
pip install -e .[cuda]
```

## Verification

```python
from decompressed import get_available_backends, get_backend_errors

backends = get_available_backends()
errors = get_backend_errors()

print(f"CUDA Native: {backends['cuda']}")
if errors['cuda']:
    print(f"Error: {errors['cuda']}")
```

## Troubleshooting

### Error: "PTX was compiled with an unsupported toolchain"

**Cause**: You installed a pre-built wheel that was compiled with a different CUDA version.

**Fix**: Reinstall from source:
```bash
pip uninstall decompressed
pip install --no-binary=decompressed 'decompressed[cuda] @ git+https://github.com/Dev-ZC/Decompressed.git'
```

### Error: "CUDA not found" during build

**Cause**: CUDA toolkit not installed or not in PATH.

**Fix**:
1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Ensure `nvcc` is in PATH: `which nvcc` (should show a path)
3. Set CUDA_HOME if needed: `export CUDA_HOME=/usr/local/cuda`

### Multiple CUDA Versions Installed

If you have multiple CUDA versions, specify which one to use:

```bash
# Set CUDA path before installing
export CUDA_HOME=/usr/local/cuda-12.1
pip install --no-binary=decompressed decompressed[cuda]
```

## Why Not Pre-Built Wheels?

Unlike CPU-only code, CUDA binaries are tightly coupled to:
- CUDA toolkit version (11.8, 12.1, 12.4, etc.)
- GPU architecture (sm_70, sm_80, sm_86, etc.)
- Driver version

Building from source ensures the binary matches YOUR system exactly.

## Alternative: Use Triton Instead

If you don't need maximum performance or want easier installation:

```bash
pip install decompressed[gpu]  # Installs Triton (JIT compiled, no build needed)
```

Triton advantages:
- No compilation needed
- Works on NVIDIA, AMD, Intel GPUs
- Comparable performance (~80-90% of CUDA native)
- No version mismatch issues

## PyTorch-Style Multi-Version Wheels (Future)

We plan to publish pre-built wheels for common CUDA versions:
- `decompressed[cu118]` - For CUDA 11.8
- `decompressed[cu121]` - For CUDA 12.1
- `decompressed[cu124]` - For CUDA 12.4

Until then, use `--no-binary=decompressed` for guaranteed compatibility.
