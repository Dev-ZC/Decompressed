"""Utility functions for error handling and compatibility checks."""

import warnings


def get_triton_ptx_error_message(torch_version, device_capability):
    """Generate helpful error message for Triton PTX compilation errors."""
    return (
        f"\n\n{'='*70}\n"
        f"⚠️  TRITON COMPATIBILITY ERROR\n"
        f"{'='*70}\n"
        f"Triton cannot compile kernels because your PyTorch CUDA version\n"
        f"doesn't match your system CUDA driver.\n\n"
        f"Your PyTorch CUDA version: {torch_version}\n"
        f"System CUDA capability: {device_capability}\n\n"
        f"FIX OPTIONS:\n"
        f"1. Install matching PyTorch (recommended):\n"
        f"   # Check system CUDA: nvcc --version\n"
        f"   # For CUDA 11.8:\n"
        f"   pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
        f"   # For CUDA 12.1:\n"
        f"   pip install torch --index-url https://download.pytorch.org/whl/cu121\n\n"
        f"2. Use CUDA native backend (if available):\n"
        f"   load_cvc(..., backend='cuda')  # Faster anyway!\n\n"
        f"3. Use CPU backend:\n"
        f"   load_cvc(..., device='cpu')\n"
        f"{'='*70}\n"
    )


def warn_triton_fallback(help_msg):
    """Warn user about Triton fallback to CUDA native."""
    warnings.warn(
        f"Triton backend failed (PTX error). Falling back to CUDA native backend.\n{help_msg}",
        RuntimeWarning,
        stacklevel=3
    )


def validate_backend_availability(backend, device, has_native, has_cuda, has_triton):
    """Validate that the requested backend is available and compatible with device."""
    if backend == "cpp" and not has_native:
        raise RuntimeError("C++ backend requested but not available. Build with: pip install .")
    
    if backend == "cuda" and not has_cuda:
        raise RuntimeError("CUDA native backend requested but not available. Build with: pip install .")
    
    if backend == "triton" and not has_triton:
        raise RuntimeError("Triton backend requested but not available. Install: pip install triton")
    
    if backend in ["cuda", "triton"] and device == "cpu":
        raise ValueError(f"Backend '{backend}' requires device='cuda', not 'cpu'")
    
    if backend in ["python", "cpp"] and device != "cpu":
        raise ValueError(f"Backend '{backend}' requires device='cpu', not '{device}'")


def select_backend(backend, device, has_native, has_cuda, has_triton):
    """Auto-select the best backend based on device and available backends."""
    if backend != "auto":
        return backend
    
    if device == "cpu":
        return "cpp" if has_native else "python"
    else:  # GPU
        # Priority: CUDA native > Triton > fallback error
        if has_cuda:
            return "cuda"
        elif has_triton:
            return "triton"
        else:
            raise RuntimeError("GPU requested but no GPU backend available. Install: pip install triton")
