"""Utility functions for error handling and compatibility checks."""

import warnings
import subprocess
import sys


def check_cuda_pytorch_compatibility():
    """
    Check if PyTorch CUDA version matches system CUDA version.
    
    Returns:
        tuple: (is_compatible, system_cuda_version, pytorch_cuda_version, fix_command)
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return (True, None, None, None)  # No CUDA, no problem
        
        pytorch_cuda = torch.version.cuda
        
        # Try to get system CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        system_cuda = line.split('release')[1].split(',')[0].strip()
                        
                        # Map to PyTorch wheel versions
                        major_minor = '.'.join(system_cuda.split('.')[:2])
                        cuda_wheel_map = {
                            '12.6': ('cu124', False),
                            '12.5': ('cu124', False),
                            '12.4': ('cu124', True),
                            '12.3': ('cu121', False),
                            '12.2': ('cu121', False),
                            '12.1': ('cu121', True),
                            '12.0': ('cu118', False),
                            '11.8': ('cu118', True),
                        }
                        
                        wheel_version, exact_match = cuda_wheel_map.get(major_minor, ('cu121', False))
                        pytorch_major = pytorch_cuda.split('.')[0] if pytorch_cuda else ''
                        system_major = major_minor.split('.')[0]
                        
                        is_compatible = pytorch_major == system_major
                        fix_cmd = f"pip install torch --index-url https://download.pytorch.org/whl/{wheel_version}"
                        
                        return (is_compatible, system_cuda, pytorch_cuda, fix_cmd)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # nvcc not available, can't check
        
        return (True, None, pytorch_cuda, None)  # Can't verify, assume OK
        
    except ImportError:
        return (True, None, None, None)


def get_cuda_mismatch_error_message(backend_name="GPU"):
    """Generate helpful error message for CUDA/PyTorch version mismatches."""
    is_compat, sys_cuda, torch_cuda, fix_cmd = check_cuda_pytorch_compatibility()
    
    return (
        f"\n\n{'='*70}\n"
        f"⚠️  {backend_name.upper()} COMPATIBILITY ERROR\n"
        f"{'='*70}\n"
        f"Your PyTorch CUDA version doesn't match your system CUDA.\n"
        f"This causes PTX compilation errors in GPU kernels.\n\n"
        f"System CUDA version: {sys_cuda or 'Unknown (nvcc not found)'}\n"
        f"PyTorch CUDA version: {torch_cuda}\n\n"
        f"{'─'*70}\n"
        f"🔧 HOW TO FIX:\n"
        f"{'─'*70}\n"
        f"1. Install matching PyTorch (RECOMMENDED):\n"
        f"   {fix_cmd if fix_cmd else 'Check: nvcc --version, then install matching torch'}\n\n"
        f"2. Alternative backends:\n"
        f"   • For GPU: load_cvc(..., backend='auto')  # Will try fallbacks\n"
        f"   • For CPU: load_cvc(..., device='cpu')    # Still fast!\n\n"
        f"3. Quick fix commands by CUDA version:\n"
        f"   CUDA 11.8: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
        f"   CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        f"   CUDA 12.4: pip install torch --index-url https://download.pytorch.org/whl/cu124\n\n"
        f"After installing, restart your Python runtime/kernel.\n"
        f"{'='*70}\n"
    )


def get_triton_ptx_error_message(torch_version, device_capability):
    """Generate helpful error message for Triton PTX compilation errors."""
    return get_cuda_mismatch_error_message("Triton")


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
