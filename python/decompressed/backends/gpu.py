"""GPU-based decompression backends."""

import numpy as np
from .base import BackendInterface
from ..utils import get_cuda_mismatch_error_message, get_triton_ptx_error_message, warn_triton_fallback


class CUDABackend(BackendInterface):
    """CUDA native GPU decompression backend (NVIDIA only)."""
    
    def __init__(self):
        try:
            from decompressed._cvc_native import (
                decompress_fp16_cuda,
                decompress_int8_cuda,
                CUDA_AVAILABLE
            )
            self.decompress_fp16 = decompress_fp16_cuda
            self.decompress_int8 = decompress_int8_cuda
            self._available = CUDA_AVAILABLE
        except ImportError:
            self._available = False
    
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset, framework="torch"):
        """Decompress using CUDA native kernels."""
        try:
            # Convert payload to numpy array
            if compression == "fp16":
                src_data = np.frombuffer(payload, dtype=np.float16).copy()
            else:  # int8
                src_data = np.frombuffer(payload, dtype=np.uint8).copy()
            
            # Upload to GPU and decompress based on framework
            if framework == "torch":
                import torch
                src_gpu = torch.from_numpy(src_data).cuda()
                dst_slice = arr[offset:offset+rows].flatten()
                n_elements = rows * dim
                
                if compression == "fp16":
                    self.decompress_fp16(
                        src_gpu.data_ptr(),
                        dst_slice.data_ptr(),
                        n_elements
                    )
                else:  # int8
                    self.decompress_int8(
                        src_gpu.data_ptr(),
                        dst_slice.data_ptr(),
                        chunk_meta["min"],
                        chunk_meta["scale"],
                        n_elements
                    )
                torch.cuda.synchronize()
                
            elif framework == "cupy":
                import cupy as cp
                src_gpu = cp.asarray(src_data)
                dst_slice = arr[offset:offset+rows].flatten()
                n_elements = rows * dim
                
                if compression == "fp16":
                    self.decompress_fp16(
                        src_gpu.data.ptr,
                        dst_slice.data.ptr,
                        n_elements
                    )
                else:  # int8
                    self.decompress_int8(
                        src_gpu.data.ptr,
                        dst_slice.data.ptr,
                        chunk_meta["min"],
                        chunk_meta["scale"],
                        n_elements
                    )
                cp.cuda.Device(0).synchronize()
                
        except RuntimeError as e:
            error_msg = str(e)
            # Check for PTX/CUDA compilation errors
            if any(keyword in error_msg.lower() for keyword in ['ptx', 'cuda error', 'unsupported toolchain']):
                help_msg = get_cuda_mismatch_error_message("CUDA Native")
                # Print help message but don't suppress original error
                print(help_msg, file=__import__('sys').stderr)
                # Re-raise original error with full traceback
                raise
            else:
                # Some other CUDA error
                raise
    
    def is_available(self):
        """Check if CUDA native extensions are built."""
        return self._available


class TritonBackend(BackendInterface):
    """Triton GPU decompression backend (vendor-agnostic)."""
    
    def __init__(self):
        self._available = False
        self._error_msg = None
        
        try:
            import triton
            from ..triton.decompress_fp16_triton import decompress_fp16_kernel
            from ..triton.decompress_int8_triton import decompress_int8_triton_kernel as decompress_int8_kernel
            
            self.decompress_fp16_kernel = decompress_fp16_kernel
            self.decompress_int8_kernel = decompress_int8_kernel
            self.triton = triton
            self._available = True
        except Exception as e:
            self._error_msg = f"{type(e).__name__}: {str(e)}"
    
    def decompress_chunk(self, payload, rows, dim, compression, chunk_meta, arr, offset, 
                        framework="torch", cuda_fallback=None):
        """
        Decompress using Triton kernels.
        
        Args:
            cuda_fallback: CUDABackend instance to fall back to on PTX errors (optional)
        """
        try:
            self._decompress_chunk_impl(payload, rows, dim, compression, chunk_meta, arr, offset, framework)
        except RuntimeError as e:
            error_msg = str(e)
            if "PTX" in error_msg and "toolchain" in error_msg:
                # Handle PTX compilation error
                self._handle_ptx_error(e, payload, rows, dim, compression, chunk_meta, 
                                      arr, offset, framework, cuda_fallback)
            else:
                raise
    
    def _decompress_chunk_impl(self, payload, rows, dim, compression, chunk_meta, arr, offset, framework):
        """Internal implementation of chunk decompression."""
        # Convert payload to numpy array
        if compression == "fp16":
            src_data = np.frombuffer(payload, dtype=np.float16).copy()
        else:  # int8
            src_data = np.frombuffer(payload, dtype=np.uint8).copy()
        
        n_elements = rows * dim
        BLOCK_SIZE = 1024
        grid = lambda meta: (self.triton.cdiv(n_elements, BLOCK_SIZE),)
        
        if framework == "torch":
            import torch
            src_gpu = torch.from_numpy(src_data).cuda()
            dst_slice = arr[offset:offset+rows].flatten()
            
            if compression == "fp16":
                self.decompress_fp16_kernel[grid](
                    src_gpu, 
                    dst_slice,
                    n_elements,
                    BLOCK_SIZE
                )
            else:  # int8
                self.decompress_int8_kernel[grid](
                    src_gpu,
                    dst_slice,
                    chunk_meta["min"],
                    chunk_meta["scale"],
                    n_elements,
                    BLOCK_SIZE
                )
            torch.cuda.synchronize()
            
        elif framework == "cupy":
            import cupy as cp
            import torch
            
            # Convert CuPy to PyTorch for Triton
            src_torch = torch.as_tensor(cp.asarray(src_data), device='cuda')
            dst_slice = arr[offset:offset+rows].flatten()
            dst_torch = torch.as_tensor(dst_slice, device='cuda')
            
            if compression == "fp16":
                self.decompress_fp16_kernel[grid](
                    src_torch,
                    dst_torch,
                    n_elements,
                    BLOCK_SIZE
                )
            else:  # int8
                self.decompress_int8_kernel[grid](
                    src_torch,
                    dst_torch,
                    chunk_meta["min"],
                    chunk_meta["scale"],
                    n_elements,
                    BLOCK_SIZE
                )
            torch.cuda.synchronize()
    
    def _handle_ptx_error(self, error, payload, rows, dim, compression, chunk_meta, 
                         arr, offset, framework, cuda_fallback):
        """Handle Triton PTX compilation errors with helpful messages and fallback."""
        import torch
        import sys
        
        help_msg = get_triton_ptx_error_message(
            torch.version.cuda,
            torch.cuda.get_device_capability()
        )
        
        # Try to fall back to CUDA native if available
        if cuda_fallback is not None and cuda_fallback.is_available():
            warn_triton_fallback(help_msg)
            cuda_fallback.decompress_chunk(
                payload, rows, dim, compression, chunk_meta, arr, offset, framework
            )
        else:
            # No fallback available - print help and re-raise original error
            print(help_msg, file=sys.stderr)
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"ORIGINAL ERROR (full traceback below):", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)
            raise
    
    def is_available(self):
        """Check if Triton is installed and importable."""
        return self._available
    
    def get_error(self):
        """Get the error message if backend failed to load."""
        return self._error_msg
