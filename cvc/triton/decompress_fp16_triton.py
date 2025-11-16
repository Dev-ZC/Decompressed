import triton
import triton.language as tl

@triton.jit
def decompress_fp16_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Direct pointer arithmetic - Triton handles typing internally
    # Load FP16 and convert to FP32
    src_ptrs = src_ptr + offsets
    dst_ptrs = dst_ptr + offsets
    
    data = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, data, mask=mask)
