import triton
import triton.language as tl

@triton.jit
def decompress_fp16_kernel(src_ptr, dst_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Load and store with proper pointer indexing
    data = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, data, mask=mask)
