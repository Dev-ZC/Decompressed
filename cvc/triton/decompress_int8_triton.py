import triton
import triton.language as tl

@triton.jit
def decompress_int8_triton_kernel(src_ptr, dst_ptr, min_val, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Direct pointer arithmetic
    src_ptrs = src_ptr + offsets
    dst_ptrs = dst_ptr + offsets
    
    # Load uint8, convert to float32 and decompress
    vals = tl.load(src_ptrs, mask=mask, other=0)
    vals_fp32 = vals.to(tl.float32)
    out = vals_fp32 * scale + min_val
    tl.store(dst_ptrs, out, mask=mask)
