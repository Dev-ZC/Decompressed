import triton
import triton.language as tl

@triton.jit
def decompress_int8_triton_kernel(src_ptr, dst_ptr, min_val, scale, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load uint8 values and decompress
    vals = tl.load(src_ptr + offsets, mask=mask, other=0)
    out = vals * scale + min_val
    tl.store(dst_ptr + offsets, out, mask=mask)
