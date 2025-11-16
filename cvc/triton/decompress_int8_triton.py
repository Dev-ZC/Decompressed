import triton
import triton.language as tl

@triton.jit
def decompress_int8_triton_kernel(src_ptr, dst_ptr, min_val, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Ensure pointers are int64 for indexing
    src_ptr_int = src_ptr.to(tl.int64)
    dst_ptr_int = dst_ptr.to(tl.int64)
    
    # Load uint8 values and decompress to fp32
    ptrs_src = src_ptr_int + offsets * 1  # 1 byte per uint8
    ptrs_dst = dst_ptr_int + offsets * 4  # 4 bytes per fp32
    
    vals_uint8 = tl.load(ptrs_src, mask=mask, other=0)
    vals_fp32 = vals_uint8.to(tl.float32)
    out = vals_fp32 * scale + min_val
    tl.store(ptrs_dst, out, mask=mask)
