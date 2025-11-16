import triton
import triton.language as tl

@triton.jit
def decompress_fp16_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Ensure pointers are int64 for indexing
    src_ptr_int = src_ptr.to(tl.int64)
    dst_ptr_int = dst_ptr.to(tl.int64)
    
    # Load FP16 data and convert to FP32
    ptrs_src = src_ptr_int + offsets * 2  # 2 bytes per fp16
    ptrs_dst = dst_ptr_int + offsets * 4  # 4 bytes per fp32
    
    data_fp16 = tl.load(ptrs_src, mask=mask, other=0)
    data_fp32 = data_fp16.to(tl.float32)
    tl.store(ptrs_dst, data_fp32, mask=mask)
