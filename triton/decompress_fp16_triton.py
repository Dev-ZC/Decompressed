import triton
import triton.language as tl

@triton.jit
def decompress_fp16_kernel(src_ptr, dst_ptr, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < n_elements
    data = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(dst_ptr + offs, data, mask=mask)
