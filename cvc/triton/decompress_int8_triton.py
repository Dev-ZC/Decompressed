import triton
import triton.language as tl

@triton.jit
def decompress_int8_triton_kernel(src_ptr, dst_ptr, min_val, scale, n_elements: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask, other=0)
    out = vals * scale + min_val
    tl.store(dst_ptr + offsets, out, mask=mask)
