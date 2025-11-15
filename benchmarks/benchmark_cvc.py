import numpy as np
import time
from python.pycvc import pack_cvc, load_cvc

N, D = 1_000_000, 768
vectors = np.random.rand(N, D).astype(np.float32)

# Pack FP16 and INT8
pack_cvc(vectors, "test_fp16.cvc", compression="fp16")
pack_cvc(vectors, "test_int8.cvc", compression="int8")

# CPU benchmarks
start = time.time()
arr_cpu_fp16 = load_cvc("test_fp16.cvc", device="cpu")
print("CPU FP16 load:", time.time() - start)

start = time.time()
arr_cpu_int8 = load_cvc("test_int8.cvc", device="cpu")
print("CPU INT8 load:", time.time() - start)

# GPU benchmarks (CuPy/Triton)
try:
    import cupy as cp
    start = time.time()
    arr_gpu_fp16 = load_cvc("test_fp16.cvc", device="gpu", framework="cupy")
    cp.cuda.Device(0).synchronize()
    print("GPU FP16 load:", time.time() - start)

    start = time.time()
    arr_gpu_int8 = load_cvc("test_int8.cvc", device="gpu", framework="cupy")
    cp.cuda.Device(0).synchronize()
    print("GPU INT8 load:", time.time() - start)

    print("FP16 max diff:", np.max(np.abs(arr_cpu_fp16 - cp.asnumpy(arr_gpu_fp16))))
    print("INT8 max diff:", np.max(np.abs(arr_cpu_int8 - cp.asnumpy(arr_gpu_int8))))

except Exception as e:
    print("GPU benchmarks skipped:", e)
