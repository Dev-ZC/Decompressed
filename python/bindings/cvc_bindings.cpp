#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cvc/cpu/cvc.h"

namespace py = pybind11;

// Wrapper for FP16 CPU decompression
py::array_t<float> decompress_fp16_cpu(py::array_t<uint16_t> src) {
    py::buffer_info src_info = src.request();
    size_t n = src_info.size;
    
    auto result = py::array_t<float>(n);
    py::buffer_info dst_info = result.request();
    
    cvc_decompress_fp16(
        static_cast<const uint16_t*>(src_info.ptr),
        static_cast<float*>(dst_info.ptr),
        n
    );
    
    return result;
}

// Wrapper for INT8 CPU decompression
py::array_t<float> decompress_int8_cpu(py::array_t<uint8_t> src, float minv, float scale) {
    py::buffer_info src_info = src.request();
    size_t n = src_info.size;
    
    auto result = py::array_t<float>(n);
    py::buffer_info dst_info = result.request();
    
    cvc_decompress_int8(
        static_cast<const uint8_t*>(src_info.ptr),
        static_cast<float*>(dst_info.ptr),
        minv,
        scale,
        n
    );
    
    return result;
}

#ifdef WITH_CUDA
// Wrapper for FP16 CUDA decompression (device pointers)
void decompress_fp16_cuda_wrapper(uintptr_t src_ptr, uintptr_t dst_ptr, size_t n) {
    cvc_decompress_fp16_cuda(
        reinterpret_cast<const uint16_t*>(src_ptr),
        reinterpret_cast<float*>(dst_ptr),
        n
    );
}

// Wrapper for INT8 CUDA decompression (device pointers)
void decompress_int8_cuda_wrapper(uintptr_t src_ptr, uintptr_t dst_ptr, float minv, float scale, size_t n) {
    cvc_decompress_int8_cuda(
        reinterpret_cast<const uint8_t*>(src_ptr),
        reinterpret_cast<float*>(dst_ptr),
        minv,
        scale,
        n
    );
}
#endif

PYBIND11_MODULE(_cvc_native, m) {
    m.doc() = "Native C++/CUDA decompression kernels for CVC format";
    
    // CPU functions
    m.def("decompress_fp16_cpu", &decompress_fp16_cpu,
          py::arg("src"),
          "Decompress FP16 data to FP32 on CPU");
    
    m.def("decompress_int8_cpu", &decompress_int8_cpu,
          py::arg("src"), py::arg("min"), py::arg("scale"),
          "Decompress INT8 data to FP32 on CPU");
    
#ifdef WITH_CUDA
    // CUDA functions (raw pointer interface)
    m.def("decompress_fp16_cuda", &decompress_fp16_cuda_wrapper,
          py::arg("src_ptr"), py::arg("dst_ptr"), py::arg("n"),
          "Decompress FP16 data to FP32 on GPU (device pointers)");
    
    m.def("decompress_int8_cuda", &decompress_int8_cuda_wrapper,
          py::arg("src_ptr"), py::arg("dst_ptr"), py::arg("min"), py::arg("scale"), py::arg("n"),
          "Decompress INT8 data to FP32 on GPU (device pointers)");
    
    m.attr("CUDA_AVAILABLE") = true;
#else
    m.attr("CUDA_AVAILABLE") = false;
#endif
    
    m.attr("__version__") = "0.1.0";
}
