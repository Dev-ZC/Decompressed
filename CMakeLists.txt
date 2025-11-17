cmake_minimum_required(VERSION 3.18)
project(Decompressed LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" OFF)

# Auto-detect CUDA availability (disable on macOS by default)
if(APPLE)
    option(BUILD_CUDA "Build CUDA kernels" OFF)
else()
    option(BUILD_CUDA "Build CUDA kernels" ON)
endif()

# Find packages
set(CUDA_ACTUALLY_AVAILABLE OFF)
if(BUILD_CUDA)
    # Try to find CUDA (not REQUIRED - graceful fallback if missing)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Use native architecture for best performance, fallback to common architectures
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")
        endif()
        set(CUDA_ACTUALLY_AVAILABLE ON)
        message(STATUS "✅ CUDA Toolkit found: Building with CUDA support")
    else()
        message(STATUS "⚠️  CUDA Toolkit not found: Building without CUDA support")
        set(BUILD_CUDA OFF)
    endif()
endif()

if(BUILD_PYTHON_BINDINGS)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        # Fall back to Python-installed pybind11
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 REQUIRED)
    endif()
endif()

# Source files
set(CVC_SOURCES
    cvc/cpu/cvc.cpp
)

set(CVC_HEADERS
    cvc/cpu/cvc.h
)

if(BUILD_CUDA)
    set(CUDA_SOURCES
        cvc/cuda/decompress_fp16.cu
        cvc/cuda/decompress_int8.cu
    )
endif()

# CPU-only library
add_library(cvc_cpu STATIC ${CVC_SOURCES} ${CVC_HEADERS})
target_include_directories(cvc_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(cvc_cpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cvc_cpu PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -march=native -Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W4>
)

# CUDA library
if(BUILD_CUDA)
    add_library(cvc_cuda STATIC ${CUDA_SOURCES} ${CVC_HEADERS})
    target_include_directories(cvc_cuda PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(cvc_cuda PUBLIC CUDA::cudart)
    
    # CUDA compilation flags
    set_target_properties(cvc_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_compile_options(cvc_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --expt-relaxed-constexpr
            -O3
            -Xcompiler=-fPIC
        >
    )
    
    # Combined library
    add_library(cvc STATIC ${CVC_SOURCES} ${CUDA_SOURCES} ${CVC_HEADERS})
    target_include_directories(cvc PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(cvc PUBLIC CUDA::cudart)
    set_target_properties(cvc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(cvc ALIAS cvc_cpu)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    # Compile sources directly into Python module (avoids static lib linking issues)
    if(CUDA_ACTUALLY_AVAILABLE)
        # Create object library for CUDA sources to handle device linking
        add_library(cuda_objs OBJECT ${CUDA_SOURCES})
        set_target_properties(cuda_objs PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF  # Compile without separable compilation
            POSITION_INDEPENDENT_CODE ON
            CUDA_RESOLVE_DEVICE_SYMBOLS OFF
        )
        target_include_directories(cuda_objs PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_compile_options(cuda_objs PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                --use_fast_math
                --expt-relaxed-constexpr
                -Xcompiler=-fPIC
            >
        )
        
        # Create CPU object library
        add_library(cpu_objs OBJECT ${CVC_SOURCES})
        set_target_properties(cpu_objs PROPERTIES
            POSITION_INDEPENDENT_CODE ON
        )
        target_include_directories(cpu_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        
        # Now create the Python module with both CPU and CUDA objects
        pybind11_add_module(_cvc_native MODULE
            python/bindings/cvc_bindings.cpp
            $<TARGET_OBJECTS:cpu_objs>
            $<TARGET_OBJECTS:cuda_objs>
        )
        
        target_link_libraries(_cvc_native PRIVATE CUDA::cudart)
        target_compile_definitions(_cvc_native PRIVATE WITH_CUDA=1)
        set_target_properties(_cvc_native PROPERTIES
            POSITION_INDEPENDENT_CODE ON
        )
        target_include_directories(_cvc_native PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
    else()
        pybind11_add_module(_cvc_native MODULE
            python/bindings/cvc_bindings.cpp
            ${CVC_SOURCES}
        )
        target_compile_definitions(_cvc_native PRIVATE WITH_CUDA=0)
        target_include_directories(_cvc_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    
    # Install to python package directory
    # For scikit-build-core: install to the root (scikit-build-core handles the package path)
    install(TARGETS _cvc_native 
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS cvc_cpu
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

if(BUILD_CUDA)
    install(TARGETS cvc_cuda cvc
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
    )
endif()

install(FILES ${CVC_HEADERS} DESTINATION include/cvc/cpu)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Decompressed Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Support: ${CUDA_ACTUALLY_AVAILABLE}")
if(CUDA_ACTUALLY_AVAILABLE)
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
    message(STATUS "🚀 GPU acceleration ENABLED (CUDA native)")
else()
    message(STATUS "⚠️  GPU acceleration via C++/CUDA not available")
    message(STATUS "   Triton GPU kernels will be used if installed (pip install triton)")
endif()
message(STATUS "Python Bindings: ${BUILD_PYTHON_BINDINGS}")
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
endif()
message(STATUS "========================================")
message(STATUS "")
