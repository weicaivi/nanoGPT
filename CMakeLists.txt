cmake_minimum_required(VERSION 3.18)
project(nanoGPT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set LibTorch paths
set(TORCH_ROOT "/Users/weicai/nanoGPT/libtorch")
set(TORCH_INCLUDE_DIRS "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include")

# Add executable
add_executable(train train.cpp)

# Include directories
target_include_directories(train PRIVATE ${TORCH_INCLUDE_DIRS})

# Platform-specific settings for macOS
if(APPLE)
    # Link against all necessary LibTorch libraries
    target_link_libraries(train 
        "${TORCH_ROOT}/lib/libtorch.dylib"
        "${TORCH_ROOT}/lib/libtorch_cpu.dylib"
        "${TORCH_ROOT}/lib/libc10.dylib"
        "${TORCH_ROOT}/lib/libtorch_global_deps.dylib"
        "${TORCH_ROOT}/lib/libshm.dylib"
        "${TORCH_ROOT}/lib/libomp.dylib"
        # Static libraries
        "${TORCH_ROOT}/lib/libkineto.a"
        "${TORCH_ROOT}/lib/libprotobuf.a"
        "${TORCH_ROOT}/lib/libonnx.a"
        "${TORCH_ROOT}/lib/libtensorpipe.a"
        "${TORCH_ROOT}/lib/libfmt.a"
        "${TORCH_ROOT}/lib/libgloo.a"
        "${TORCH_ROOT}/lib/libcpuinfo.a"
        "${TORCH_ROOT}/lib/libpytorch_qnnpack.a"
        "${TORCH_ROOT}/lib/libXNNPACK.a"
        "${TORCH_ROOT}/lib/libnnpack.a"
        "${TORCH_ROOT}/lib/libpthreadpool.a"
        "${TORCH_ROOT}/lib/libprotobuf-lite.a"
        "${TORCH_ROOT}/lib/libonnx_proto.a"
        "${TORCH_ROOT}/lib/libtensorpipe_uv.a"
        "${TORCH_ROOT}/lib/libuv_a.a"
        "${TORCH_ROOT}/lib/libclog.a"
        "${TORCH_ROOT}/lib/libcpuinfo_internals.a"
        "${TORCH_ROOT}/lib/libnnpack_reference_layers.a"
        "${TORCH_ROOT}/lib/libprotoc.a"
        "${TORCH_ROOT}/lib/libkleidiai.a"
        "${TORCH_ROOT}/lib/libmicrokernels-all.a"
        "${TORCH_ROOT}/lib/libmicrokernels-prod.a"
        # System frameworks
        "-framework Accelerate"
        "-framework CoreFoundation"
        "-framework SystemConfiguration"
    )
    
    # Set RPATH
    set_target_properties(train PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@executable_path/../libtorch/lib"
    )
else()
    # For Linux, use the standard approach
    find_package(Torch REQUIRED)
    target_link_libraries(train ${TORCH_LIBRARIES})
endif()

# Set compiler flags
set_property(TARGET train PROPERTY CXX_STANDARD 17)

# Print some information
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Building for platform: ${CMAKE_SYSTEM_NAME}") 