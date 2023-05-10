cmake_minimum_required(VERSION 3.13)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "riscv64")
    set(ARCH riscv64)
else()
    set(ARCH x86_64)
endif()

set(SRCS convolution.cpp
    concat.cpp
    slice.cpp
    copy.cpp
    dequantize.cpp
    resize_image.cpp
    gather.cpp
    gather_nd.cpp
    quantize.cpp
    onehot.cpp
    ${ARCH}/binary.cpp
    ${ARCH}/unary.cpp
    ${ARCH}/matmul.cpp
    ${ARCH}/sigmoid.cpp
    ${ARCH}/softmax.cpp
    ${ARCH}/layernorm.cpp
    ${ARCH}/ternary.cpp
    ${ARCH}/reduce.cpp
    ${ARCH}/instancenorm.cpp)
target_sources(kernels PRIVATE ${SRCS})
