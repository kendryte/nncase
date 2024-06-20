cmake_minimum_required (VERSION 3.13)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRCS
    binary.cpp
    layer_norm.cpp
    log_softmax.cpp
    matmul.cpp
    reduce.cpp
    sigmoid.cpp
    unary.cpp
    where.cpp
)

add_library(riscv_kernel STATIC ${SRCS})

install(TARGETS riscv_kernel
    EXPORT kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)