cmake_minimum_required (VERSION 3.13)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
    set(ARCH riscv64)
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(x86)|(X86)|(amd64)|(AMD64)|(x86_64)|(X86_64)")
    set(ARCH x86_64)
else()
    set(ARCH generic)
endif()

if (NOT ${ARCH} STREQUAL "generic")
    add_subdirectory(${ARCH})
endif()

set(SRCS concat.cpp
         convolution.cpp
         slice.cpp
         dequantize.cpp
         resize_image.cpp
         gather.cpp
         gather_nd.cpp
         quantize.cpp
         onehot.cpp
         transpose.cpp
)

function(_TARGET_ARCH_FILES)
    set(oneValueArgs TARGET)
    set(multiValueArgs FILES)
    cmake_parse_arguments(ARGS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(FILE ${ARGS_FILES})
        set(ARCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}/${FILE}")
        if(EXISTS ${ARCH_FILE})
            target_sources(${ARGS_TARGET} PRIVATE ${ARCH_FILE})
        else()
            target_sources(${ARGS_TARGET} PRIVATE "${FILE}")
        endif()
    endforeach()
endfunction()

target_sources(kernels PRIVATE ${SRCS})

_TARGET_ARCH_FILES(TARGET kernels
                   FILES
                   binary.cpp
                   layer_norm.cpp
                   #matmul.cpp
                   sigmoid.cpp
                   softmax.cpp
                   unary.cpp
                   where.cpp
                   log_softmax.cpp
                   reduce.cpp
                   gather_elements.cpp
                   tile.cpp
                   topk.cpp
)

if(NOT MSVC)
    target_compile_options(kernels PRIVATE -Wno-sign-compare)
endif()
