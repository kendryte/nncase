
include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

if(CPU_FREQUENCY_MHZ)
    add_definitions(-DCPU_FREQUENCY_MHZ=${CPU_FREQUENCY_MHZ})
endif()

if(CLOCK_SOURCE_FREQUENCY_MHZ)
    add_definitions(-DCLOCK_SOURCE_FREQUENCY_MHZ=${CLOCK_SOURCE_FREQUENCY_MHZ})
endif()

# clear TEST_NAMES
set(TEST_NAMES)

# add f32 test files
file(GLOB TEST_NAMES_F32 CONFIGURE_DEPENDS
    benchmark_ntt_f32_binary.cpp
    # benchmark_ntt_f32_cast.cpp
    benchmark_ntt_f32_clamp.cpp
    benchmark_ntt_f32_compare.cpp
    benchmark_ntt_f32_expand.cpp
    benchmark_ntt_f32_gather.cpp
    benchmark_ntt_f32_layernorm.cpp
    benchmark_ntt_f32_matmul_primitive_size.cpp
    benchmark_ntt_f32_matmul.cpp
    benchmark_ntt_f32_pack.cpp
    benchmark_ntt_f32_reduce.cpp
    benchmark_ntt_f32_rmsnorm.cpp
    benchmark_ntt_f32_scatter_nd.cpp
    benchmark_ntt_f32_slice.cpp
    benchmark_ntt_f32_softmax.cpp
    benchmark_ntt_f32_transpose.cpp
    benchmark_ntt_f32_unary.cpp
    benchmark_ntt_f32_unpack.cpp
    benchmark_ntt_f32_where.cpp
)

# add fp16 test files
file(GLOB TEST_NAMES_FP16 CONFIGURE_DEPENDS
    benchmark_ntt_fp16_binary.cpp
    benchmark_ntt_fp16_unary.cpp
    benchmark_ntt_fp16_clamp.cpp
    benchmark_ntt_fp16_compare.cpp
    benchmark_ntt_fp16_expand.cpp
    benchmark_ntt_fp16_gather.cpp
    benchmark_ntt_fp16_pack.cpp
    benchmark_ntt_fp16_reduce.cpp
    benchmark_ntt_fp16_scatter_nd.cpp
    benchmark_ntt_fp16_slice.cpp
    benchmark_ntt_fp16_transpose.cpp
    benchmark_ntt_fp16_unpack.cpp
    benchmark_ntt_fp16_where.cpp
)

# merge F32 AND FP16 INTO TEST_NAMES
list(APPEND TEST_NAMES ${TEST_NAMES_F32} ${TEST_NAMES_FP16})

foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} PRIVATE nncaseruntime)
endforeach()