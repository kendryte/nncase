
include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

if(CPU_FREQUENCY_MHZ)
    add_definitions(-DCPU_FREQUENCY_MHZ=${CPU_FREQUENCY_MHZ})
endif()

if(CLOCK_SOURCE_FREQUENCY_MHZ)
    add_definitions(-DCLOCK_SOURCE_FREQUENCY_MHZ=${CLOCK_SOURCE_FREQUENCY_MHZ})
endif()

file(GLOB TEST_NAMES CONFIGURE_DEPENDS
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
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} PRIVATE nncaseruntime)
endforeach()