
include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

if(CPU_FREQUENCY_MHZ)
    add_definitions(-DCPU_FREQUENCY_MHZ=${CPU_FREQUENCY_MHZ})
endif()

if(CLOCK_SOURCE_FREQUENCY_MHZ)
    add_definitions(-DCLOCK_SOURCE_FREQUENCY_MHZ=${CLOCK_SOURCE_FREQUENCY_MHZ})
endif()

file(GLOB TEST_NAMES CONFIGURE_DEPENDS
    benchmark_ntt_binary.cpp
    # benchmark_ntt_cast.cpp
    benchmark_ntt_clamp.cpp
    benchmark_ntt_compare.cpp
    benchmark_ntt_expand.cpp
    benchmark_ntt_gather.cpp
    benchmark_ntt_layernorm.cpp
    benchmark_ntt_matmul_primitive_size.cpp
    benchmark_ntt_matmul.cpp
    benchmark_ntt_vectorize.cpp
    benchmark_ntt_reduce.cpp
    benchmark_ntt_rmsnorm.cpp
    benchmark_ntt_scatter_nd.cpp
    benchmark_ntt_slice.cpp
    benchmark_ntt_softmax.cpp
    benchmark_ntt_transpose.cpp
    benchmark_ntt_unary.cpp
    benchmark_ntt_devectorize.cpp
    benchmark_ntt_where.cpp
)
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} PRIVATE nncaseruntime)
endforeach()