enable_testing()

find_package(ortki)
find_package(GTest REQUIRED)
find_package(Python3 REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

# generate ctests sources
message(STATUS "Running generate_pack_tests.py to create generated_tests.cmake")
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_pack_tests.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE return_code
)

if(NOT return_code EQUAL 0)
    message(FATAL_ERROR "generate_pack_tests.py failed to run. Error code: ${return_code}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/generated_tests.cmake)


macro(add_test_exec name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE GTest::gtest_main nncaseruntime ortki::ortki)
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:${name}> -P ${NNCASE_CMAKE_DIR}/run_test.cmake)
endmacro()

file(GLOB HANDWRITTEN_TESTS CONFIGURE_DEPENDS

    # test_ntt_binary_mul.cpp
    # test_ntt_binary_sub.cpp
    # test_ntt_cast.cpp
    # test_ntt_clamp.cpp
    # test_ntt_compare_equal.cpp
    # test_ntt_compare_greater_or_equal.cpp
    # test_ntt_compare_greater.cpp
    # test_ntt_compare_less_or_equal.cpp
    # test_ntt_compare_less.cpp
    test_ntt_expand.cpp
    test_ntt_gather.cpp
    test_ntt_layer_norm.cpp
    test_ntt_matmul.cpp
    # test_ntt_pack_generated_Float8e4m3.cpp
    test_ntt_reduce.cpp
    test_ntt_scatter_nd.cpp
    test_ntt_slice.cpp
    test_ntt_transpose.cpp
    test_ntt_unpack.cpp
    test_ntt_where.cpp
)

list(APPEND TEST_NAMES ${HANDWRITTEN_TESTS} ${GENERATED_TEST_SOURCES})


foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_test_exec(${tname})
endforeach()
