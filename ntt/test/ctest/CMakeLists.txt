enable_testing()

find_package(ortki)
find_package(GTest REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

macro(add_test_exec name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE GTest::gtest_main nncaseruntime ortki::ortki)
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:${name}> -P ${NNCASE_CMAKE_DIR}/run_test.cmake)
endmacro()

file(GLOB TEST_NAMES CONFIGURE_DEPENDS
    test_ntt_binary_add.cpp

    # test_ntt_binary_mul.cpp
    # test_ntt_binary_sub.cpp
    test_ntt_cast.cpp
    test_ntt_clamp.cpp
    test_ntt_compare_equal.cpp
    test_ntt_compare_greater_or_equal.cpp
    test_ntt_compare_greater.cpp
    test_ntt_compare_less_or_equal.cpp
    test_ntt_compare_less.cpp
    test_ntt_expand.cpp
    test_ntt_gather.cpp
    test_ntt_matmul.cpp
    test_ntt_pack_generated.cpp
    test_ntt_reduce.cpp
    test_ntt_scatter_nd.cpp
)
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_test_exec(${tname})
endforeach()
