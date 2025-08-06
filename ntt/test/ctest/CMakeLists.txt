enable_testing()
# How to run a single test case:
# 1. Build only the specific test executable (using test_ntt_pack_generated_Uint32 as an example):
#    cmake --build build/<BuildType> --target test_ntt_pack_generated_Uint32
#    where <BuildType> is typically Debug, Release, etc.
# 2. Execute the corresponding test in the generated test directory (use the same <BuildType>):
#    ctest -C <BuildType> --test-dir build/<BuildType>/ntt/test/ctest -R test_ntt_pack_generated_Uint32
#    -C specifies the build configuration, and -R filters the test by regex.

find_package(ortki)
find_package(GTest REQUIRED)
find_package(Python3 REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

# --- Generate test source files ---
# Define script paths and output file paths in the source directory
set(GENERATE_PACK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate_pack_tests.py)
set(GENERATED_PACK_CMAKE ${CMAKE_CURRENT_SOURCE_DIR}/generated_pack_tests.cmake)
set(GENERATE_UNPACK_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/generate_unpack_tests.py)
set(GENERATED_UNPACK_CMAKE ${CMAKE_CURRENT_SOURCE_DIR}/generated_unpack_tests.cmake)

# Macro to run a generator script only when the output is missing or outdated
macro(run_generator_if_needed script_path output_file)
    if(NOT EXISTS ${output_file} OR ${script_path} IS_NEWER_THAN ${output_file})
        message(STATUS "Running ${script_path} to generate test files.")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} ${script_path}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE return_code
        )
        if(NOT return_code EQUAL 0)
            message(FATAL_ERROR "${script_path} failed to run. Error code: ${return_code}")
        endif()
    else()
        message(STATUS "Skipping script execution, ${output_file} is up-to-date.")
    endif()
endmacro()

# Run the generators
run_generator_if_needed(${GENERATE_PACK_SCRIPT} ${GENERATED_PACK_CMAKE})
run_generator_if_needed(${GENERATE_UNPACK_SCRIPT} ${GENERATED_UNPACK_CMAKE})

# Include the generated cmake files which define test source variables
include(${GENERATED_PACK_CMAKE})
include(${GENERATED_UNPACK_CMAKE})


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
    test_ntt_rms_norm.cpp
    test_ntt_scatter_nd.cpp
    test_ntt_slice.cpp
    test_ntt_softmax.cpp
    test_ntt_transpose.cpp
    test_ntt_transpose_half.cpp
    test_ntt_unpack.cpp
    # test_ntt_where.cpp
)

list(APPEND TEST_NAMES ${HANDWRITTEN_TESTS} ${GENERATED_PACK_TEST_SOURCES} ${GENERATED_UNPACK_TEST_SOURCES})


foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_test_exec(${tname})
endforeach()
