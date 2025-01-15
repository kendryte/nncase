cmake_minimum_required(VERSION 3.15)

include(cmake/compile_flags.cmake)


if(BUILD_TESTING)
   add_subdirectory(test/ctest)
endif()


if(BUILD_BENCHMARK)
    add_subdirectory(test/benchmark_test)
endif()
