cmake_minimum_required(VERSION 3.15)

include(${CMAKE_CURRENT_LIST_DIR}/compile_flags.cmake)

add_library(nncase_cpu_runtime STATIC ${CMAKE_CURRENT_LIST_DIR}/cpu_runtime.cpp)
target_compile_features(nncase_cpu_runtime PUBLIC cxx_std_20)
target_include_directories(nncase_cpu_runtime PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

if (MSVC)
    set_property(TARGET nncase_cpu_runtime PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
