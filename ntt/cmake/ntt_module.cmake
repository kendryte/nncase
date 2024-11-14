cmake_minimum_required(VERSION 3.15)

include(${CMAKE_CURRENT_LIST_DIR}/compile_flags.cmake)

if (BUILD_STANDALONE)
    add_executable(nncase_ntt_module ${CMAKE_CURRENT_LIST_DIR}/../src/dummy.cpp)
else()
    add_library(nncase_ntt_module SHARED ${CMAKE_CURRENT_LIST_DIR}/../src/dummy.cpp)
endif()

target_compile_features(nncase_ntt_module PUBLIC cxx_std_20)
target_include_directories(nncase_ntt_module PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
set_target_properties(nncase_ntt_module PROPERTIES PREFIX "" SUFFIX "")
set_target_properties(nncase_ntt_module PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(nncase_ntt_module PUBLIC -DNNCASE_CPU_MODULE=1)

target_sources(nncase_ntt_module PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../src/cpu_runtime.cpp)

if (BUILD_STANDALONE)
    target_compile_definitions(nncase_ntt_module PUBLIC -DNNCASE_STANDALONE=1)
endif()

if (MSVC)
    set_property(TARGET nncase_ntt_module PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_target_properties(nncase_ntt_module PROPERTIES LINK_FLAGS /SUBSYSTEM:CONSOLE)
    target_link_options(nncase_ntt_module PRIVATE /NODEFAULTLIB)
    target_link_libraries(nncase_ntt_module PRIVATE libvcruntime msvcrt ucrt "libcpmt$<$<CONFIG:Debug>:d>")
elseif(APPLE)
    target_link_options(nncase_ntt_module PRIVATE -ld_classic -lc)
else()
    target_link_options(nncase_ntt_module PRIVATE -static)
endif()
