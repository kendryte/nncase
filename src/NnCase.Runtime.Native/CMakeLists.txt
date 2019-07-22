cmake_minimum_required (VERSION 3.8)

set(SRCS nceval.cpp
         dataset.cpp
         runtime/interpreter.cpp
         runtime/kernel_registry.cpp
         targets/neutral/neutral_ops.cpp
         targets/cpu/cpu_ops.cpp
         targets/k210/k210_ops.cpp
         targets/k210/interpreter.cpp)

if (MSVC_VERSION GREATER_EQUAL "1900")
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    if (_cpp_latest_flag_supported)
        add_compile_options("/std:c++latest")
        add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    endif()
endif()

if (WIN32)
    add_definitions(/DNNCASE_TARGET=${NNCASE_TARGET})
else()
    add_definitions(-DNNCASE_TARGET=${NNCASE_TARGET})
endif()

add_executable (nceval ${SRCS})
target_include_directories(nceval PUBLIC include)
target_include_directories(nceval PUBLIC ${THIRD_PARTY}/xtl/include)
target_include_directories(nceval PRIVATE ${THIRD_PARTY}/clipp/include ${THIRD_PARTY}/xtensor/include)
target_link_libraries(nceval PRIVATE CONAN_PKG::opencv)
