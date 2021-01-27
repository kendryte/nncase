cmake_minimum_required (VERSION 3.8)

set(SRCS convolution.cpp)

if (BUILDING_RUNTIME)
    add_library(kernels OBJECT ${SRCS})
    target_include_directories(kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(kernels PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant CONAN_PKG::xtensor)
    target_link_libraries(kernels PRIVATE CONAN_PKG::fmt)

    if (BUILD_SHARED_LIBS)
        target_compile_definitions(kernels PUBLIC -DNNCASE_DLL)
    endif()
else()
    add_library(kernels OBJECT ${SRCS})
    target_include_directories(kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(kernels PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant CONAN_PKG::xtensor)
    target_link_libraries(kernels PRIVATE CONAN_PKG::fmt)
    target_compile_definitions(kernels PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
endif()

add_subdirectory(cpu)
