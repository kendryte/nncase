cmake_minimum_required (VERSION 3.8)

set(SRCS kernel_context.cpp)

if(C900_RISCV_ROOT_PATH)
    set(CMAKE_C_COMPILER ${RISCV_C_COMPILER} CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER ${RISCV_CXX_COMPILER} CACHE STRING "" FORCE)
    set(CMAKE_FIND_ROOT_PATH ${RISCV_FIND_ROOT_PATH} CACHE STRING "" FORCE)
endif()

if (BUILDING_RUNTIME)
    # used for rvv
    if(ENABLE_RVV)
        add_definitions(-D__riscv_vector=1)
        add_compile_options(-march=rv64imafdcv)
    endif()

    add_library(kernels OBJECT ${SRCS})
    target_include_directories(kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(kernels PUBLIC gsl::gsl-lite)
    set_property(TARGET kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS kernels EXPORT nncaseruntimeTargets)
else()
    add_library(kernels OBJECT ${SRCS})
    target_include_directories(kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(kernels PUBLIC gsl::gsl-lite)
    if(ENABLE_HALIDE)
        hkg_get_runtime_lib(hkg_runtime_lib os_name)
        hkg_get_suffix(obj_suffix lib_suffix)
        target_link_libraries(kernels PRIVATE hkg::${os_name}_src ${hkg_runtime_lib})
        if(os_name STREQUAL "linux")
            target_link_libraries(kernels PRIVATE -lpthread)
        endif()
        target_compile_definitions(kernels PRIVATE "-DNNCASE_HALIDE")
    endif()

    target_compile_definitions(kernels PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    set_property(TARGET kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(ENABLE_OPENMP)
    target_link_libraries(kernels PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(kernels PRIVATE "-DNNCASE_OPENMP")
endif()

if(APPLE)
  target_compile_options(kernels PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
endif(APPLE)


add_subdirectory(stackvm)
