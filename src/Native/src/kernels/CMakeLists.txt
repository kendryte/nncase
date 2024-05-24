cmake_minimum_required (VERSION 3.8)

set(SRCS kernel_context.cpp)

if (BUILDING_RUNTIME)
    # used for rvv
    if(ENABLE_RVV)
        add_definitions(-D__riscv_vector=1)
        add_compile_options(-march=rv64imafdcv)
    endif()

    add_library(kernels OBJECT ${SRCS})
    target_include_directories(kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    set_property(TARGET kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS kernels EXPORT nncaseruntimeTargets)
else()
    add_library(kernels OBJECT ${SRCS})
    target_include_directories(kernels PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    
    target_compile_definitions(kernels PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    set_property(TARGET kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

if(ENABLE_OPENMP)
    target_link_libraries(kernels PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(kernels PRIVATE "-DNNCASE_OPENMP")
endif()

if(APPLE)
  target_compile_options(kernels PRIVATE -Wno-gnu-zero-variadic-macro-arguments -Wno-array-bounds)  
else()
  target_compile_options(kernels PRIVATE -Wno-array-bounds -Wno-stringop-overread)
endif(APPLE)


add_subdirectory(stackvm)
