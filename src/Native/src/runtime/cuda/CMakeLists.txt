cmake_minimum_required (VERSION 3.13)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(loaders)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         runtime_function.run.cpp)

if (BUILDING_RUNTIME)
    add_library(runtime_cuda OBJECT ${SRCS})
    set_property(TARGET runtime_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime_cuda EXPORT nncaseruntimeTargets)
    target_link_libraries(runtime_cuda PUBLIC CUDA::cudart)
    target_link_libraries(nncaseruntime PUBLIC runtime_cuda cuda_loaders)
else()
    add_library(simulator_cuda OBJECT ${SRCS})
    target_link_libraries(simulator_cuda PUBLIC simulator cuda_loaders)
    set_property(TARGET simulator_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(nncaseruntime PRIVATE simulator_cuda cuda_loaders)
endif()
