cmake_minimum_required (VERSION 3.13)

set(SRCS cuda_loader.cpp)

add_library(cuda_loaders OBJECT ${SRCS})
set_property(TARGET cuda_loaders PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_loaders PUBLIC CUDA::cudart)

if (BUILDING_RUNTIME)
    install(TARGETS cuda_loaders EXPORT nncaseruntimeTargets)
endif()
