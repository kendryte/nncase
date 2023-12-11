cmake_minimum_required (VERSION 3.13)

add_subdirectory(loaders)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         runtime_function.run.cpp)

if (BUILDING_RUNTIME)
    add_library(runtime_cpu OBJECT ${SRCS})
    target_link_libraries(runtime_cpu PUBLIC runtime cpu_loaders)
    target_link_libraries(runtime_cpu PRIVATE kernels)
    set_property(TARGET runtime_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime_cpu EXPORT nncaseruntimeTargets)
else()
    add_library(simulator_cpu OBJECT ${SRCS})
    target_link_libraries(simulator_cpu PUBLIC simulator cpu_loaders)
    target_link_libraries(simulator_cpu PRIVATE kernels)
    set_property(TARGET simulator_cpu PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
