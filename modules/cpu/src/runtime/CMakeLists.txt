cmake_minimum_required (VERSION 3.13)

add_subdirectory(cmodel)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         thread_pool.cpp
         elfload.cpp
         elfloader.cpp
         elfreloc_aarch64.cpp
         elfreloc_amd64.cpp
         elfreloc_i386.cpp
         elfreloc_riscv64.cpp)

if (BUILDING_RUNTIME)
    if (ENABLE_CPU_RUNTIME)
        add_library(runtime_cpu OBJECT ${SRCS})
        target_link_libraries(runtime_cpu PUBLIC nncaseruntime pthread)
        set_target_properties(runtime_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
        install(TARGETS runtime_cpu EXPORT nncaseruntimeTargets)
    endif()
else()
    add_library(simulator_cpu OBJECT ${SRCS})
    target_link_libraries(simulator_cpu PUBLIC nncasebase nncaseruntime pthread)
    target_compile_definitions(simulator_cpu PUBLIC -DNNCASE_MODULES_CPU_DLL -DNNCASE_SIMULATOR)
    set_target_properties(simulator_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
