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
         elfreloc_riscv64.cpp
         hardware_context.cpp
         shared_memory.cpp)

if (BUILDING_RUNTIME)
    if (ENABLE_XPU_RUNTIME)
        add_library(runtime_xpu OBJECT ${SRCS})
        target_link_libraries(runtime_xpu PUBLIC nncaseruntime pthread rt)
        # set_target_properties(runtime_xpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
        install(TARGETS runtime_xpu EXPORT nncaseruntimeTargets)
    endif()
else()
    add_library(simulator_xpu OBJECT ${SRCS})
    target_link_libraries(simulator_xpu PUBLIC nncasebase nncaseruntime pthread)
    target_compile_definitions(simulator_xpu PUBLIC -DNNCASE_MODULES_XPU_DLL -DNNCASE_SIMULATOR)
    set_target_properties(simulator_xpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
