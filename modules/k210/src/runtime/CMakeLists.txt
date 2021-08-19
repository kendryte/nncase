cmake_minimum_required (VERSION 3.13)

add_subdirectory(kendryte)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         op_reader.cpp
         error.cpp
         ops/kpu_conv2d.cpp
         ops/kpu_download.cpp
         ops/kpu_upload.cpp
         ops/copy.cpp)

if (BUILDING_RUNTIME)
    if (ENABLE_K210_RUNTIME)
        if (NOT K210_SDK_DIR)
            message(FATAL_ERROR "Please set K210_SDK_DIR to kendryte sdk root path")
        endif()

        list(APPEND SRCS shared_runtime_tensor.platform.cpp)
        add_library(runtime_k210 OBJECT ${SRCS})
        if (NOT MSVC)
            target_compile_options(runtime_k210 PRIVATE -Wno-pedantic)
        endif()
        target_link_libraries(runtime_k210 PUBLIC nncaseruntime)
        target_link_libraries(runtime_k210 PRIVATE kernels_k210)
        target_link_kendryte(runtime_k210)
        set_target_properties(runtime_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)
        install(TARGETS runtime_k210 EXPORT nncaseruntimeTargets)
    endif()
else()
    add_library(simulator_k210 OBJECT ${SRCS})
    target_link_libraries(simulator_k210 PUBLIC nncase)
    target_compile_definitions(simulator_k210 PUBLIC -DNNCASE_MODULES_K210_DLL -DNNCASE_SIMULATOR)
    set_target_properties(simulator_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
