﻿cmake_minimum_required (VERSION 3.13)

set(SRCS kpu_upload.cpp)

if (BUILDING_RUNTIME)
    if (ENABLE_K210_RUNTIME)
        add_library(kernels_k210 OBJECT ${SRCS})
        if (NOT MSVC)
            target_compile_options(kernels_k210 PRIVATE -Wno-pedantic)
        endif()
        target_link_libraries(kernels_k210 PUBLIC nncaseruntime)
        target_link_kendryte(kernels_k210)
        set_target_properties(kernels_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)
        install(TARGETS kernels_k210 EXPORT nncaseruntimeTargets)
    endif()
else()
    add_library(kernels_k210 OBJECT ${SRCS})
    target_link_libraries(kernels_k210 PUBLIC nncase)
    target_compile_definitions(kernels_k210 PUBLIC -DNNCASE_MODULES_K210_DLL -DNNCASE_SIMULATOR)
    set_target_properties(kernels_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
