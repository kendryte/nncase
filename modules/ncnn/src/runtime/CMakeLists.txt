cmake_minimum_required (VERSION 3.13)

set(SRCS runtime_module.cpp
         runtime_function.cpp)

if (BUILDING_RUNTIME)
    add_library(runtime_ncnn OBJECT ${SRCS})
    if (NOT MSVC)
        target_compile_options(runtime_ncnn PRIVATE -Wno-unused-parameter)
    endif()

    target_link_libraries(runtime_ncnn PUBLIC nncaseruntime ncnn::ncnn)
    set_target_properties(runtime_ncnn PROPERTIES POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime_ncnn EXPORT nncase_rt_modules_ncnnTargets)
else()
    add_library(simulator_ncnn OBJECT ${SRCS})
    if (NOT MSVC)
        target_compile_options(simulator_ncnn PRIVATE -Wno-unused-parameter)
    endif()
    target_link_libraries(simulator_ncnn PRIVATE nncaseruntime ncnn::ncnn)
    target_compile_definitions(simulator_ncnn PUBLIC -DNNCASE_MODULES_NCNN_DLL -DNNCASE_SIMULATOR)
    set_target_properties(simulator_ncnn PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
