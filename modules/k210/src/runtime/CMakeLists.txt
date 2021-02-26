cmake_minimum_required (VERSION 3.13)

set(SRCS runtime_module.cpp
         op_reader.cpp
         error.cpp
         ops/kpu_conv2d.cpp
         ops/kpu_download.cpp
         ops/kpu_upload.cpp)

if (BUILDING_RUNTIME)
    target_sources(runtime PRIVATE ${SRCS})
    target_link_libraries(runtime PRIVATE kernels)
else()
    add_library(simulator_k210 OBJECT ${SRCS})
    target_link_libraries(simulator_k210 PUBLIC nncase)
    target_compile_definitions(simulator_k210 PUBLIC -DNNCASE_MODULES_K210_DLL -DNNCASE_SIMULATOR)
    set_target_properties(simulator_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
