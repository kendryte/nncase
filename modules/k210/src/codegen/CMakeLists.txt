cmake_minimum_required (VERSION 3.8)

set(SRCS module_builder.cpp
         ops/kpu_conv2d.cpp
         ops/kpu_download.cpp
         ops/kpu_upload.cpp
         ops/copy.cpp)

add_library(codegen_k210 OBJECT ${SRCS})
target_link_libraries(codegen_k210 PUBLIC nncase)
target_compile_definitions(codegen_k210 PUBLIC -DNNCASE_MODULES_K210_DLL)
set_target_properties(codegen_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)