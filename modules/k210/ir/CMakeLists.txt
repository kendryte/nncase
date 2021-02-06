cmake_minimum_required (VERSION 3.8)

set(SRCS ops/fake_kpu_conv2d.cpp
         ops/fake_piecewise_linear.cpp
         ops/kpu_conv2d.cpp
         ops/kpu_data_exchange.cpp)

add_library(ir_k210 OBJECT ${SRCS})
target_link_libraries(ir_k210 PUBLIC nncase)
target_compile_definitions(ir_k210 PUBLIC -DNNCASE_MODULES_K210_DLL)
set_target_properties(ir_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)