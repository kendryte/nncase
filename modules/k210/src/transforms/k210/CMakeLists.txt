cmake_minimum_required (VERSION 3.8)

set(SRCS piecewise_regression.cpp
         fake_kpu_conv2d.cpp
         strided_slice_motion.cpp
         kpu_conv2d.cpp
         fuse_kpu_download.cpp
         fold_kpu_upload.cpp
         fuse_kpu_conv2d_pool.cpp
         fused_unary_motion.cpp)

add_library(transforms_k210 OBJECT ${SRCS})
target_link_libraries(transforms_k210 PUBLIC nncase)
target_compile_definitions(transforms_k210 PUBLIC -DNNCASE_MODULES_K210_DLL)
set_target_properties(transforms_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)
