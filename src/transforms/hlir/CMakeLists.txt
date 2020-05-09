cmake_minimum_required (VERSION 3.8)

set(SRCS transform.cpp
         pass.cpp
         neutral/add_quant_checkpoints.cpp
         neutral/fold_constant.cpp
         neutral/fold_reshape.cpp
         neutral/fold_transpose.cpp
         neutral/fold_pad.cpp
         neutral/fold_quantize.cpp
         neutral/fuse_pad.cpp
         neutral/fuse_clamp.cpp
         neutral/fuse_unary.cpp
         neutral/fused_unary_to_lookup1d.cpp
         neutral/transpose_motion.cpp
         neutral/dequantize_motion.cpp
         neutral/quantized_conv2d.cpp
         neutral/quantized_matmul.cpp
         neutral/quantized_binary.cpp
         neutral/eliminate_dilated_conv2d.cpp
         neutral/simplify_reduce.cpp
         k210/piecewise_regression.cpp
         k210/fake_kpu_conv2d.cpp
         k210/strided_slice_motion.cpp
         k210/kpu_conv2d.cpp
         k210/fuse_kpu_download.cpp
         k210/fold_kpu_upload.cpp
         k210/matmul_to_fake_kpu_conv2d.cpp
         k210/fuse_kpu_conv2d_pool.cpp)

add_library(hlir_transforms STATIC ${SRCS})
target_include_directories(hlir_transforms PUBLIC ../include)
target_link_libraries(hlir_transforms PUBLIC hlir evaluator)
target_link_libraries(hlir_transforms PRIVATE CONAN_PKG::range-v3)