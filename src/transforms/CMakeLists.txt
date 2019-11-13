cmake_minimum_required (VERSION 3.8)

set(SRCS transform.cpp
         neutral/add_quant_checkpoints.cpp
         neutral/fold_constant.cpp
         neutral/fold_reshape.cpp
         neutral/fold_transpose.cpp
         neutral/fold_pad.cpp
         neutral/fold_quantize.cpp
         neutral/fuse_pad.cpp
         neutral/transpose_motion.cpp
         neutral/dequantize_motion.cpp
         neutral/quantized_conv2d.cpp
         neutral/quantized_matmul.cpp
         neutral/quantized_binary.cpp
         k210/fake_kpu_conv2d.cpp
         k210/strided_slice_motion.cpp
         k210/kpu_conv2d.cpp
         k210/fuse_kpu_download.cpp
         k210/fold_kpu_upload.cpp
         k210/fake_piecewise_linear.cpp)

add_library(transforms STATIC ${SRCS})
target_include_directories(transforms PUBLIC include)
target_link_libraries(transforms PUBLIC ir evaluator)
target_link_libraries(transforms PRIVATE CONAN_PKG::range-v3)