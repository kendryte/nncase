cmake_minimum_required(VERSION 3.8)

set(SRCS module_builder.cpp
        op_writer.cpp
        ops/batch_to_space.cpp
        ops/binary.cpp
        ops/broadcast.cpp
        ops/call.cpp
        ops/compare.cpp
        ops/compress.cpp
        ops/conv2d.cpp
        ops/convert.cpp
        ops/copy.cpp
        ops/cumsum.cpp
        ops/dequantize.cpp
        ops/gather.cpp
        ops/gather_elements.cpp
        ops/gather_nd.cpp
        ops/gru.cpp
        ops/hardmax.cpp
        ops/matmul.cpp
        ops/onehot.cpp
        ops/pad.cpp
        ops/quantize.cpp
        ops/random_normal.cpp
        ops/random_uniform.cpp
        ops/reduce.cpp
        ops/reduce_arg.cpp
        ops/reduce_prod.cpp
        ops/reduce_window2d.cpp
        ops/resize_image.cpp
        ops/roi_align.cpp
        ops/slice.cpp
        ops/sigmoid.cpp
        ops/softmax.cpp
        ops/space_to_batch.cpp
        ops/table_lookup1d.cpp
        ops/ternary.cpp
        ops/topk.cpp
        ops/transpose.cpp
        ops/trilu.cpp
        ops/tflite_detection_postprocess.cpp
        ops/unary.cpp
        ops/layernorm.cpp
        ops/instancenorm.cpp)

add_library(codegen_stackvm OBJECT ${SRCS})
target_link_libraries(codegen_stackvm PUBLIC ir schedule)
target_link_libraries(codegen_stackvm PRIVATE evaluator mpark_variant::mpark_variant)
target_compile_definitions(codegen_stackvm PUBLIC -DNNCASE_DLL)
set_target_properties(codegen_stackvm PROPERTIES POSITION_INDEPENDENT_CODE ON)