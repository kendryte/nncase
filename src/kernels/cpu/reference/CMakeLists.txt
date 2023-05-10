cmake_minimum_required (VERSION 3.13)

set(SRCS batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         compare.cpp
         compress.cpp
         concat.cpp
         convolution.cpp
         convert.cpp
         copy.cpp
         cumsum.cpp
         dequantize.cpp
         gather.cpp
         gather_elements.cpp
         gather_nd.cpp
         gru.cpp
         hardmax.cpp
         lut1d.cpp
         matmul.cpp
         nnil.cpp
         onehot.cpp
         pad.cpp
         quantize.cpp
         random.cpp
         reduce.cpp
         reduce_arg.cpp
         reduce_prod.cpp
         reduce_window.cpp
         resize_image.cpp
         roi_align.cpp
         sigmoid.cpp
         softmax.cpp
         slice.cpp
         space_to_batch.cpp
         ternary.cpp
         topk.cpp
         transpose.cpp
         trilu.cpp
         tflite_detection_postprocess.cpp
         unary.cpp
         layernorm.cpp
         instancenorm.cpp)
target_sources(kernels PRIVATE ${SRCS})
