cmake_minimum_required (VERSION 3.13)

set(SRCS batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         concat.cpp
         convolution.cpp
         convert.cpp
         copy.cpp
         cumsum.cpp
         dequantize.cpp
         gather.cpp
         gather_nd.cpp
         hardmax.cpp
         lut1d.cpp
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
         sigmoid.cpp
         slice.cpp
         ternary.cpp
         topk.cpp
         transpose.cpp
         trilu.cpp
         unary.cpp)
target_sources(kernels PRIVATE ${SRCS})
