cmake_minimum_required (VERSION 3.13)

set(SRCS batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         concat.cpp
         convolution.cpp
         convert.cpp
         copy.cpp
         dequantize.cpp
         lut1d.cpp
         nnil.cpp
         pad.cpp
         quantize.cpp
         reduce.cpp
         reduce_window.cpp
         resize_image.cpp
         transpose.cpp
         slice.cpp
         unary.cpp)
target_sources(kernels PRIVATE ${SRCS})
