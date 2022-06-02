cmake_minimum_required(VERSION 3.13)

set(SRCS dummy_tensor_ops.cpp
         batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         cast.cpp
         concat.cpp
         convolution.cpp
         cumsum.cpp
         dequantize.cpp
         gather.cpp
         gather_nd.cpp
         hardmax.cpp
         matmul.cpp
         onehot.cpp
         pad.cpp
         quantize.cpp
         reduce.cpp
         reduce_arg.cpp
         reduce_window.cpp
         resize_image.cpp
         sigmoid.cpp
         slice.cpp
         transpose.cpp
         unary.cpp)
target_sources(kernels PRIVATE ${SRCS})