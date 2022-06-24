cmake_minimum_required(VERSION 3.13)

set(SRCS ../tensor_ops.cpp
         activation.cpp
         batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         cast.cpp
         compare.cpp
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
         split.cpp
         slice.cpp
         softmax.cpp
         stack.cpp
         transpose.cpp
         unary.cpp)
target_sources(kernels PRIVATE ${SRCS})