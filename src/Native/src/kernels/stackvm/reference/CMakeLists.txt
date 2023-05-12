cmake_minimum_required(VERSION 3.13)

set(SRCS ../tensor_ops.cpp
         activation.cpp
         batchnorm.cpp
         batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         cast.cpp
         clamp.cpp
         compare.cpp
         concat.cpp
         constant_of_shape.cpp
         convolution.cpp
         cumsum.cpp
         dequantize.cpp
         expand.cpp
         gather.cpp
         gather_nd.cpp
         hardmax.cpp
         instance_norm.cpp
         lrn.cpp
         lstm.cpp
         matmul.cpp
         onehot.cpp
         pad.cpp
         prelu.cpp
         quantize.cpp
         random.cpp
         reduce.cpp
         reduce_arg.cpp
         reduce_window.cpp
         resize_image.cpp
         reverse_sequence.cpp
         split.cpp
         slice.cpp
         softmax.cpp
         space_to_batch.cpp
         stack.cpp
         tile.cpp
         topk.cpp
         transpose.cpp
         unary.cpp
         where.cpp
         uninitialized.cpp
         layer_norm.cpp
         scatter_nd.cpp)
target_sources(kernels PRIVATE ${SRCS})