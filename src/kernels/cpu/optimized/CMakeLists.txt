cmake_minimum_required (VERSION 3.13)

set(SRCS convolution.cpp
         concat.cpp
         slice.cpp
         copy.cpp
         resize_image.cpp
         gather.cpp
         gather_nd.cpp
         onehot.cpp)
target_sources(kernels PRIVATE ${SRCS})