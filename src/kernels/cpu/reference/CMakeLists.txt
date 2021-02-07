cmake_minimum_required (VERSION 3.13)

set(SRCS batch_to_space.cpp
         binary.cpp
         broadcast.cpp
         convolution.cpp
         copy.cpp
         nnil.cpp
         reduce.cpp
         transpose.cpp
         unary.cpp)
target_sources(kernels PRIVATE ${SRCS})
