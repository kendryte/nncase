cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         kernel_registry.cpp)

add_library(runtime STATIC ${SRCS})
target_link_libraries(runtime PUBLIC common)