cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         kernel_registry.cpp
         neutral/neutral_ops.cpp
         cpu/cpu_ops.cpp
         k210/interpreter.cpp
         k210/k210_ops.cpp)

add_library(runtime STATIC ${SRCS})
target_link_libraries(runtime PUBLIC common)