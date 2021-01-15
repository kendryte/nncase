cmake_minimum_required (VERSION 3.8)

set(SRCS runtime_module.cpp
         op_reader.cpp)

add_library(simulator_stackvm OBJECT ${SRCS})
target_link_libraries(simulator_stackvm PUBLIC simulator)
