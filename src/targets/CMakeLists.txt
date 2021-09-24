cmake_minimum_required (VERSION 3.8)

set(SRCS neutral_target.cpp
         target.cpp)

add_library(targets OBJECT ${SRCS})
#target_link_libraries(targets PUBLIC ir schedule transforms codegen codegen_stackvm)
target_link_libraries(targets PUBLIC ir transforms)
set_target_properties(targets PROPERTIES POSITION_INDEPENDENT_CODE ON)
