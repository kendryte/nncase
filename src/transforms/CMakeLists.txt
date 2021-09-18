cmake_minimum_required (VERSION 3.16)

set(SRCS egraph.cpp)

add_library(transforms OBJECT ${SRCS})
#target_link_libraries(transforms PUBLIC ir evaluator)
target_link_libraries(transforms PUBLIC ir)
set_target_properties(transforms PROPERTIES POSITION_INDEPENDENT_CODE ON)

#add_subdirectory(neutral)
