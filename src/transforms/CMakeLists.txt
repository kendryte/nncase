cmake_minimum_required (VERSION 3.8)

set(SRCS transform.cpp
         pass.cpp)

add_library(transforms OBJECT ${SRCS})
target_include_directories(transforms PUBLIC include)
target_link_libraries(transforms PUBLIC ir evaluator)
set_target_properties(transforms PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(neutral)
