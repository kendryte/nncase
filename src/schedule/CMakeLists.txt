cmake_minimum_required (VERSION 3.8)

set(SRCS scheduler.cpp
         freelist.cpp
         buffer_allocator.cpp)

add_library(schedule OBJECT ${SRCS})
target_link_libraries(schedule PUBLIC ir)
set_target_properties(schedule PROPERTIES POSITION_INDEPENDENT_CODE ON)
