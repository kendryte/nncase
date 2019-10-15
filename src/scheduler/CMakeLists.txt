cmake_minimum_required (VERSION 3.8)

set(SRCS scheduler.cpp
         freelist.cpp
         memory_allocator.cpp
         k210/kpu_memory_allocator.cpp)

add_library(scheduler STATIC ${SRCS})
target_include_directories(scheduler PUBLIC include)
target_link_libraries(scheduler PUBLIC ir)
target_link_libraries(scheduler PRIVATE CONAN_PKG::range-v3 CONAN_PKG::ortools)