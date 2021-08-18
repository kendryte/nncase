cmake_minimum_required (VERSION 3.8)

set(SRCS scheduler.cpp
         freelist.cpp
         buffer_allocator.cpp
         liveness_analysis.cpp
         function_schedule_context.cpp
         module_schedule_context.cpp
         model_schedule_context.cpp)

add_library(schedule OBJECT ${SRCS})
target_link_libraries(schedule PUBLIC ir)
target_link_libraries(schedule PRIVATE fmt::fmt)
set_target_properties(schedule PROPERTIES POSITION_INDEPENDENT_CODE ON)
