cmake_minimum_required (VERSION 3.13)

set(SRCS runtime_module.cpp
         op_reader.cpp
         evaluate_stack.cpp
         ops/ctrl.cpp)

if (BUILDING_RUNTIME)
    target_sources(runtime PRIVATE ${SRCS})

    if (BUILD_SHARED_LIBS)
        target_sources(runtime_shared PRIVATE ${SRCS})
    endif()
else()
    add_library(simulator_stackvm OBJECT ${SRCS})
    target_link_libraries(simulator_stackvm PUBLIC simulator)
endif()
