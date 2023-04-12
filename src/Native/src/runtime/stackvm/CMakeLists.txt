cmake_minimum_required (VERSION 3.13)

set(SRCS runtime_module.cpp
         runtime_function.cpp
         runtime_function.run.cpp
         op_profile.cpp
         op_reader.cpp
         call_frame.cpp
         evaluate_stack.cpp
         ops/control.cpp
         ops/tensor.cpp)

if (BUILDING_RUNTIME)
    add_library(runtime_stackvm OBJECT ${SRCS})
    target_link_libraries(runtime_stackvm PUBLIC runtime)
    target_link_libraries(runtime_stackvm PRIVATE kernels)
    set_property(TARGET runtime_stackvm PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime_stackvm EXPORT nncaseruntimeTargets)
else()
    add_library(simulator_stackvm OBJECT ${SRCS})
    target_link_libraries(simulator_stackvm PUBLIC simulator)
    target_link_libraries(simulator_stackvm PRIVATE kernels)
    set_property(TARGET simulator_stackvm PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
