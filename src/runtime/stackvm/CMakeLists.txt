cmake_minimum_required (VERSION 3.13)

set(SRCS runtime_module.cpp
         op_reader.cpp
         evaluate_stack.cpp
         ops/control.cpp
         ops/loadstore.cpp
         ops/stack.cpp
         ops/scalar.cpp
         ops/conversion.cpp
         ops/tensor.batch_to_space.cpp
         ops/tensor.binary.cpp
         ops/tensor.broadcast.cpp
         ops/tensor.call.cpp
         ops/tensor.conv2d.cpp
         ops/tensor.dequantize.cpp
         ops/tensor.pad.cpp
         ops/tensor.quantize.cpp
         ops/tensor.reduce.cpp
         ops/tensor.reduce_window2d.cpp
         ops/tensor.slice.cpp
         ops/tensor.transpose.cpp
         ops/tensor.unary.cpp)

if (BUILDING_RUNTIME)
    target_sources(runtime PRIVATE ${SRCS})
    target_link_libraries(runtime PRIVATE kernels)

    if (BUILD_SHARED_LIBS)
        target_sources(runtime_shared PRIVATE ${SRCS})
        target_link_libraries(runtime_shared PRIVATE kernels)
    endif()
else()
    add_library(simulator_stackvm OBJECT ${SRCS})
    target_link_libraries(simulator_stackvm PUBLIC simulator)
    target_link_libraries(simulator_stackvm PRIVATE kernels)
    set_property(TARGET simulator_stackvm PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
