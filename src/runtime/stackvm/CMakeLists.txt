cmake_minimum_required (VERSION 3.13)

set(SRCS runtime_module.cpp
         runtime_function.cpp
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
         ops/tensor.convert.cpp
         ops/tensor.copy.cpp
         ops/tensor.dequantize.cpp
         ops/tensor.gather.cpp
         ops/tensor.gather_nd.cpp
         ops/tensor.lut1d.cpp
         ops/tensor.onehot.cpp
         ops/tensor.pad.cpp
         ops/tensor.quantize.cpp
         ops/tensor.reduce.cpp
         ops/tensor.reduce_arg.cpp
         ops/tensor.reduce_window2d.cpp
         ops/tensor.resize_image.cpp
         ops/tensor.slice.cpp
         ops/tensor.transpose.cpp
         ops/tensor.unary.cpp)

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
