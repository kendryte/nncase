cmake_minimum_required (VERSION 3.8)

set(SRCS connectors.cpp
         node.cpp
         graph.cpp
         graph.partition.cpp
         debug.cpp
         placeholders.cpp
         visitor.cpp)

add_library(ir OBJECT ${SRCS})
target_link_libraries(ir PUBLIC gsl::gsl-lite xtensor::xtensor magic_enum::magic_enum)
target_link_libraries(ir PRIVATE onnx_proto mpark_variant::mpark_variant)
target_compile_definitions(ir PUBLIC -DNNCASE_DLL)
set_property(TARGET ir PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(ops)