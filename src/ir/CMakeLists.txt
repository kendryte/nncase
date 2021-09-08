cmake_minimum_required (VERSION 3.8)

#set(SRCS connectors.cpp
#         node.cpp
#         graph.cpp
#         graph.partition.cpp
#         debug.cpp
#         placeholders.cpp
#         visitor.cpp)

set(SRCS var.cpp
		 object.cpp
         expr.cpp
		 constant.cpp
		 debug.cpp
		 function.cpp
		 call.cpp
		 op.cpp
		 type.cpp
		 shape.cpp
		 module.cpp)

if(MSVC)
	list(APPEND SRCS ir_types.natvis)
endif()

add_library(ir OBJECT ${SRCS})
target_link_libraries(ir PUBLIC gsl::gsl-lite xtensor::xtensor magic_enum::magic_enum)
target_link_libraries(ir PRIVATE onnx_proto mpark_variant::mpark_variant)
target_compile_definitions(ir PUBLIC -DNNCASE_DLL)
set_property(TARGET ir PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(math)
add_subdirectory(tensors)
add_subdirectory(nn)
