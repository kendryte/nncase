cmake_minimum_required (VERSION 3.13)

add_subdirectory(runtime/stackvm)

set(SRCS object.cpp
		 shape.cpp
		 tensor.cpp
		 type.cpp
		 value.cpp
		 runtime/buffer.cpp
		 runtime/datatypes.cpp
		 runtime/error.cpp
		 runtime/host_allocator.cpp
		 runtime/host_buffer.cpp
		 runtime/host_runtime_tensor.cpp
	     runtime/interpreter.cpp
	     runtime/runtime_loader.cpp
	     runtime/runtime_module.cpp
		 runtime/runtime_function.cpp
		 runtime/section.cpp
		 runtime/type_serializer.cpp
		 runtime/runtime_tensor.cpp)

add_library(nncaseruntime STATIC ${SRCS})
target_link_libraries(nncaseruntime PRIVATE fmt::fmt)
target_compile_definitions(nncaseruntime PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
if (DEFAULT_BUILTIN_RUNTIMES)
    target_compile_definitions(nncaseruntime PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
endif ()
set_property(TARGET nncaseruntime PROPERTY POSITION_INDEPENDENT_CODE ON)