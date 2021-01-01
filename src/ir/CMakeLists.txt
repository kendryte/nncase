cmake_minimum_required (VERSION 3.8)

set(SRCS connectors.cpp
         node.cpp
         graph.cpp
         #debug.cpp
         visitor.cpp)

add_library(ir OBJECT ${SRCS})
target_link_libraries(ir PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::xtensor CONAN_PKG::magic_enum)
target_compile_definitions(ir PUBLIC -DNNCASE_DLL)
set_property(TARGET ir PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(ops)