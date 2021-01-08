cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         error.cpp
         runtime_loader.cpp)

add_library(simulator OBJECT ${SRCS})
target_include_directories(simulator PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(simulator PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant)
target_link_libraries(simulator PRIVATE CONAN_PKG::fmt)
target_compile_definitions(simulator PUBLIC -DNNCASE_DLL)
set_target_properties(simulator PROPERTIES CXX_STANDARD 11)
