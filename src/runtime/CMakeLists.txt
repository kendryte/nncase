cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         error.cpp)

add_library(simulator OBJECT ${SRCS})
target_link_libraries(simulator PUBLIC CONAN_PKG::gsl-lite)
target_compile_definitions(simulator PUBLIC -DNNCASE_DLL)
set_target_properties(simulator PROPERTIES CXX_STANDARD 11)
