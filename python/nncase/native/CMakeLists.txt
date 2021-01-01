cmake_minimum_required (VERSION 3.8)

find_package (Python3 COMPONENTS Interpreter Development REQUIRED)

set(SRCS ffi.cpp)

pybind11_add_module (nncase_python ${SRCS})
target_link_libraries(nncase_python PRIVATE nncase CONAN_PKG::pybind11)
set_target_properties(nncase_python PROPERTIES
                                 OUTPUT_NAME _nncase)