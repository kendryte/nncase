cmake_minimum_required (VERSION 3.18)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

set(SRCS ffi.cpp)

pybind11_add_module(nncase_python ${SRCS})
target_link_libraries(nncase_python PRIVATE nncaseruntime)
target_include_directories(nncase_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
set_target_properties(nncase_python PROPERTIES
                                    OUTPUT_NAME _nncase)
install(TARGETS nncase_python DESTINATION lib)
