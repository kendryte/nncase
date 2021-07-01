cmake_minimum_required (VERSION 3.15)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

set(SRCS ffi.cpp)

pybind11_add_module(nncaseruntime_python ${SRCS})
target_link_libraries(nncaseruntime_python PRIVATE nncaseruntime)
target_include_directories(nncaseruntime_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
set_target_properties(nncaseruntime_python PROPERTIES
                                 OUTPUT_NAME _nncaseruntime)
install(TARGETS nncaseruntime_python DESTINATION lib)
