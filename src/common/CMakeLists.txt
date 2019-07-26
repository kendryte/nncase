cmake_minimum_required (VERSION 3.8)

add_library(common INTERFACE)
target_include_directories(common INTERFACE include)
target_include_directories(common INTERFACE ${THIRD_PARTY}/xtl/include)