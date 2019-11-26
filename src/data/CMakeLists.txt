cmake_minimum_required (VERSION 3.8)

set(SRCS dataset.cpp)

add_library(data STATIC ${SRCS})
target_include_directories(data PUBLIC include)
target_include_directories(data PUBLIC ${THIRD_PARTY}/xtl/include ${THIRD_PARTY}/xtensor/include)
target_link_libraries(data PUBLIC common CONAN_PKG::boost)
target_link_libraries(data PRIVATE CONAN_PKG::opencv)