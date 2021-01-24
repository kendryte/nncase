cmake_minimum_required (VERSION 3.8)

set(SRCS cli.cpp
         compile.cpp
         inference.cpp)

add_executable (ncc ${SRCS})
target_link_libraries(ncc PRIVATE nncase CONAN_PKG::lyra)
