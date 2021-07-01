cmake_minimum_required (VERSION 3.8)

set(SRCS cli.cpp
         compile.cpp
         inference.cpp)

add_executable (ncc ${SRCS})
target_link_libraries(ncc PRIVATE nncase CONAN_PKG::lyra)
if (APPLE)
    set_target_properties(evaluator PROPERTIES INSTALL_RPATH "@loader_path/../lib")
else()
    set_target_properties(evaluator PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
endif()
install(TARGETS ncc)
