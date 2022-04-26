cmake_minimum_required (VERSION 3.18)

set(SRCS interpreter.cpp)

add_library(nncase_csharp SHARED ${SRCS})
target_link_libraries(nncase_csharp PRIVATE simulator)
target_include_directories(nncase_csharp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS nncase_csharp
        COMPONENT nncase-runtime
        DESTINATION lib)