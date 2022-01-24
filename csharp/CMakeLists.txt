cmake_minimum_required (VERSION 3.18)

set(SRCS interpreter.cpp)

add_library(nncaseruntime_csharp SHARED ${SRCS})
target_link_libraries(nncaseruntime_csharp PRIVATE nncaseruntime)
target_include_directories(nncaseruntime_csharp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# set_target_properties(nncaseruntime_csharp PROPERTIES OUTPUT_NAME _nncaseruntime_csharp)
install(TARGETS nncaseruntime_csharp DESTINATION lib)
