cmake_minimum_required (VERSION 3.8)

add_subdirectory(models)

set(SRCS cli.cpp)

add_executable (benchnncase EXCLUDE_FROM_ALL ${SRCS})
target_link_libraries(benchnncase PRIVATE nncaseruntime bench_models)
install(TARGETS benchnncase)
