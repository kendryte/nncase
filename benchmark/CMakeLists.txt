cmake_minimum_required (VERSION 3.8)

add_subdirectory(models)

set(SRCS cli.cpp)

add_executable (benchnncase ${SRCS})
target_link_libraries(benchnncase PRIVATE nncaseruntime bench_models)
if(MSVC)
    target_link_libraries(benchnncase PRIVATE bench_models_rc)
endif()
install(TARGETS benchnncase)
