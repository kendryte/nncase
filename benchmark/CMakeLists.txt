cmake_minimum_required (VERSION 3.8)

add_subdirectory(models)

set(SRCS cli.cpp)

add_executable (benchnncase ${SRCS})
target_link_libraries(benchnncase PRIVATE nncaseruntime bench_models)
install(TARGETS benchnncase
        COMPONENT nncase-tools)

if(MSVC)
    target_link_libraries(benchnncase PRIVATE bench_models_rc)
endif()

if(ENABLE_K210_RUNTIME)
    target_link_libraries(benchnncase PRIVATE nncase_rt_modules_k210)
    target_link_kendryte(benchnncase)
endif()
