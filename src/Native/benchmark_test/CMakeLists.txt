enable_testing()

file(GLOB TEST_NAMES CONFIGURE_DEPENDS benchmark_*.cpp)
# file(GLOB TEST_NAMES CONFIGURE_DEPENDS benchmark_*_u.cpp)
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} PRIVATE nncaseruntime)
endforeach()