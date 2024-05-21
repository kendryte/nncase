
include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

file(GLOB TEST_NAMES CONFIGURE_DEPENDS benchmark_*.cpp)
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} PRIVATE nncaseruntime)
endforeach()