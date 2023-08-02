enable_testing()

find_package(ortki)

include_directories(${CMAKE_SOURCE_DIR}/third_party)

macro(add_test_exec name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE GTest::gtest_main nncaseruntime ortki::ortki)
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -DTEST_EXECUTABLE=$<TARGET_FILE:${name}> -P ${CMAKE_CURRENT_SOURCE_DIR}/../../toolchains/run_test.cmake)
endmacro()

file(GLOB TEST_NAMES CONFIGURE_DEPENDS test_*.cpp)
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    add_test_exec(${tname})
endforeach()
