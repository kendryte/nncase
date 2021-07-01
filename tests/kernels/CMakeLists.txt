enable_testing()

macro(add_test_exec name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
    GTest::gtest_main nncase)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

set(CMAKE_CXX_STANDARD 17)

file(GLOB TEST_NAMES CONFIGURE_DEPENDS test_*.cpp)

foreach(test_name ${TEST_NAMES}) 
    get_filename_component(tname ${test_name} NAME_WE)
    add_test_exec(${tname})
endforeach()
