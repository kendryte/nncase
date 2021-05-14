macro(add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
  CONAN_PKG::gtest nncase)
endmacro()

set(CMAKE_CXX_STANDARD 17)

file(GLOB TEST_NAMES CONFIGURE_DEPENDS test_*.cpp)

foreach(test_name ${TEST_NAMES}) 
    get_filename_component(tname ${test_name} NAME_WE)
    add_test(${tname})
endforeach()