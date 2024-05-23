
include_directories(${CMAKE_CURRENT_LIST_DIR}/..)

file(GLOB TEST_NAMES CONFIGURE_DEPENDS benchmark_*.cpp)
foreach(test_name ${TEST_NAMES})
    get_filename_component(tname ${test_name} NAME_WE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
        add_compile_options(-march=rv64gvzfh)
    endif()    
    add_executable(${tname} ${tname}.cpp)
    target_link_libraries(${tname} PRIVATE nncaseruntime)
endforeach()