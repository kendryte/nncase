add_library(cpu_cmodel STATIC
  src/hardware_context.cpp
)

target_include_directories(cpu_cmodel PUBLIC include)
target_link_libraries(cpu_cmodel PUBLIC fmt::fmt spdlog::spdlog)
if(NOT APPLE AND NOT MSVC)
  target_link_libraries(cpu_cmodel PRIVATE rt)
endif()
set_target_properties(cpu_cmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)
        
add_executable(cpu_cmodel_cli src/cpu_cmodel.cpp ../shared_memory.cpp)
target_link_libraries(cpu_cmodel_cli PUBLIC cpu_cmodel)
set_target_properties(cpu_cmodel_cli PROPERTIES POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "nncase.simulator.cpu.c")
install(TARGETS cpu_cmodel_cli COMPONENT nncase-runtime)


function(add_test name test_path)
  add_executable(${name} "${test_path}/main.cpp")
  target_link_libraries(${name} cpu_cmodel spdlog::spdlog)
endfunction(add_test)

add_test(demo1 tests/demo1)
add_test(demo2 tests/demo2)
add_test(demo3 tests/demo3)
add_test(norm tests/norm)
add_test(embed tests/embed)
add_test(head tests/head)
