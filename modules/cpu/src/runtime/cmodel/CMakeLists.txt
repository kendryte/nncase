
add_library(cpu_cmodel STATIC
  src/hardware_context.cpp
)

target_include_directories(cpu_cmodel PUBLIC include)
set_target_properties(cpu_cmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)
        
add_executable(cpu_cmodel_cli src/cpu_cmodel.cpp ../shared_memory.cpp)
target_link_libraries(cpu_cmodel_cli PUBLIC cpu_cmodel)
set_target_properties(cpu_cmodel_cli PROPERTIES POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "nncase.simulator.cpu.c")
install(TARGETS cpu_cmodel_cli COMPONENT nncase-runtime)


function(add_test name test_path)
  add_executable(${name} "${test_path}/main.cpp")
  target_link_libraries(${name} cpu_cmodel)
endfunction(add_test)

add_test(demo1 tests/demo1)
