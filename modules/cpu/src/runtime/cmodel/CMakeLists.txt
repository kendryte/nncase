project(cpu)
cmake_minimum_required(VERSION 3.13)

add_library(cpu_cmodel STATIC
  src/dummy.cpp
  ../hardware_context.cpp
)

set(CMAKE_CXX_STANDARD 20)

target_include_directories(cpu_cmodel PUBLIC include)
target_include_directories(cpu_cmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src/Native/include)
target_include_directories(cpu_cmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_include_directories(cpu_cmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime)
target_include_directories(cpu_cmodel PUBLIC /root/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/)
# target_link_libraries(cpu_cmodel PUBLIC gsl::gsl-lite)
# if(NOT APPLE AND NOT MSVC)
#   target_link_libraries(cpu_cmodel PRIVATE rt)
# endif()
set_target_properties(cpu_cmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_executable(cpu_cmodel_cli src/cpu_cmodel.cpp ../shared_memory.cpp)
# target_link_libraries(cpu_cmodel_cli PUBLIC cpu_cmodel)
# set_target_properties(cpu_cmodel_cli PROPERTIES POSITION_INDEPENDENT_CODE ON
#   OUTPUT_NAME "nncase.simulator.cpu.c")
# install(TARGETS cpu_cmodel_cli COMPONENT nncase-runtime)


function(add_test name test_path)
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O1)
  endif()
  add_link_options(-no-pie -nostartfiles -fPIC -fno-stack-protector -static -Wl,-e,_Z6_startPN6nncase7runtime3cpu19hardware_context_mtEPNS1_15runtime_util_mtEPNS1_19nncase_method_tableEPPh)
  add_executable(${name} "${test_path}/main.cpp")
  target_link_libraries(${name} cpu_cmodel)
endfunction(add_test)

# add_test(demo1 tests/demo1)
# add_test(demo2 tests/demo2)
# add_test(demo3 tests/demo3)
add_test(demo4 tests/demo4)
add_test(norm tests/norm)
# add_test(embed tests/embed)
add_test(head tests/head)
