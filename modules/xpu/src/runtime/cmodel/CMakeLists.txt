project(xpu)
cmake_minimum_required(VERSION 3.13)

add_library(xpu_cmodel STATIC
  src/dummy.cpp
  ../hardware_context.cpp
)

set(CMAKE_CXX_STANDARD 20)

target_include_directories(xpu_cmodel PUBLIC include)
# target_include_directories(xpu_cmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src/Native/include)
target_include_directories(xpu_cmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
target_include_directories(xpu_cmodel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../runtime)
# target_include_directories(xpu_cmodel PUBLIC /root/.conan/data/gsl-lite/0.37.0/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/)
# target_link_libraries(xpu_cmodel PUBLIC gsl::gsl-lite)
# if(NOT APPLE AND NOT MSVC)
#   target_link_libraries(xpu_cmodel PRIVATE rt)
# endif()
set_target_properties(xpu_cmodel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_executable(xpu_cmodel_cli src/xpu_cmodel.cpp ../shared_memory.cpp)
# target_link_libraries(xpu_cmodel_cli PUBLIC xpu_cmodel)
# set_target_properties(xpu_cmodel_cli PROPERTIES POSITION_INDEPENDENT_CODE ON
#   OUTPUT_NAME "nncase.simulator.xpu.c")
# install(TARGETS xpu_cmodel_cli COMPONENT nncase-runtime)

# add_definitions(-D__riscv_vector)
foreach(LIBRARY_PATH ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
    # Construct the full path to the library file
    set(CRTI_LIB_PATH "${LIBRARY_PATH}/crti.o")

    # Check if the library file exists
    if(EXISTS ${CRTI_LIB_PATH})
        message("crti.o Library found at: ${CRTI_LIB_PATH}")
        break()  # Stop searching if found
    endif()
endforeach()

if(NOT CRTI_LIB_PATH)
    message("crti.o Library not found")
endif()

foreach(LIBRARY_PATH ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
    # Construct the full path to the library file
    set(CRT1_LIB_PATH "${LIBRARY_PATH}/crt1.o")

    # Check if the library file exists
    if(EXISTS ${CRT1_LIB_PATH})
        message("crt1.o Library found at: ${CRT1_LIB_PATH}")
        break()  # Stop searching if found
    endif()
endforeach()

if(NOT CRT1_LIB_PATH)
    message("crt1.o Library not found")
endif()

function(add_test name test_path)
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O1)
  endif()
  add_link_options(-no-pie -nostartfiles -fPIC -fno-stack-protector -static -Wl,-e,_Z6_startPN6nncase7runtime3xpu19hardware_context_mtEPNS1_15runtime_util_mtEPNS1_19nncase_method_tableEPPh)
  add_executable(${name} "${test_path}/main.cpp")
  target_link_libraries(${name} xpu_cmodel ${CRT1_LIB_PATH} ${CRTI_LIB_PATH})
endfunction(add_test)

# add_test(demo1 tests/demo1)
# add_test(demo2 tests/demo2)
if(NOT MSVC AND NOT APPLE)
  add_test(decoder_layer tests/decoder_layer)
  add_test(norm tests/norm)
  add_test(embed tests/embed)
  add_test(head tests/head)
  add_test(unary tests/unary)
endif(NOT MSVC AND NOT APPLE)