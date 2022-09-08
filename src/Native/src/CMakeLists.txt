cmake_minimum_required (VERSION 3.13)

set(SRCS 
        api.cpp
        object.cpp
        shape.cpp
        tensor.cpp
        type.cpp
        value.cpp)

add_library(nncasebase OBJECT ${SRCS})
target_compile_definitions(nncasebase PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
if (DEFAULT_BUILTIN_RUNTIMES)
    target_compile_definitions(nncasebase PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
endif ()
set_property(TARGET nncasebase PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(runtime)
add_subdirectory(kernels)

if (NOT BUILDING_RUNTIME)
    add_executable(nncasetest test.cpp)
    target_link_libraries(nncasetest PRIVATE nncaseruntime)
endif()

install(TARGETS nncasebase EXPORT nncaseTargets
  COMPONENT nncase-runtime
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(EXPORT nncaseTargets DESTINATION lib/cmake/nncase)

configure_file(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/nncaseConfig.cmake.in nncaseConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseConfig.cmake DESTINATION lib/cmake/nncase)    
