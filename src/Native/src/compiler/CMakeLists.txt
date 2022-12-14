cmake_minimum_required (VERSION 3.13)

set(SRCS compiler.cpp)

if (NOT BUILDING_RUNTIME)
    add_library(compiler OBJECT ${SRCS})
    target_include_directories(compiler PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(compiler PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
    target_link_libraries(compiler PRIVATE nethost::nethost absl::absl)
    target_compile_definitions(compiler PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    set_property(TARGET compiler PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
