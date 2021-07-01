cmake_minimum_required (VERSION 3.8)

set(SRCS plugin_loader.cpp)

add_library(plugin OBJECT ${SRCS})
target_link_libraries(plugin PRIVATE targets fmt::fmt)
target_link_libraries(plugin PUBLIC gsl::gsl-lite xtensor::xtensor mpark_variant::mpark_variant)
target_compile_definitions(plugin PRIVATE -DNNCASE_DLL)
set_target_properties(plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)
