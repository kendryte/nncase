cmake_minimum_required (VERSION 3.16)

set(MODELS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cpu/mnist.kmodel)

set(SRCS models.cpp)
add_library(bench_models EXCLUDE_FROM_ALL STATIC ${SRCS})
target_link_libraries(bench_models PUBLIC gsl::gsl-lite)

if(MSVC)
    target_compile_options(bench_models PRIVATE /wd4244) #inja
else()
    target_include_directories(bench_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_source_files_properties(
        models.cpp
        PROPERTIES
        OBJECT_DEPENDS ${MODELS_SRCS}
    )
endif()