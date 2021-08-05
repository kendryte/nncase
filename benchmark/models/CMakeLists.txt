cmake_minimum_required (VERSION 3.16)

set(MODELS_SRCS cpu/mnist.kmodel)

set(SRCS models.cpp)
add_library(bench_models STATIC ${SRCS})
#target_link_libraries(bench_models PRIVATE )

if(MSVC)
    target_compile_options(bench_models PRIVATE /wd4244) #inja
else()
    set_source_files_properties(
        models.cpp
        PROPERTIES
        OBJECT_DEPENDS ${MODELS_SRCS}
    )
endif()