cmake_minimum_required (VERSION 3.16)

set(MODELS_SRCS cpu/mnist.kmodel
                cpu/mobilenet_v2.kmodel)

set(MODELS_DEPEND)
foreach(SRC ${MODELS_SRCS})
    set(MODELS_DEPEND "${MODELS_DEPEND}${CMAKE_CURRENT_SOURCE_DIR}/${SRC};")
endforeach()

if(MSVC)
    add_library(bench_models_rc OBJECT Resource.rc)
    target_include_directories(bench_models_rc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    set_source_files_properties(
        Resource.rc
        PROPERTIES
        OBJECT_DEPENDS "${MODELS_DEPEND}"
    )
endif()

set(SRCS models.cpp)
add_library(bench_models STATIC ${SRCS})
target_link_libraries(bench_models PUBLIC gsl::gsl-lite)

if(MSVC)
    target_compile_options(bench_models PRIVATE /wd4244) #inja
else()
    target_include_directories(bench_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_source_files_properties(
        models.cpp
        PROPERTIES
        OBJECT_DEPENDS "${MODELS_DEPEND}"
    )
endif()