cmake_minimum_required (VERSION 3.21)

set(SRCS compiler.cpp)

if (NOT BUILDING_RUNTIME)
    add_library(compiler OBJECT ${SRCS})
    target_include_directories(compiler PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(compiler PUBLIC gsl::gsl-lite)
    target_link_libraries(compiler PRIVATE nethost::nethost nlohmann_json::nlohmann_json)
    target_compile_definitions(compiler PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    set_property(TARGET compiler PROPERTY POSITION_INDEPENDENT_CODE ON)

    if (DOTNET_INIT_FOR_CONFIG)
        target_compile_definitions(compiler PRIVATE -DNNCASE_DOTNET_INIT_FOR_CONFIG)
    endif ()
endif()
