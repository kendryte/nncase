cmake_minimum_required (VERSION 3.13)

set(SRCS 
        api.cpp
        object.cpp
        shape.cpp
        tensor.cpp
        type.cpp
        value.cpp)

add_library(nncasebase OBJECT ${SRCS})
if(BUILDING_RUNTIME)
    target_compile_definitions(nncasebase PUBLIC -DNNCASE_DLL)
else()
    target_compile_definitions(nncasebase PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
endif()

if (DEFAULT_BUILTIN_RUNTIMES)
    target_compile_definitions(nncasebase PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
endif ()
set_property(TARGET nncasebase PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(nncasebase PUBLIC gsl::gsl-lite)

add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(kernels)

if (NOT BUILDING_RUNTIME)
    add_executable(nncasetest test.cpp)
    target_link_libraries(nncasetest PRIVATE nncaseruntime)
    if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" AND MSVC)
        set_target_properties(nncasetest PROPERTIES LINK_FLAGS "/PROFILE")
    endif()

    add_executable(nncasetest_cli test_cli.cpp)
    target_link_libraries(nncasetest_cli PRIVATE nncaseruntime)

    add_executable(nncasetest_dynamic_cli test_dynamic_cli.cpp)
    target_link_libraries(nncasetest_dynamic_cli PRIVATE nncaseruntime)
    install(TARGETS nncasetest_cli nncasetest_dynamic_cli COMPONENT nncase-tools)
endif()

if(BUILDING_RUNTIME)
    install(TARGETS nncasebase EXPORT nncaseruntimeTargets)
else()
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
endif()