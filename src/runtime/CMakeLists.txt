cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         error.cpp
         runtime_loader.cpp
         runtime_module.cpp
         runtime_tensor.cpp
         allocator.cpp)

if (BUILDING_RUNTIME)
    add_library(runtime OBJECT ${SRCS})
    target_include_directories(runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(runtime PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant)
    target_link_libraries(runtime PRIVATE kernels)
    if (DEFAULT_BUILTIN_RUNTIMES)
        target_compile_definitions(runtime PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
    endif ()
    set_property(TARGET runtime PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime EXPORT nncaseruntimeTargets)

    add_library(nncaseruntime STATIC dummy.cpp)
    target_link_libraries(nncaseruntime PRIVATE kernels runtime runtime_stackvm)
    target_link_libraries(nncaseruntime PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant)
    set_target_properties(nncaseruntime PROPERTIES
                                        OUTPUT_NAME "nncase.runtime")
    install(TARGETS nncaseruntime EXPORT nncaseruntimeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    install(EXPORT nncaseruntimeTargets
            DESTINATION lib/cmake/nncaseruntime)
    
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../../cmake/nncaseruntimeConfig.cmake.in nncaseruntimeConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseruntimeConfig.cmake DESTINATION lib/cmake/nncaseruntime)

else()
    add_library(simulator OBJECT ${SRCS})
    target_include_directories(simulator PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(simulator PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant)
    target_link_libraries(simulator PRIVATE kernels CONAN_PKG::fmt)
    target_compile_definitions(simulator PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    if (DEFAULT_BUILTIN_RUNTIMES)
        target_compile_definitions(simulator PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
    endif ()
    set_property(TARGET simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(stackvm)
