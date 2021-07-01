cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         error.cpp
         runtime_loader.cpp
         runtime_module.cpp
         runtime_tensor.cpp
         runtime_tensor_impl.cpp
         host_runtime_tensor.cpp
         allocator.cpp)

if ((NOT BUILDING_RUNTIME) OR DEFAULT_SHARED_RUNTIME_TENSOR_PLATFORM_IMPL)
    list(APPEND SRCS shared_runtime_tensor.platform.cpp)
endif()

if (BUILDING_RUNTIME)
    add_library(runtime OBJECT ${SRCS})
    target_include_directories(runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(runtime PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
    target_link_libraries(runtime PRIVATE kernels)
    if (DEFAULT_BUILTIN_RUNTIMES)
        target_compile_definitions(runtime PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
    endif ()
    set_property(TARGET runtime PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime EXPORT nncaseruntimeTargets)

    add_library(nncaseruntime STATIC dummy.cpp)
    target_link_libraries(nncaseruntime PRIVATE kernels runtime runtime_stackvm)
    target_link_libraries(nncaseruntime PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
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
    target_link_libraries(simulator PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
    target_link_libraries(simulator PRIVATE kernels fmt::fmt)
    target_compile_definitions(simulator PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    if (DEFAULT_BUILTIN_RUNTIMES)
        target_compile_definitions(simulator PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
    endif ()
    set_property(TARGET simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(stackvm)
