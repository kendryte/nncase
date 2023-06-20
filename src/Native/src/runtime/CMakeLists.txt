cmake_minimum_required (VERSION 3.13)

set(SRCS buffer.cpp
		 datatypes.cpp
		 error.cpp
		 host_buffer.cpp
		 host_runtime_tensor.cpp
         interpreter.cpp
         runtime_loader.cpp
         runtime_module.cpp
		 runtime_function.cpp
		 section.cpp
		 type_serializer.cpp
		 runtime_tensor.cpp
         dump_manager.cpp)

if ((NOT BUILDING_RUNTIME) OR DEFAULT_SHARED_RUNTIME_TENSOR_PLATFORM_IMPL)
    list(APPEND SRCS host_allocator.cpp)
endif()

if (BUILDING_RUNTIME)
    add_library(runtime OBJECT ${SRCS})
    target_include_directories(runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(runtime PUBLIC gsl::gsl-lite)
    target_link_libraries(runtime PRIVATE kernels)
    if (DEFAULT_BUILTIN_RUNTIMES)
      target_compile_definitions(runtime PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
    endif()
    if (ENABLE_DUMP_MANAGER)
      target_compile_definitions(runtime PUBLIC -DNNCASE_DUMP_MANAGER)
    endif()
    set_property(TARGET runtime PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS runtime EXPORT nncaseruntimeTargets)

    add_library(nncaseruntime STATIC dummy.cpp)
    target_link_libraries(nncaseruntime PRIVATE nncasebase kernels runtime runtime_stackvm)
    target_link_libraries(nncaseruntime PUBLIC gsl::gsl-lite)
    set_target_properties(nncaseruntime PROPERTIES
                                        OUTPUT_NAME "Nncase.Runtime.Native")
    install(TARGETS nncaseruntime EXPORT nncaseruntimeTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )

    install(EXPORT nncaseruntimeTargets
            DESTINATION lib/cmake/nncaseruntime)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/nncaseruntimeConfig.cmake.in nncaseruntimeConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseruntimeConfig.cmake DESTINATION lib/cmake/nncaseruntime)
else()
    add_library(simulator OBJECT ${SRCS})
    target_include_directories(simulator PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(simulator PUBLIC gsl::gsl-lite)
    target_link_libraries(simulator PRIVATE kernels)
    target_compile_definitions(simulator PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    if (DEFAULT_BUILTIN_RUNTIMES)
      target_compile_definitions(simulator PRIVATE -DNNCASE_DEFAULT_BUILTIN_RUNTIMES)
    endif()
    if (ENABLE_DUMP_MANAGER)
      target_compile_definitions(simulator PUBLIC -DNNCASE_DUMP_MANAGER)
    endif()
    set_property(TARGET simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    add_library(nncaseruntime SHARED dummy.cpp)
    target_link_libraries(nncaseruntime PRIVATE nncasebase kernels simulator compiler simulator_stackvm fmt::fmt)
    target_link_libraries(nncaseruntime PUBLIC gsl::gsl-lite)
    set_target_properties(nncaseruntime PROPERTIES
                                        OUTPUT_NAME "Nncase.Runtime.Native")

    install(TARGETS nncaseruntime EXPORT nncaseTargets
        COMPONENT nncase-runtime
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
    install(EXPORT nncaseTargets
        DESTINATION lib/cmake/nncase)

    if (WIN32)
        install(FILES ${CONAN_NETHOST_ROOT}/bin/nethost.dll
            COMPONENT nncase-runtime
            DESTINATION bin)
    endif()

    configure_file(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/nncaseConfig.cmake.in nncaseConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncaseConfig.cmake DESTINATION lib/cmake/nncase)
endif()

add_subdirectory(stackvm)
