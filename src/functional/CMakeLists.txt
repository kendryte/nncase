cmake_minimum_required (VERSION 3.13)

set(SRCS dummy.cpp)

if ((NOT BUILDING_RUNTIME) OR NNCASE_FUNCTIONAL_IMPL_PLATFORM_HEADER)
    list(APPEND SRCS ops.cpu.cpp)
endif()

if (BUILDING_RUNTIME)
    add_library(functional OBJECT ${SRCS})
    target_include_directories(functional PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(functional PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
    set_property(TARGET functional PROPERTY POSITION_INDEPENDENT_CODE ON)
    install(TARGETS functional EXPORT nncasefunctionalTargets)

    add_library(nncasefunctional STATIC dummy.cpp)
    target_link_libraries(nncasefunctional PRIVATE functional runtime)
    target_link_libraries(nncasefunctional PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
    set_target_properties(nncasefunctional PROPERTIES
                                        OUTPUT_NAME "nncase.functional")
    INSTALL (
        DIRECTORY ${CMAKE_SOURCE_DIR}/include/nncase/functional/
        DESTINATION include/nncase/functional/
        FILES_MATCHING PATTERN "*.h")

    install(TARGETS nncasefunctional EXPORT nncasefunctionalTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
    
    install(EXPORT nncasefunctionalTargets
            DESTINATION lib/cmake/nncasefunctional)
    
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../../cmake/nncasefunctionalConfig.cmake.in nncasefunctionalConfig.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/nncasefunctionalConfig.cmake DESTINATION lib/cmake/nncasefunctional)
else()
    add_library(functional OBJECT ${SRCS})
    target_include_directories(functional PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(functional PUBLIC gsl::gsl-lite mpark_variant::mpark_variant)
    set_property(TARGET functional PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()