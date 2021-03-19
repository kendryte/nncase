cmake_minimum_required (VERSION 3.8)

set(SRCS interpreter.cpp
         error.cpp
         runtime_loader.cpp
         runtime_module.cpp
         runtime_tensor.cpp
         allocator.cpp)

if (BUILDING_RUNTIME)
    add_library(runtime STATIC ${SRCS})
    target_include_directories(runtime PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(runtime PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant CONAN_PKG::xtensor)
    target_link_libraries(runtime PRIVATE kernels CONAN_PKG::fmt)
    set_target_properties(runtime PROPERTIES
                                  OUTPUT_NAME "nncase.runtime_static")

    if (BUILD_SHARED_LIBS)
        add_library(runtime_shared SHARED ${SRCS})
        target_link_libraries(runtime_shared PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant CONAN_PKG::xtensor)
        target_link_libraries(runtime_shared PRIVATE kernels CONAN_PKG::fmt)
        set_target_properties(runtime_shared PROPERTIES
                                             OUTPUT_NAME "nncase.runtime")
        set_property(TARGET runtime_shared PROPERTY POSITION_INDEPENDENT_CODE ON)
        install(TARGETS runtime_shared)
    endif()
else()
    add_library(simulator OBJECT ${SRCS})
    target_include_directories(simulator PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(simulator PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::mpark-variant CONAN_PKG::xtensor)
    target_link_libraries(simulator PRIVATE kernels CONAN_PKG::fmt)
    target_compile_definitions(simulator PUBLIC -DNNCASE_DLL -DNNCASE_SIMULATOR)
    set_property(TARGET simulator PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(stackvm)
