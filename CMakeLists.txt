cmake_minimum_required(VERSION 3.15)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)

if(NOT DEFINED NNCASE_VERSION)
    set(NNCASE_VERSION "2.0.0")
endif()

if(DEFINED ENV{NNCASE_VERSION_SUFFIX})
    set(NNCASE_VERSION_SUFFIX $ENV{NNCASE_VERSION_SUFFIX})
endif()

if(NOT DEFINED NNCASE_VERSION_SUFFIX)
    find_package (Git)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tag
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_TAG
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if ("v${NNCASE_VERSION}" STREQUAL "${GIT_TAG}")
        set(NNCASE_VERSION_SUFFIX "")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --always --dirty
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE GIT_DESC
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(NNCASE_VERSION_SUFFIX "-${GIT_DESC}")
    endif()
endif()

if (NOT PACKAGE_VERSION)
    set(PACKAGE_VERSION
        "${NNCASE_VERSION}${NNCASE_VERSION_SUFFIX}")
endif()

project(nncase
  VERSION ${NNCASE_VERSION}
  LANGUAGES C CXX ASM)

find_package(nlohmann_json REQUIRED)
include_directories(${nlohmann_json_INCLUDE_DIRS})

option(ENABLE_OPENMP "OpenMP support" OFF)
option(ENABLE_HALIDE "halide kernels support" ON)
option(DOTNET_INIT_FOR_CONFIG "Initialize dotnet from runtimeconfig" OFF)
option(BUILD_PYTHON_BINDING "Build python binding" ON)
option(BUILD_CSHARP_BINDING "Build csharp binding" ON)
option(BUILD_BENCHMARK "Build benchmark programs" ON)
option(BUILD_TESTING "Build test programs" OFF)
option(ENABLE_DUMP_MANAGER "Enable dump manager" OFF)
option(ENABLE_RVV "Some kernel impl by rvv" OFF)
option(ENABLE_DUMP_MEM "Dump mem usage" OFF)

if (BUILDING_RUNTIME)
    # option(ENABLE_VULKAN_RUNTIME "Enable Vulkan runtime" OFF)
    option(ENABLE_K210_RUNTIME "Enable k210 runtime" OFF)
    option(DEFAULT_BUILTIN_RUNTIMES "Use default builtin runtimes" ON)
    option(DEFAULT_SHARED_RUNTIME_TENSOR_PLATFORM_IMPL "Use default shared memory platform impl" ON)
endif()

if(ENABLE_DUMP_MEM)
    add_definitions(-DDUMP_MEM)
endif()

include(cmake/dependencies.cmake)

set(NNCASE_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/src/Native/include)
set(NNCASE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/Native/include)
set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/third_party)
set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SKIP_RPATH OFF)

if (BUILDING_RUNTIME)
    if (MSVC)
        add_definitions(/D_CRT_SECURE_NO_WARNINGS /DNOMINMAX)
        add_compile_options(/wd4267 /wd4251 /wd4244 /FC /utf-8 /W3 /WX /wd4297 -Wno-unused-function -Wno-unused-command-line-argument)
    else()
        add_compile_options(-Wall -Wextra -Werror -Wno-multichar -Wno-missing-field-initializers -Wno-unused-function -Wno-type-limits)
        if (APPLE)
            add_compile_options(-Wno-four-char-constants -Wno-sometimes-uninitialized)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_options(-Wno-uninitialized -Wno-unused-private-field)
        else()
            add_compile_options(-Wno-maybe-uninitialized -Wno-attributes)
            if (CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 9)
                add_compile_options(-Wno-unused-private-field)
            endif()
        endif()
    endif()

    include_directories(${NNCASE_MAIN_INCLUDE_DIR})
    include_directories(${NNCASE_INCLUDE_DIR})

    add_subdirectory(src/Native/include/nncase)
    #    add_subdirectory(src/Native/src/kernels)
    #    add_subdirectory(src/Native/src/runtime)
    add_subdirectory(src/Native/src)
#    add_subdirectory(src/Native/src/functional)
    if(BUILD_BENCHMARK)
#        add_subdirectory(benchmark)
    endif()

    # Python binding
    if(BUILD_PYTHON_BINDING)
#        add_subdirectory(python/nncaseruntime/native)
    endif()

    install(DIRECTORY ${NNCASE_INCLUDE_DIR}/nncase
        DESTINATION include
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*.def"
        PATTERN "*.h"
        PATTERN "*version.h"
        PATTERN "CMakeFiles" EXCLUDE
        )

    install(DIRECTORY src/Native/include/nncase/
            DESTINATION include/nncase
            COMPONENT nncase-headers
            FILES_MATCHING
            PATTERN "*.def"
            PATTERN "*.h"
            PATTERN "*version.h"
            PATTERN "CMakeFiles" EXCLUDE
            )

    install(DIRECTORY src/Native/include/nncase/kernels
        DESTINATION include/nncase
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*.def"
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.td"
        PATTERN "*.inc"
        PATTERN "LICENSE.TXT"
        )

    install(DIRECTORY src/Native/include/nncase/runtime
        DESTINATION include/nncase
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*.def"
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.td"
        PATTERN "*.inc"
        PATTERN "LICENSE.TXT"
        )
else()

    if (APPLE)
        set(CMAKE_MACOSX_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "@loader_path")
        set(CMAKE_INSTALL_NAME_DIR "@rpath")
    else ()
        set(CMAKE_INSTALL_RPATH "$ORIGIN")
    endif()

    if (MSVC)
        add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX)
        add_compile_options(/wd4267 /wd4251 /wd4244 /FC /utf-8 /W3 /WX -Wno-unused-function -Wno-unused-command-line-argument)
        set(PYBIND11_CPP_STANDARD "/std:c++latest")
    else()
        add_compile_options(-fvisibility=hidden)
        add_compile_options(-Wall -Wextra -Werror -Wno-multichar -Wno-missing-field-initializers -Wno-unused-function -Wno-type-limits -Wno-unused-local-typedefs -Wno-sign-compare)
        if (APPLE)
            add_compile_options(-Wno-four-char-constants -Wno-sometimes-uninitialized -Wno-deprecated)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_options(-Wno-uninitialized -Wno-gnu-zero-variadic-macro-arguments)
        else ()
            add_compile_options(-Wno-maybe-uninitialized -Wno-deprecated-copy)
            add_link_options(-Wl,--exclude-libs,ALL)
        endif()
    endif()

    include_directories(${NNCASE_MAIN_INCLUDE_DIR})
    include_directories(${NNCASE_INCLUDE_DIR})

    add_subdirectory(src/Native/include/nncase)
    add_subdirectory(src/Native/src)

    # Python binding
    if(BUILD_PYTHON_BINDING)
        add_subdirectory(python/nncase/native)
    endif()

    # Csharp binding
    if(BUILD_CSHARP_BINDING)
        #add_subdirectory(csharp)
    endif()

    # Thrid party
    # add_subdirectory(third_party/onnx)

    install(DIRECTORY src/Native/include/nncase
        DESTINATION include
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*.def"
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.td"
        PATTERN "*.inc"
        PATTERN "LICENSE.TXT"
        PATTERN "*.h"
        )

    install(DIRECTORY ${NNCASE_INCLUDE_DIR}/nncase
        DESTINATION include
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*.def"
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.td"
        PATTERN "*.inc"
        PATTERN "CMakeFiles" EXCLUDE
        PATTERN "config.h" EXCLUDE
        PATTERN "*.h"
        )

    install(DIRECTORY python/nncase
        DESTINATION python
        COMPONENT nncase-python
        FILES_MATCHING
        PATTERN "*.py"
        )

    # Targets
    #add_subdirectory(targets/cpu)
    #add_subdirectory(targets/k210)
    #add_subdirectory(targets/vulkan)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests/kernels)
endif()

# Modules

#add_subdirectory(modules/vulkan)
