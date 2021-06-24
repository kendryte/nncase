cmake_minimum_required(VERSION 3.13)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules)
find_package (Git)
execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --always --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DESC
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT DEFINED NNCASE_VERSION_MAJOR)
    set(NNCASE_VERSION_MAJOR 1)
endif()
if(NOT DEFINED NNCASE_VERSION_MINOR)
    set(NNCASE_VERSION_MINOR 0)
endif()
if(NOT DEFINED NNCASE_VERSION_PATCH)
    set(NNCASE_VERSION_PATCH 0)
endif()
if(NOT DEFINED NNCASE_VERSION_SUFFIX)
    set(NNCASE_VERSION_SUFFIX ${GIT_DESC})
endif()

if (NOT PACKAGE_VERSION)
    set(PACKAGE_VERSION
        "${NNCASE_VERSION_MAJOR}.${NNCASE_VERSION_MINOR}.${NNCASE_VERSION_PATCH}${NNCASE_VERSION_SUFFIX}")
endif()

project(nncase
  VERSION ${NNCASE_VERSION_MAJOR}.${NNCASE_VERSION_MINOR}.${NNCASE_VERSION_PATCH}
  LANGUAGES C CXX ASM)

include(cmake/conan.cmake)

conan_check()
conan_add_remote(NAME sunnycase URL https://conan.sunnycase.moe INDEX 0)
conan_add_remote(NAME bincrafts URL https://api.bintray.com/conan/bincrafters/public-conan)

option(ENABLE_OPENMP "openmp support" ON)
option(BUILD_TESTING "Build test programs" OFF)

if (ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    message("Enable OpenMP")
endif ()

if (BUILDING_RUNTIME)
    option(BUILD_PYTHON_BINDING "Build python binding" ON)
    option(ENABLE_K210_RUNTIME "Enable k210 runtime" OFF)
    option(DEFAULT_BUILTIN_RUNTIMES "Use default builtin runtimes" ON)
    option(DEFAULT_SHARED_RUNTIME_TENSOR_PLATFORM_IMPL "Use default shared memory platform impl" ON)

    conan_cmake_run(CONANFILE conanfile-runtime.txt
                    BASIC_SETUP CMAKE_TARGETS
                    SETTINGS compiler.cppstd=14
                    BUILD missing)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
    
    set(NNCASE_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
    set(NNCASE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
    set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/third_party)
    set(CMAKE_CXX_STANDARD 14)
    set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)
    
    if (MSVC)
        add_definitions(/D_CRT_SECURE_NO_WARNINGS /DNOMINMAX)
        add_compile_options(/wd4267 /wd4251 /FC /utf-8 /W3 /WX)
    else()
        add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-multichar -Wno-missing-field-initializers -Wno-unused-function -Wno-type-limits)
        if (APPLE)
            add_compile_options(-Wno-four-char-constants -Wno-sometimes-uninitialized)
        else ()
            add_compile_options(-Wno-maybe-uninitialized)
        endif()
    endif()
    
    include_directories(${NNCASE_MAIN_INCLUDE_DIR})
    include_directories(${NNCASE_INCLUDE_DIR})
    
    add_subdirectory(include/nncase)
    add_subdirectory(src/kernels)
    add_subdirectory(src/runtime)
    
    # Python binding
    if(BUILD_PYTHON_BINDING)
        add_subdirectory(python/nncaseruntime/native)
    endif()
        
    install(DIRECTORY ${NNCASE_INCLUDE_DIR}/nncase
        DESTINATION include
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*version.h"
        PATTERN "CMakeFiles" EXCLUDE
        )
    
    install(DIRECTORY include/nncase/kernels
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
    
    install(DIRECTORY include/nncase/runtime
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

    # Modules
    add_subdirectory(modules/k210)
else()
    conan_cmake_run(CONANFILE conanfile.py
                    BASIC_SETUP CMAKE_TARGETS
                    SETTINGS compiler.cppstd=20
                    BUILD missing)
    include(${CMAKE_BINARY_DIR}/conan_paths.cmake)
    set(CMAKE_SKIP_RPATH OFF)

    find_package (Protobuf REQUIRED)
    include(cmake/flatc.cmake)

    set(NNCASE_MAIN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
    set(NNCASE_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
    set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/third_party)
    set(CMAKE_CXX_STANDARD 20)
    set_property(GLOBAL PROPERTY POSITION_INDEPENDENT_CODE ON)
    if (APPLE)
        set(CMAKE_MACOSX_RPATH TRUE)
        set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
        set(CMAKE_INSTALL_NAME_DIR "@rpath")
    else ()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
    endif()
    
    if (MSVC)
        add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX)
        add_compile_options(/wd4267 /wd4251 /FC /utf-8 /W3 /WX)
        set(PYBIND11_CPP_STANDARD "/std:c++latest")
    else()
        add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-multichar -Wno-missing-field-initializers -Wno-unused-function -Wno-type-limits)
        set(PYBIND11_CPP_STANDARD "-std=c++20")
        if (APPLE)
            add_compile_options(-Wno-four-char-constants -Wno-sometimes-uninitialized)
        else ()
            add_compile_options(-Wno-maybe-uninitialized)
        endif()
    endif()
    
    include_directories(${NNCASE_MAIN_INCLUDE_DIR})
    include_directories(${NNCASE_INCLUDE_DIR})
    
    add_subdirectory(include/nncase)
    add_subdirectory(src/data)
    add_subdirectory(src/ir)
    add_subdirectory(src/importer)
    add_subdirectory(src/schedule)
    add_subdirectory(src/evaluator)
    add_subdirectory(src/transforms)
    add_subdirectory(src/codegen)
    add_subdirectory(src/kernels)
    add_subdirectory(src/runtime)
    add_subdirectory(src/targets)
    add_subdirectory(src/plugin)
    add_subdirectory(src/nncase)
    add_subdirectory(src/cli)

    if(BUILD_TESTING)
        add_subdirectory(tests/kernels)
    endif()
    
    # Python binding
    add_subdirectory(python/nncase/native)
    
    # Thrid party
    add_subdirectory(third_party/onnx)
    
    install(DIRECTORY include/nncase
        DESTINATION include
        COMPONENT nncase-headers
        FILES_MATCHING
        PATTERN "*.def"
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.td"
        PATTERN "*.inc"
        PATTERN "LICENSE.TXT"
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
        )
    
    install(DIRECTORY python/nncase
        DESTINATION python
        COMPONENT nncase-python
        FILES_MATCHING
        PATTERN "*.py"
        )
    
    # Targets
    add_subdirectory(targets/cpu)
    add_subdirectory(targets/k210)

    # Modules
    add_subdirectory(modules/k210)

endif()
