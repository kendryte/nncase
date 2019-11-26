cmake_minimum_required(VERSION 3.8)
project("nncase")

# Download automatically, you can also just copy the conan.cmake file
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.14/conan.cmake"
                 "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

include(${CMAKE_BINARY_DIR}/conan.cmake)

if (NOT NNCASE_TARGET)
    message(FATAL_ERROR "Please define NNCASE_TARGET")
else()
    add_definitions(-DNNCASE_TARGET=${NNCASE_TARGET})
endif()

conan_check()
conan_add_remote(NAME bincrafts URL https://api.bintray.com/conan/bincrafters/public-conan)
conan_add_remote(NAME sunnycase URL https://conan.sunnycase.moe)
conan_cmake_run(CONANFILE conanfile.txt
                BASIC_SETUP CMAKE_TARGETS
                SETTINGS compiler.cppstd=17
                BUILD missing)
include(ExternalProject)
find_package(Protobuf REQUIRED)
include(cmake/flatc.cmake)

set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/third_party)
set(CMAKE_CXX_STANDARD 17)
 
if (MSVC)
    add_definitions(/D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS /DNOMINMAX /FC)
else()
    add_compile_options(-Wno-multichar)
endif()

add_subdirectory(src/cli)
add_subdirectory(src/common)
add_subdirectory(src/ir)
add_subdirectory(src/transforms)
add_subdirectory(src/importer)
add_subdirectory(src/data)
add_subdirectory(src/scheduler)
add_subdirectory(src/evaluator)
add_subdirectory(src/codegen)
add_subdirectory(src/runtime)
