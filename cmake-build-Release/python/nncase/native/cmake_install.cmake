# Install script for directory: /home/curio/project/k230/rebuild-ir/nncase/python/nncase/native

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xnncase-runtimex" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so"
         RPATH "\$ORIGIN")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE MODULE FILES "/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Release/lib/_nncase.cpython-38-x86_64-linux-gnu.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so"
         OLD_RPATH "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib:/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib:/home/curio/.conan/data/abseil/20220623.1/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/lib:/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/lib:/home/curio/.conan/data/spdlog/1.8.2/_/_/package/2f8d6866984cf9c9262a45a6675ee1fab9a81fe2/lib:/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/56ab386b83e5f1276a7374d6a809c723c88dc6aa/lib:/home/curio/.conan/data/fmt/7.1.3/_/_/package/1291f461f6832a5b3098e2156f727f267fd98612/lib:/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Release/lib:"
         NEW_RPATH "\$ORIGIN")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/_nncase.cpython-38-x86_64-linux-gnu.so")
    endif()
  endif()
endif()

