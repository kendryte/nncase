# Install script for directory: /home/curio/project/k230/rebuild-ir/nncase/src/Native/src/runtime

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
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so"
         RPATH "$ORIGIN")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Debug/lib/libNncase.Runtime.Native.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so"
         OLD_RPATH "/home/curio/.conan/data/hkg/0.0.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib:/home/curio/.conan/data/pybind11/2.6.1/_/_/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/lib:/home/curio/.conan/data/abseil/20220623.1/_/_/package/a16cd4cc16fd73821dc08bebe2bb68c62e7143b2/lib:/home/curio/.conan/data/nethost/6.0.11/_/_/package/4db1be536558d833e52e862fd84d64d75c2b3656/lib:/home/curio/.conan/data/spdlog/1.8.2/_/_/package/4a17d92125ff1ce8bad17b0d8cbed2a8baea8470/lib:/home/curio/.conan/data/vulkan-loader/1.2.182/_/_/package/3076560b39e981740fdd1dc149763deb1b2c620e/lib:/home/curio/.conan/data/fmt/7.1.3/_/_/package/a16cd4cc16fd73821dc08bebe2bb68c62e7143b2/lib:"
         NEW_RPATH "$ORIGIN")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libNncase.Runtime.Native.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xnncase-runtimex" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase/nncaseTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase/nncaseTargets.cmake"
         "/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Debug/src/Native/src/runtime/CMakeFiles/Export/lib/cmake/nncase/nncaseTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase/nncaseTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase/nncaseTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase" TYPE FILE FILES "/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Debug/src/Native/src/runtime/CMakeFiles/Export/lib/cmake/nncase/nncaseTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase" TYPE FILE FILES "/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Debug/src/Native/src/runtime/CMakeFiles/Export/lib/cmake/nncase/nncaseTargets-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/nncase" TYPE FILE FILES "/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Debug/src/Native/src/runtime/nncaseConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/curio/project/k230/rebuild-ir/nncase/cmake-build-Debug/src/Native/src/runtime/stackvm/cmake_install.cmake")

endif()

