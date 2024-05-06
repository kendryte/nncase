# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

from conans import ConanFile, CMake, tools


class nncaseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "cmake_find_package", "cmake_paths"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "runtime": [True, False],
        "tests": [True, False],
        "halide": [True, False],
        "python": [True, False],
        "vulkan_runtime": [True, False],
        "openmp": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "runtime": False,
        "tests": False,
        "halide": True,
        "python": True,
        "vulkan_runtime": False,
        "openmp": True
    }
    
    def imports(self):
        if self.settings.os == 'Windows':
            self.copy("nethost.dll", "bin", "bin")
            self.copy("ortki.dll", "bin", "bin")

    def requirements(self):
        self.requires('gsl-lite/0.37.0')
        self.requires('hkg/0.0.1')
        if self.options.tests:
            self.requires('gtest/1.10.0')
            self.requires('ortki/0.0.2')
            self.requires('rapidjson/1.1.x')

        if self.options.python:
            self.requires('pybind11/2.12.0')

        if not self.options.runtime:
            self.requires('abseil/20220623.1')
            self.requires('nethost/6.0.11')
            self.requires('fmt/7.1.3')
            self.requires('magic_enum/0.7.0')
            self.requires('spdlog/1.8.2')
            self.requires('inja/3.2.0')
            if self.options.tests:
                self.requires('gtest/1.10.0')

        if (not self.options.runtime) or self.options.vulkan_runtime:
            self.requires('vulkan-headers/1.2.182')
            self.requires('vulkan-loader/1.2.182')

    def build_requirements(self):
        pass

    def configure(self):
        min_cppstd = "17" if self.options.runtime else "20"
        tools.check_min_cppstd(self, min_cppstd)

        if self.settings.os == 'Windows':
            self.settings.compiler.toolset = 'ClangCL'

        if self.settings.arch not in ("x86_64",):
            self.options.halide = False
            
        if not self.options.runtime:
            if self.settings.os == 'Windows':
                self.options["nethost"].shared = True

        if (not self.options.runtime) or self.options.vulkan_runtime:
            if self.settings.os == 'Linux':
                self.options["vulkan-loader"].with_wsi_xcb = False
                self.options["vulkan-loader"].with_wsi_xlib = False
                self.options["vulkan-loader"].with_wsi_wayland = False
                self.options["vulkan-loader"].with_wsi_directfb = False

        if self.options.tests:
            self.options["ortki"].shared = True

    def cmake_configure(self):
        cmake = CMake(self)
        cmake.definitions['BUILDING_RUNTIME'] = self.options.runtime
        cmake.definitions['ENABLE_OPENMP'] = self.options.openmp
        cmake.definitions['ENABLE_VULKAN_RUNTIME'] = self.options.vulkan_runtime
        cmake.definitions['ENABLE_HALIDE'] = self.options.halide
        cmake.definitions['BUILD_PYTHON_BINDING'] = self.options.python
        cmake.definitions['BUILD_TESTING'] = self.options.tests
        if self.options.runtime:
            cmake.definitions["CMAKE_CXX_STANDARD"] = 17
        cmake.configure()
        return cmake

    def build(self):
        cmake = self.cmake_configure()
        cmake.build()
