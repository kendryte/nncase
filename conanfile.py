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
        "python": [True, False],
        "vulkan_runtime": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "runtime": False,
        "tests": False,
        "python": True,
        "vulkan_runtime": False
    }
    
    def imports(self):
        if self.settings.os == 'Windows':
            self.copy("nethost.dll", "bin", "bin")
            self.copy("ortki.dll", "bin", "bin")

    def requirements(self):
        if self.options.tests:
            self.requires('gtest/1.10.0')
            self.requires('ortki/0.0.3')
            self.requires('rapidjson/cci.20230929')

        if self.options.python:
            self.requires('pybind11/2.11.1')

        if not self.options.runtime:
            self.requires('nethost/7.0.5')
            self.requires('fmt/7.1.3')
            self.requires('nlohmann_json/3.9.1')

    def build_requirements(self):
        pass

    def configure(self):
        min_cppstd = "20"
        tools.check_min_cppstd(self, min_cppstd)

        if self.settings.os == 'Windows':
            self.settings.compiler.toolset = 'ClangCL'
            
        if not self.options.runtime:
            if self.settings.os == 'Windows':
                self.options["nethost"].shared = True

        if self.options.tests:
            self.options["ortki"].shared = True

    def cmake_configure(self):
        cmake = CMake(self)
        cmake.definitions['BUILDING_RUNTIME'] = self.options.runtime
        cmake.definitions['ENABLE_VULKAN_RUNTIME'] = self.options.vulkan_runtime
        cmake.definitions['BUILD_PYTHON_BINDING'] = self.options.python
        cmake.definitions['BUILD_TESTING'] = self.options.tests
        cmake.configure()
        return cmake

    def build(self):
        cmake = self.cmake_configure()
        cmake.build()
