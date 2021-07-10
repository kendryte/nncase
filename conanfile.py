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
    generators = "cmake", "cmake_find_package", "cmake_paths"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "runtime": [True, False],
        "tests": [True, False],
        "halide": [True, False],
        "python": [True, False],
        "vulkan": [True, False],
        "openmp": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "runtime": False,
        "tests": False,
        "halide": False,
        "python": True,
        "vulkan": True,
        "openmp": True
    }

    def requirements(self):
        self.requires('gsl-lite/0.37.0')
        self.requires('mpark-variant/1.4.0')
        self.requires('hkg/0.0.1')
        if self.options.tests:
            self.requires('gtest/1.10.0')

        if self.options.python:
            self.requires('pybind11/2.6.1')

        if self.options.vulkan:
            self.requires('vulkan-headers/1.2.182')
            self.requires('vulkan-loader/1.2.172')

        if not self.options.runtime:
            self.requires('flatbuffers/2.0.0')
            self.requires('fmt/7.1.3')
            self.requires('lyra/1.5.0')
            self.requires('magic_enum/0.7.0')
            self.requires('nlohmann_json/3.9.1')
            self.requires('opencv/4.5.1')
            self.requires('protobuf/3.17.1')
            self.requires('xtensor/0.21.5')
            self.requires('spdlog/1.8.2')

    def build_requirements(self):
        pass

    def configure(self):
        min_cppstd = "14" if self.options.runtime else "20"
        tools.check_min_cppstd(self, min_cppstd)

        if not self.options.runtime:
            self.options["opencv"].contrib = False
            self.options["opencv"].with_webp = False
            self.options["opencv"].with_openexr = False
            self.options["opencv"].with_eigen = False
            self.options["opencv"].with_quirc = False
            self.options["opencv"].dnn = False
            self.options["flatbuffers"].options_from_context = False
            self.options["xtensor"].xsimd = False
            if self.settings.os == 'Linux':
                self.options["opencv"].with_gtk = False

    def cmake_configure(self):
        cmake = CMake(self)
        cmake.definitions['BUILDING_RUNTIME'] = self.options.runtime
        cmake.definitions['ENABLE_OPENMP'] = self.options.openmp
        cmake.definitions['ENABLE_VULKAN'] = self.options.vulkan
        cmake.definitions['BUILD_PYTHON_BINDING'] = self.options.python
        if self.options.runtime:
            cmake.definitions["CMAKE_CXX_STANDARD"] = 14
        return cmake

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
