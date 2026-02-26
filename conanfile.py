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

from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd, cross_building
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import apply_conandata_patches, copy, export_conandata_patches, get, rmdir

class nncaseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "runtime": [True, False],
        "tests": [True, False],
        "python": [True, False],
        # "vulkan_runtime": [True, False],
        "python_root": ["ANY"],
        "dump_mem": [True, False],

    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "runtime": False,
        "tests": False,
        "python": True,
        # "vulkan_runtime": False,
        "python_root": "",
        "dump_mem": False
    }

    @property
    def _min_cppstd(self):
        return 17

    def layout(self):
        cmake_layout(self, build_folder="build")

    def requirements(self):
        self.requires('gsl-lite/0.37.0')
        self.requires('nlohmann_json/3.11.3')
        if self.options.tests:
            self.requires('gtest/1.10.0')
            self.requires('ortki/0.0.4')

        if self.options.python:
            self.requires('pybind11/2.13.6')

        if not self.options.runtime:
            self.requires('nethost/8.0.8')
            self.requires('fmt/7.1.3')

        # if not self.options.runtime or self.options.tests:
        #     self.requires('nlohmann_json/3.11.3')

        # if (not self.options.runtime) or self.options.vulkan_runtime:
        #     self.requires('vulkan-headers/1.2.182')
        #     self.requires('vulkan-loader/1.2.182')

    def build_requirements(self):
        pass

    def configure(self):
        if not self.options.runtime:
            if self.settings.os == 'Windows' and self.settings.build_type == 'Debug':
                self.options["nethost"].shared = True

        if self.options.tests:
            self.options["ortki"].shared = True
            self.options["date"].header_only = True

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._min_cppstd)


    def generate(self):
        tc = CMakeToolchain(self, generator="Ninja")
        tc.variables['BUILDING_RUNTIME'] = self.options.runtime
        tc.variables['BUILD_PYTHON_BINDING'] = self.options.python
        tc.variables['BUILD_TESTING'] = self.options.tests
        tc.variables['ENABLE_DUMP_MEM'] = self.options.dump_mem
        if self.options.get_safe("python_root", default="") != "":
            tc.variables['Python3_ROOT_DIR'] = str(self.options.python_root).replace('\\', '/')
        if self.options.runtime:
            tc.presets_prefix += "-runtime";
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
