from conans import ConanFile, CMake, tools


class nncaseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "cmake_find_package", "cmake_paths"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "runtime": [True, False],
        "tests": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "runtime": False,
        "tests": False
    }

    def requirements(self):
        self.requires('flatbuffers/2.0.0')
        self.requires('fmt/7.1.3')
        self.requires('gsl-lite/0.37.0')
        self.requires('lyra/1.5.0')
        self.requires('magic_enum/0.7.0')
        self.requires('nlohmann_json/3.9.1')
        self.requires('opencv/4.5.1')
        self.requires('protobuf/3.17.1')
        self.requires('pybind11/2.6.1')
        self.requires('xtensor/0.21.5')
        self.requires('mpark-variant/1.4.0')
        self.requires('spdlog/1.8.2')
        if self.options.tests:
            self.requires('gtest/1.10.0')

    def build_requirements(self):
        pass

    def configure(self):
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
        if self.options.runtime:
            cmake.definitions["CMAKE_CXX_STANDARD"] = 14
        return cmake

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()