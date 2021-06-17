from conans import ConanFile, CMake


class nncaseConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "cmake_find_package", "cmake_paths"

    def requirements(self):
        self.requires('flatbuffers/1.12.0')
        self.requires('fmt/7.1.3')
        self.requires('gsl-lite/0.37.0')
        self.requires('gtest/1.10.0')
        self.requires('lyra/1.5.0')
        self.requires('magic_enum/0.7.0')
        self.requires('nlohmann_json/3.9.1')
        self.requires('opencv/4.5.1')
        self.requires('protobuf/3.13.0')
        self.requires('pybind11/2.6.1')
        self.requires('xtensor/0.21.5')
        self.requires('mpark-variant/1.4.0')
        self.requires('spdlog/1.8.2')

    def build_requirements(self):
        self.build_requires("flatc/1.12.0")

    def configure(self):
        self.options["opencv"].contrib = False
        self.options["opencv"].with_webp = False
        self.options["opencv"].with_openexr = False
        self.options["opencv"].with_eigen = False
        self.options["opencv"].with_quirc = False
        self.options["opencv"].dnn = False
        if self.settings.os == 'Linux':
            self.options["opencv"].with_gtk = False
